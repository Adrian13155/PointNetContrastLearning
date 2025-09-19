
from __future__ import annotations
import argparse
import os
import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from contrast_dataset import ContrastiveModelNetDataset
from baseline_dataset import ModelNetDataset  # for test/eval loader
from contrast_model import ContrastPointNet
from baseline_model import feature_transform_regularizer

from datetime import datetime

import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_args():
    p = argparse.ArgumentParser("Contrastive + Classification (PointNet)")
    p.add_argument("--dataset", type=str,default="/data/cjj/projects/pointnet.pytorch/data/modelnet40_ply_hdf5_2048",
                   help="ModelNet40 HDF5 dir (contains train_files.txt/test_files.txt)")
    p.add_argument("--batchSize", type=int, default=32)
    p.add_argument("--nepoch", type=int, default=200)
    p.add_argument("--npoints", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--step_size", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--feature_transform", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_cpu", action="store_true", default=False)
    p.add_argument("--gpu_id", type=str, default="1")
    # augmentation & contrastive
    p.add_argument("--rotation_mode", type=str, default="xyz", choices=["xyz", "yaw", "none"])
    p.add_argument("--no_jitter", action="store_true", default=False)
    p.add_argument("--lambda_con", type=float, default=0.1, help="weight for contrastive loss")
    p.add_argument("--lambda_region", type=float, default=0.1, help="weight for region contrastive loss")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--use_region_contrast", action="store_true", default=True ,help="use region contrastive learning")
    
    # graphcut segmentation parameters
    p.add_argument("--use_graphcut_segmentation", action="store_true", default=True, help="use graphcut segmentation for grouping")
    p.add_argument("--graphcut_k", type=int, default=8, help="number of neighbors for graphcut")
    p.add_argument("--graphcut_c", type=float, default=0.5, help="merging threshold for graphcut")
    p.add_argument("--graphcut_min_size", type=int, default=20, help="minimum region size for graphcut")

    p.add_argument('--exp_name', type=str, default='GroupContrast', help='experiment name')
    p.add_argument('--save_dir', help='日志保存路径', default='/data/cjj/projects/pointnet.pytorch/Contrast_exp/experiment', type=str)
    return p.parse_args()


def make_loaders(args):
    train_ds = ContrastiveModelNetDataset(
        root=args.dataset, split="train", npoints=args.npoints,
        rotation_mode=args.rotation_mode, jitter=(not args.no_jitter),
        center_unit=True,
        use_graphcut_segmentation=args.use_graphcut_segmentation,
        graphcut_k=args.graphcut_k,
        graphcut_c=args.graphcut_c,
        graphcut_min_size=args.graphcut_min_size
    )
    test_ds = ModelNetDataset(
        root=args.dataset, split="test", npoints=args.npoints,
        augment=False, center_unit=True, rotation_mode="none"
    )
    train_loader = DataLoader(train_ds, batch_size=args.batchSize, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batchSize, shuffle=False,
                             num_workers=args.workers)
    return train_loader, test_loader

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    B, D = z1.shape
    assert z2.shape == (B, D)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    sim = torch.matmul(z, z.t())  # (2B, 2B)
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim / temperature

    pos_indices = torch.arange(2 * B, device=z.device)
    pos_indices = (pos_indices + B) % (2 * B)
    pos_sim = sim[torch.arange(2 * B, device=z.device), pos_indices]

    sim_exp = torch.exp(sim.masked_fill(mask, float('-inf')))
    denom = sim_exp.sum(dim=1)
    loss = -torch.log(torch.exp(pos_sim) / denom)
    return loss.mean()

def region_contrastive_loss(point_feat1: torch.Tensor, point_feat2: torch.Tensor, 
                           segments: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    计算分组对比损失 - 超快速向量化版本
    
    Args:
        point_feat1: (B, D, N) 第一个视图的点级别特征
        point_feat2: (B, D, N) 第二个视图的点级别特征  
        segments: (B, N) 每个点的分组ID
        temperature: 温度参数
    
    Returns:
        loss: 分组对比损失
    """
    B, D, N = point_feat1.shape
    device = point_feat1.device
    
    # 归一化特征
    feat1_norm = F.normalize(point_feat1, dim=1)  # (B, D, N)
    feat2_norm = F.normalize(point_feat2, dim=1)  # (B, D, N)
    
    total_loss = 0.0
    total_pairs = 0
    
    for b in range(B):
        # 当前点云的特征和分组
        f1 = feat1_norm[b]  # (D, N)
        f2 = feat2_norm[b]  # (D, N)
        seg = segments[b]    # (N,)
        
        # 计算相似度矩阵
        sim = torch.matmul(f1.t(), f2) / temperature  # (N, N)
        
        # 获取唯一的分组ID
        unique_segments = torch.unique(seg)
        
        for seg_id in unique_segments:
            # 找到属于当前分组的点
            mask = (seg == seg_id)
            seg_indices = torch.where(mask)[0]
            
            if len(seg_indices) < 2:  # 分组中至少需要2个点
                continue
            
            # 完全向量化计算
            n_seg = len(seg_indices)
            seg_sim = sim[seg_indices][:, seg_indices]  # (n_seg, n_seg)
            
            # 创建上三角掩码
            triu_mask = torch.triu(torch.ones(n_seg, n_seg, device=device), diagonal=1).bool()
            
            if triu_mask.sum() == 0:
                continue
                
            # 正样本相似度
            pos_sim = seg_sim[triu_mask]  # (n_pairs,)
            
            # 负样本相似度（与其他分组的点）
            neg_mask = (seg != seg_id)
            if neg_mask.sum() == 0:
                continue
                
            neg_indices = torch.where(neg_mask)[0]
            neg_sim = sim[seg_indices][:, neg_indices]  # (n_seg, n_neg)
            
            # 完全向量化损失计算
            pos_exp = torch.exp(pos_sim)  # (n_pairs,)
            
            # 对每个正样本点，计算与所有负样本的相似度
            row_indices = torch.where(triu_mask)[0]  # 正样本的行索引
            neg_exp_sum = torch.exp(neg_sim[row_indices]).sum(dim=1)  # (n_pairs,)
            
            # 计算损失
            denominator = pos_exp + neg_exp_sum
            losses = -torch.log(pos_exp / denominator)
            
            total_loss += losses.sum()
            total_pairs += len(losses)
    
    return total_loss / max(total_pairs, 1)

@torch.no_grad()
def evaluate(net: ContrastPointNet, loader, device, feature_transform=False):
    net.eval()
    correct, total = 0, 0
    reg_accum = 0.0
    
    # 创建验证进度条
    eval_pbar = tqdm(loader, desc="[Eval]", leave=False, ncols=80)
    
    for pts, label in eval_pbar:
        pts = pts.to(device).transpose(2, 1)
        label = label.to(device)
        logp, _, _, _, _, trans_feat = net(pts)
        pred = logp.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum().item()
        total += label.size(0)
        if feature_transform and trans_feat is not None:
            reg_accum += float(feature_transform_regularizer(trans_feat).item())
        
        # 更新验证进度条
        current_acc = correct / total if total > 0 else 0.0
        eval_pbar.set_postfix({'Acc': f'{current_acc:.4f}'})
    
    return (correct / total if total > 0 else 0.0), (reg_accum / max(1, len(loader)))

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    now = datetime.now()
    formatted_time = now.strftime("%m-%d_%H:%M")
    save_dir = os.path.join(args.save_dir, f'{formatted_time}_{args.exp_name}')
    
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")

    if not args.use_cpu:
        torch.cuda.set_device(int(args.gpu_id))

    train_loader, test_loader = make_loaders(args)

    num_classes = 40
    net = ContrastPointNet(k=num_classes, feature_transform=args.feature_transform).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(save_dir,exist_ok=True)
    logger = get_logger(os.path.join(save_dir,f'run_{args.exp_name}.log'))
    logger.info(f"Arguments: {args}")
    logger.info(f"model params: {sum(p.numel() for p in net.parameters() )/1e6} M")
    logger.info(f"Network Structure: {str(net)}")
    best_acc = 0.0

    for epoch in range(args.nepoch):
        net.train()
        running = {"loss": 0.0, "ce": 0.0, "con": 0.0, "region_con": 0.0, "reg": 0.0}
        correct, total = 0, 0

        # 创建训练进度条
        train_pbar = tqdm(train_loader, 
                         desc=f"Epoch {epoch+1:03d}/{args.nepoch:03d} [Train]",
                         leave=False,
                         ncols=100)
        
        for batch_data in train_pbar:
            if len(batch_data) == 4:  # 包含分组信息
                v1, v2, y, segments = batch_data
                v1 = v1.to(device).transpose(2, 1)  # (B, 3, N)
                v2 = v2.to(device).transpose(2, 1)
                y = y.to(device)
                if segments is not None:
                    segments = segments.to(device)
            else:  # 不包含分组信息
                v1, v2, y = batch_data
                v1 = v1.to(device).transpose(2, 1)  # (B, 3, N)
                v2 = v2.to(device).transpose(2, 1)
                y = y.to(device)
                segments = None

            optimizer.zero_grad()

            # two views
            logp1, z1, h1, point_feat1, _, trans_feat1 = net(v1)
            logp2, z2, h2, point_feat2, _, trans_feat2 = net(v2)

            # classification loss
            ce1 = F.nll_loss(logp1, y)
            ce2 = F.nll_loss(logp2, y)
            ce = 0.5 * (ce1 + ce2)

            # feature transform regularization
            reg = 0.0
            if args.feature_transform and (trans_feat1 is not None) and (trans_feat2 is not None):
                reg = 0.5 * (feature_transform_regularizer(trans_feat1) +
                             feature_transform_regularizer(trans_feat2)) * 0.001

            # global contrastive loss
            con = info_nce_loss(z1, z2, temperature=args.temperature)

            # region contrastive loss
            region_con = 0.0
            if args.use_region_contrast and segments is not None:
                # 使用真正的点级特征 point_feat1, point_feat2 (B, 1024, N)
                region_con = region_contrastive_loss(point_feat1, point_feat2, segments, temperature=args.temperature)

            # total loss
            loss = ce + args.lambda_con * con + args.lambda_region * region_con + (reg if isinstance(reg, torch.Tensor) else 0.0)
            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.item())
            running["ce"] += float(ce.item())
            running["con"] += float(con.item())
            running["region_con"] += float(region_con if isinstance(region_con, torch.Tensor) else region_con)
            running["reg"] += float(reg if not isinstance(reg, torch.Tensor) else reg.item())

            pred = logp1.data.max(1)[1]
            correct += pred.eq(y.data).cpu().sum().item()
            total += y.size(0)
            
            # 更新进度条显示
            # 计算当前已处理的batch数
            current_batch = train_pbar.n
            current_loss = running["loss"] / max(1, current_batch)
            current_acc = correct / total if total > 0 else 0.0
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        train_loss = running["loss"] / max(1, len(train_loader))
        train_acc = correct / total if total > 0 else 0.0

        # evaluation of single view,
        test_acc, reg_eval = evaluate(net, test_loader, device, feature_transform=args.feature_transform)
        scheduler.step()

        if args.use_region_contrast:
            logger.info(f"Epoch [{epoch+1:03d}/{args.nepoch:03d}] "
                       f"Train Loss: {train_loss:.4f} (CE {running['ce']/max(1,len(train_loader)):.4f}, "
                       f"Con {running['con']/max(1,len(train_loader)):.4f}, "
                       f"RegionCon {running['region_con']/max(1,len(train_loader)):.4f}, "
                       f"Reg {running['reg']/max(1,len(train_loader)):.6f}) "
                       f"| Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        else:
            logger.info(f"Epoch [{epoch+1:03d}/{args.nepoch:03d}] "
                       f"Train Loss: {train_loss:.4f} (CE {running['ce']/max(1,len(train_loader)):.4f}, "
                       f"Con {running['con']/max(1,len(train_loader)):.4f}, "
                       f"Reg {running['reg']/max(1,len(train_loader)):.6f}) "
                       f"| Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # save last & best
        torch.save(net.state_dict(), os.path.join(args.save_dir, "contrast_model_last.pth"))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), os.path.join(args.save_dir, "contrast_model_best.pth"))

    logger.info(f"Training done. Best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()

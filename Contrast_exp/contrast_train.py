
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

from contrast_dataset import ContrastiveModelNetDataset
from baseline_dataset import ModelNetDataset  # for test/eval loader
from contrast_model import ContrastPointNet
from baseline_model import feature_transform_regularizer


def parse_args():
    p = argparse.ArgumentParser("Contrastive + Classification (PointNet)")
    p.add_argument("--dataset", type=str, required=True,
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
    p.add_argument("--save_dir", type=str, default="./checkpoints_contrast")
    p.add_argument("--use_cpu", action="store_true")

    # augmentation & contrastive
    p.add_argument("--rotation_mode", type=str, default="xyz", choices=["xyz", "yaw", "none"])
    p.add_argument("--no_jitter", action="store_true")
    p.add_argument("--lambda_con", type=float, default=0.1, help="weight for contrastive loss")
    p.add_argument("--temperature", type=float, default=0.07)

    return p.parse_args()


def make_loaders(args):
    train_ds = ContrastiveModelNetDataset(
        root=args.dataset, split="train", npoints=args.npoints,
        rotation_mode=args.rotation_mode, jitter=(not args.no_jitter),
        center_unit=True
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


@torch.no_grad()
def evaluate(net: ContrastPointNet, loader, device, feature_transform=False):
    net.eval()
    correct, total = 0, 0
    reg_accum = 0.0
    for pts, label in loader:
        pts = pts.to(device).transpose(2, 1)
        label = label.to(device)
        logp, _, _, trans_feat = net(pts)
        pred = logp.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum().item()
        total += label.size(0)
        if feature_transform and trans_feat is not None:
            reg_accum += float(feature_transform_regularizer(trans_feat).item())
    return (correct / total if total > 0 else 0.0), (reg_accum / max(1, len(loader)))


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")

    train_loader, test_loader = make_loaders(args)

    num_classes = 40
    net = ContrastPointNet(k=num_classes, feature_transform=args.feature_transform).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.nepoch):
        net.train()
        running = {"loss": 0.0, "ce": 0.0, "con": 0.0, "reg": 0.0}
        correct, total = 0, 0

        for v1, v2, y in train_loader:
            v1 = v1.to(device).transpose(2, 1)  # (B, 3, N)
            v2 = v2.to(device).transpose(2, 1)
            y = y.to(device)

            optimizer.zero_grad()

            # two views
            logp1, z1, _, trans_feat1 = net(v1)
            logp2, z2, _, trans_feat2 = net(v2)

            # classifi
            ce1 = F.nll_loss(logp1, y)
            ce2 = F.nll_loss(logp2, y)
            ce = 0.5 * (ce1 + ce2)

            # (optional)
            reg = 0.0
            if args.feature_transform and (trans_feat1 is not None) and (trans_feat2 is not None):
                reg = 0.5 * (feature_transform_regularizer(trans_feat1) +
                             feature_transform_regularizer(trans_feat2)) * 0.001

            # contrast
            con = info_nce_loss(z1, z2, temperature=args.temperature)

            loss = ce + args.lambda_con * con + (reg if isinstance(reg, torch.Tensor) else 0.0)
            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.item())
            running["ce"] += float(ce.item())
            running["con"] += float(con.item())
            running["reg"] += float(reg if not isinstance(reg, torch.Tensor) else reg.item())

            pred = logp1.data.max(1)[1]
            correct += pred.eq(y.data).cpu().sum().item()
            total += y.size(0)

        train_loss = running["loss"] / max(1, len(train_loader))
        train_acc = correct / total if total > 0 else 0.0

        # evaluation of single view,
        test_acc, reg_eval = evaluate(net, test_loader, device, feature_transform=args.feature_transform)
        scheduler.step()

        print(f"Epoch [{epoch+1:03d}/{args.nepoch:03d}] "
              f"Train Loss: {train_loss:.4f} (CE {running['ce']/max(1,len(train_loader)):.4f}, "
              f"Con {running['con']/max(1,len(train_loader)):.4f}, "
              f"Reg {running['reg']/max(1,len(train_loader)):.6f}) "
              f"| Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # save last & best
        torch.save(net.state_dict(), os.path.join(args.save_dir, "contrast_model_last.pth"))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), os.path.join(args.save_dir, "contrast_model_best.pth"))

    print(f"Training done. Best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()

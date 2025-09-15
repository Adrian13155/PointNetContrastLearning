from __future__ import print_function
import argparse
import os
import random
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from baseline_dataset import ModelNetDataset
from baseline_model import PointNetCls, feature_transform_regularizer


def parse_args():
    parser = argparse.ArgumentParser("PointNet-Classification Baseline")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to ModelNet40 HDF5 dataset dir (with train_files.txt/test_files.txt)")
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--nepoch", type=int, default=200)
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--feature_transform", action="store_true", help="Enable feature transform (STNkd)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_baseline")
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--rotation_mode", type=str, default="xyz", choices=["xyz", "none"],
                        help="Rotation augmentation policy for training set")
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()


def make_loaders(args) -> Tuple[data.DataLoader, data.DataLoader]:
    train_ds = ModelNetDataset(
        root=args.dataset, split="train",
        npoints=args.npoints,
        augment=(not args.no_aug),
        center_unit=True,
        rotation_mode=args.rotation_mode
    )
    test_ds = ModelNetDataset(
        root=args.dataset, split="test",
        npoints=args.npoints,
        augment=False,
        center_unit=True,
        rotation_mode="none"
    )
    train_loader = data.DataLoader(train_ds, batch_size=args.batchSize, shuffle=True,
                                   num_workers=args.workers, drop_last=True)
    test_loader = data.DataLoader(test_ds, batch_size=args.batchSize, shuffle=False,
                                  num_workers=args.workers)
    return train_loader, test_loader


def evaluate(net, loader, device, feature_transform=False):
    net.eval()
    correct, total = 0, 0
    reg_loss_accum = 0.0
    with torch.no_grad():
        for pts, label in loader:
            pts = pts.to(device)  # (B, N, 3)
            label = label.to(device)
            pts = pts.transpose(2, 1)  # (B, 3, N)
            pred, trans, trans_feat = net(pts)
            loss_reg = feature_transform_regularizer(trans_feat) if (feature_transform and trans_feat is not None) else 0.0
            reg_loss_accum += float(loss_reg) if isinstance(loss_reg, torch.Tensor) else loss_reg
            pred_choice = pred.data.max(1)[1]
            correct += pred_choice.eq(label.data).cpu().sum().item()
            total += label.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc, reg_loss_accum / max(1, len(loader))


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")

    train_loader, test_loader = make_loaders(args)

    num_classes = 40  # ModelNet40
    net = PointNetCls(k=num_classes, feature_transform=args.feature_transform).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.nepoch):
        net.train()
        running_loss, running_ce, running_reg = 0.0, 0.0, 0.0
        correct, total = 0, 0

        for pts, label in train_loader:
            pts = pts.to(device)         # (B, N, 3)
            label = label.to(device)
            pts = pts.transpose(2, 1)    # (B, 3, N)

            optimizer.zero_grad()
            pred, trans, trans_feat = net(pts)
            loss = F.nll_loss(pred, label)

            if args.feature_transform and trans_feat is not None:
                reg = feature_transform_regularizer(trans_feat) * 0.001
                loss = loss + reg
                running_reg += float(reg.item())

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_ce += float(F.nll_loss(pred, label).item())
            pred_choice = pred.data.max(1)[1]
            correct += pred_choice.eq(label.data).cpu().sum().item()
            total += label.size(0)

        train_acc = correct / total if total > 0 else 0.0
        train_loss = running_loss / max(1, len(train_loader))

        # evaluate
        test_acc, reg_eval = evaluate(net, test_loader, device, feature_transform=args.feature_transform)

        # lr step
        scheduler.step()

        print(f"Epoch [{epoch+1:03d}/{args.nepoch:03d}] "
              f"Train Loss: {train_loss:.4f} (CE {running_ce/max(1,len(train_loader)):.4f}, Reg {running_reg/max(1,len(train_loader)):.6f}) "
              f"| Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # save last and best
        torch.save(net.state_dict(), os.path.join(args.save_dir, "cls_model_last.pth"))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), os.path.join(args.save_dir, "cls_model_best.pth"))

    print(f"Training done. Best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()

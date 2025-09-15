
from __future__ import annotations
import os
import os.path as osp
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

# Reuse baseline dataset utilities (HDF5 loading + augmentation helpers)
import baseline_dataset as bd


class ContrastiveModelNetDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 npoints: int = 1024,
                 rotation_mode: str = "xyz",
                 jitter: bool = True,
                 center_unit: bool = True):
        super().__init__()
        self.root = root
        self.split = split
        self.npoints = npoints
        self.rotation_mode = rotation_mode
        self.jitter = jitter
        self.center_unit = center_unit

        if split in ("train", "trainval"):
            list_file = osp.join(root, "train_files.txt")
        elif split == "test":
            list_file = osp.join(root, "test_files.txt")
        else:
            raise ValueError(f"Unknown split: {split}")

        self.points, self.labels = bd._load_h5_files(list_file, root)

    def __len__(self):
        return len(self.points)

    def _prep(self, pts: np.ndarray) -> np.ndarray:
        if self.center_unit:
            pts = bd._center_and_scale_unit_sphere(pts)
        N = pts.shape[0]
        if N > self.npoints:
            idx = np.random.choice(N, self.npoints, replace=False)
            pts = pts[idx, :]
        elif N < self.npoints:
            pad_idx = np.random.choice(N, self.npoints - N, replace=True)
            pts = np.concatenate([pts, pts[pad_idx]], axis=0)
        return pts

    def _augment_once(self, pts: np.ndarray) -> np.ndarray:
        if self.split in ("train", "trainval"):
            if self.rotation_mode == "xyz":
                pts = bd.random_rotate_xyz(pts)
            elif self.rotation_mode == "yaw":
                pts = bd.random_rotate_yaw(pts)
            elif self.rotation_mode == "none":
                pass
            else:
                raise ValueError(f"Unknown rotation_mode: {self.rotation_mode}")
            if self.jitter:
                pts = bd.jitter_points(pts, sigma=0.01, clip=0.02)
        return pts

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pts = self.points[idx].copy()  # (N, 3)
        label = int(self.labels[idx])
        base = self._prep(pts)

        v1 = self._augment_once(base.copy())
        v2 = self._augment_once(base.copy())

        v1 = torch.from_numpy(v1).float()
        v2 = torch.from_numpy(v2).float()
        y = torch.tensor(label, dtype=torch.long)
        return v1, v2, y

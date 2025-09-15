
from __future__ import print_function
import os
import os.path as osp
import h5py
import numpy as np
import torch
import torch.utils.data as data
from typing import Tuple, List, Optional


def _load_h5_files(list_txt: str, root: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read HDF5 files listed in train_files.txt / test_files.txt (as in ModelNet40 HDF5 release).
    Each .h5 is expected to have datasets: 'data' (B, N, 3) and 'label' (B, 1).
    """
    with open(list_txt, "r") as f:
        rel_paths = [line.strip() for line in f if line.strip()]
    all_data, all_label = [], []
    for rel in rel_paths:
        # allow either absolute paths in txt or filenames placed under root
        h5_path = rel if osp.isabs(rel) else osp.join(root, osp.basename(rel))
        if not osp.exists(h5_path):
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")
        with h5py.File(h5_path, "r") as h5f:
            data = h5f["data"][:]    # (B, N, 3)
            label = h5f["label"][:]  # (B, 1)
            all_data.append(data)
            all_label.append(label)
    data = np.concatenate(all_data, axis=0).astype(np.float32)
    label = np.concatenate(all_label, axis=0).astype(np.int64).squeeze(1)
    return data, label


def _center_and_scale_unit_sphere(x: np.ndarray) -> np.ndarray:

    # x: (N, 3)

    centroid = np.mean(x, axis=0, keepdims=True)
    x = x - centroid
    m = np.max(np.linalg.norm(x, axis=1))
    if m > 0:
        x = x / m
    return x


def _rotation_matrix_xyz(ax: float, ay: float, az: float) -> np.ndarray:
    """
    随机 xyz 轴随机角度"增强（依次绕 X、Y、Z 旋转随机角度）。
    """
    sx, cx = np.sin(ax), np.cos(ax)
    sy, cy = np.sin(ay), np.cos(ay)
    sz, cz = np.sin(az), np.cos(az)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=np.float32)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float32)
    return Rx @ Ry @ Rz


def random_rotate_xyz(points: np.ndarray) -> np.ndarray:
    ax = np.random.uniform(-np.pi, np.pi)
    ay = np.random.uniform(-np.pi, np.pi)
    az = np.random.uniform(-np.pi, np.pi)
    R = _rotation_matrix_xyz(ax, ay, az)
    return points @ R.T


def jitter_points(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip).astype(np.float32)
    return points + noise


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 npoints: int = 1024,
                 augment: bool = True,
                 center_unit: bool = True,
                 rotation_mode: str = "xyz"):
        super().__init__()
        self.root = root
        self.split = split
        self.npoints = npoints
        self.augment = augment
        self.center_unit = center_unit
        self.rotation_mode = rotation_mode

        if split in ("train", "trainval"):
            list_file = osp.join(root, "train_files.txt")
        elif split == "test":
            list_file = osp.join(root, "test_files.txt")
        else:
            raise ValueError(f"Unknown split: {split}")

        self.points, self.labels = _load_h5_files(list_file, root)
        assert self.points.ndim == 3 and self.points.shape[2] == 3, f"Bad data shape: {self.points.shape}"
        assert len(self.points) == len(self.labels), "Data/label length mismatch"

    def __len__(self) -> int:
        return len(self.points)

    def _maybe_sample_n(self, x: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        if N == self.npoints:
            return x
        if N > self.npoints:
            idx = np.random.choice(N, self.npoints, replace=False)
            return x[idx, :]
        # pad by repeating random points
        pad_idx = np.random.choice(N, self.npoints - N, replace=True)
        return np.concatenate([x, x[pad_idx]], axis=0)

    def __getitem__(self, idx: int):
        pts = self.points[idx].copy()  # (N, 3)
        label = int(self.labels[idx])

        if self.center_unit:
            pts = _center_and_scale_unit_sphere(pts)

        if self.split in ("train", "trainval") and self.augment:
            if self.rotation_mode == "xyz":
                pts = random_rotate_xyz(pts)
            pts = jitter_points(pts, sigma=0.01, clip=0.02)

        pts = self._maybe_sample_n(pts)

        return torch.from_numpy(pts).float(), torch.tensor(label, dtype=torch.long)

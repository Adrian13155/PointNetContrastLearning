
from __future__ import annotations
import os
import os.path as osp
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

# Reuse baseline dataset utilities (HDF5 loading + augmentation helpers)
import baseline_dataset as bd


class DisjointSet:
    """并查集数据结构，用于图割算法"""
    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)

    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]  # 路径压缩
            u = self.parent[u]
        return u

    def union(self, u, v):
        u = self.find(u)
        v = self.find(v)
        if u == v:
            return False
        if self.size[u] < self.size[v]:
            u, v = v, u
        self.parent[v] = u
        self.size[u] += self.size[v]
        return True


def segment_pointcloud(points, k=8, c=0.5, min_size=20):
    """
    基于Felzenszwalb图分割思想的点云分割
    
    Args:
        points: Nx3 numpy array 点云坐标
        k: 每个点邻居数量
        c: 合并阈值参数，越大区域越少
        min_size: 最小区域大小，太小的区域会被合并
    
    Returns:
        labels: 每个点的分割类别
    """
    N = points.shape[0]
    
    if N < k + 1:
        # 如果点数太少，直接返回单个分组
        return np.zeros(N, dtype=int)

    # 1. 计算k近邻图边
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # 构建边列表：(点i, 点j, 权重)
    edges = []
    for i in range(N):
        for j in range(1, k+1):  # indices[i, 0]是自己，跳过
            neighbor = indices[i, j]
            dist = distances[i, j]
            edges.append((i, neighbor, dist))
    edges = np.array(edges, dtype=[('p1', int), ('p2', int), ('w', float)])

    # 2. 按权重排序边（距离越小越优先合并）
    edges.sort(order='w')

    # 3. 初始化并查集，每个点自己一个集合
    u = DisjointSet(N)

    # 4. 初始化阈值函数
    threshold = np.full(N, c)

    # 5. 依次遍历边，决定是否合并
    for edge in edges:
        a = u.find(edge['p1'])
        b = u.find(edge['p2'])
        if a != b:
            if edge['w'] <= threshold[a] and edge['w'] <= threshold[b]:
                merged = u.union(a, b)
                if merged:
                    a_new = u.find(a)
                    threshold[a_new] = edge['w'] + c / u.size[a_new]

    # 6. 合并小区域
    for edge in edges:
        a = u.find(edge['p1'])
        b = u.find(edge['p2'])
        if a != b:
            if u.size[a] < min_size or u.size[b] < min_size:
                u.union(a, b)

    # 7. 给每个点赋标签
    labels = np.zeros(N, dtype=int)
    label_map = {}
    label_id = 0
    for i in range(N):
        root = u.find(i)
        if root not in label_map:
            label_map[root] = label_id
            label_id += 1
        labels[i] = label_map[root]

    # print(f"分割得到 {label_id} 个区域")

    return labels


class ContrastiveModelNetDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 npoints: int = 1024,
                 rotation_mode: str = "xyz",
                 jitter: bool = True,
                 center_unit: bool = True,
                 use_graphcut_segmentation: bool = False,
                 graphcut_k: int = 8,
                 graphcut_c: float = 0.5,
                 graphcut_min_size: int = 20):
        super().__init__()
        self.root = root
        self.split = split
        self.npoints = npoints
        self.rotation_mode = rotation_mode
        self.jitter = jitter
        self.center_unit = center_unit
        self.use_graphcut_segmentation = use_graphcut_segmentation
        self.graphcut_k = graphcut_k
        self.graphcut_c = graphcut_c
        self.graphcut_min_size = graphcut_min_size

        if split in ("train", "trainval"):
            list_file = osp.join(root, "train_files.txt")
        elif split == "test":
            list_file = osp.join(root, "test_files.txt")
        else:
            raise ValueError(f"Unknown split: {split}")

        self.points, self.labels = bd._load_h5_files(list_file, root)

    def __len__(self):
        return len(self.points)

    def _prep(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理点云，返回处理后的点云和对应的分组标签
        
        Returns:
            pts_processed: 处理后的点云
            segments: 对应的分组标签
        """
        if self.center_unit:
            pts = bd._center_and_scale_unit_sphere(pts)
        
        N = pts.shape[0]
        segments = None
        
        # 如果需要分组，先对原始点云进行分组
        if self.use_graphcut_segmentation:
            segments = segment_pointcloud(pts, k=self.graphcut_k, 
                                       c=self.graphcut_c, 
                                       min_size=self.graphcut_min_size)
        
        # 处理点数
        if N > self.npoints:
            idx = np.random.choice(N, self.npoints, replace=False)
            pts = pts[idx, :]
            if segments is not None:
                segments = segments[idx]
        elif N < self.npoints:
            pad_idx = np.random.choice(N, self.npoints - N, replace=True)
            pts = np.concatenate([pts, pts[pad_idx]], axis=0)
            if segments is not None:
                segments = np.concatenate([segments, segments[pad_idx]], axis=0)
        
        return pts, segments

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pts = self.points[idx].copy()  # (N, 3)
        label = int(self.labels[idx])
        base, segments = self._prep(pts)

        v1 = self._augment_once(base.copy())
        v2 = self._augment_once(base.copy())

        v1 = torch.from_numpy(v1).float()
        v2 = torch.from_numpy(v2).float()
        y = torch.tensor(label, dtype=torch.long)
        
        # 返回分组信息
        if self.use_graphcut_segmentation:
            segments_tensor = torch.from_numpy(segments).long()
            return v1, v2, y, segments_tensor
        else:
            # 如果没有分组信息，返回None
            return v1, v2, y

if __name__ == "__main__":
    dataset = ContrastiveModelNetDataset(
        root="data/modelnet40_ply_hdf5_2048",
        split="train",
        npoints=1024,
        rotation_mode="xyz",
        jitter=True,
        use_graphcut_segmentation=True,
        graphcut_k=8,
        graphcut_c=0.5,
        graphcut_min_size=20,
    )
    DataLoader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True)
    # for v1, v2, y, segments in DataLoader:
    #     print(v1.shape, v2.shape, y.shape, segments.shape)
    #     break
    device = torch.device("cpu")
    for batch_data in DataLoader:
        if len(batch_data) == 4:  # 包含分组信息
            v1, v2, y, segments = batch_data
            v1 = v1.to(device).transpose(2, 1)  # (B, 3, N)
            v2 = v2.to(device).transpose(2, 1)
            y = y.to(device)
            print(v1.shape, v2.shape, y.shape, segments.shape)
            if segments is not None:
                segments = segments.to(device)
        break

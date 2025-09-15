
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_model import PointNetfeat, feature_transform_regularizer


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 1024, hid_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, h):
        z = self.net(h)
        z = F.normalize(z, dim=1)
        return z


class ContrastPointNet(nn.Module):

    def __init__(self, k: int = 40, feature_transform: bool = False):
        super().__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

        self.proj = ProjectionHead(in_dim=1024, hid_dim=256, out_dim=128)

    def forward(self, x):
        # x: (B, 3, N)
        h, trans, trans_feat = self.feat(x)  # h: (B,1024)
        xcls = F.relu(self.bn1(self.fc1(h)))
        xcls = F.relu(self.bn2(self.dropout(self.fc2(xcls))))
        logits = self.fc3(xcls)
        log_probs = F.log_softmax(logits, dim=1)

        z = self.proj(h)

        return log_probs, z, trans, trans_feat

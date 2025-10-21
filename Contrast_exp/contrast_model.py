
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
        point_feat, trans, trans_feat = self.feat(x)  # point_feat: (B, 1024, N)
        
        # 对点级特征进行最大池化得到全局特征
        h = torch.max(point_feat, 2, keepdim=True)[0]  # (B, 1024, 1)
        h = h.view(-1, 1024)  # (B, 1024)
        
        # 分类
        xcls = F.relu(self.bn1(self.fc1(h)))
        xcls = F.relu(self.bn2(self.dropout(self.fc2(xcls))))
        logits = self.fc3(xcls)
        log_probs = F.log_softmax(logits, dim=1)

        # 对比学习投影
        z = self.proj(h)

        return log_probs, z, h, point_feat, trans, trans_feat


if __name__ == "__main__":
    # 测试四元数PointNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 8
    num_points = 1024
    x = torch.randn(batch_size, 3, num_points).to(device)
    
    # 创建模型
    model = ContrastPointNet(k=40, feature_transform=True).to(device)
    
    # 前向传播
    log_probs, z, h, point_feat, trans, trans_feat = model(x)
    
    print(f"输出log_probs形状: {log_probs.shape}")
    print(f"输出z形状: {z.shape}")
    print(f"输出h形状: {h.shape}")
    print(f"输出point_feat形状: {point_feat.shape}")
    print(f"输出trans形状: {trans.shape}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
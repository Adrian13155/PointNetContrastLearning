

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.reqnn_pointnet_utils import PointNetEncoderREQNN
from util.reqnn_util import QLinear, QBatchNorm1d, QReLU, QDropout, Quaternion2Real

class ContrastPointNetREQNN(nn.Module):
    def __init__(self, k=40, normal_channel=False):
        super(ContrastPointNetREQNN, self).__init__()
        
        # 1. REQNN 特征编码器
        self.feat = PointNetEncoderREQNN(global_feat=True, feature_transform=False)
        
        # 2. 四元数全连接层 (Quaternion MLP)
        # 论文建议在四元数域继续处理特征
        self.q_fc1 = QLinear(1024, 512)
        self.q_bn1 = QBatchNorm1d(512)
        self.q_relu = QReLU()
        self.q_dropout = QDropout(p=0.4) 
        self.q_fc2 = QLinear(512, 256)
        self.q_bn2 = QBatchNorm1d(256)

        # 3. Quaternion2Real 模块
        # 在送入最终分类器前，将四元数特征转换为旋转不变的实数特征
        self.q2r = Quaternion2Real()

        # 4. 最终的分类器 (Task module - Standard Real-valued MLP)
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, k)

    def forward(self, x):
        # x: [B, 3, N]
        # 1. 提取全局四元数特征
        quat_global_feat, quat_local_feat, trans_feat = self.feat(x) # Shape: [B, 1024, 4]
        # 2. 通过四元数MLP处理
        x_q = self.q_relu(self.q_bn1(self.q_fc1(quat_global_feat)))
        x_q = self.q_dropout(x_q)
        final_quat_vec = self.q_relu(self.q_bn2(self.q_fc2(x_q))) # Shape: [B, 256, 4]
        local_feat = self.q2r(quat_local_feat)
        # 3. 转换为实数特征
        # 论文中提到使用模的平方 [cite: 576]
        real_invariant_feat = self.q2r(final_quat_vec) # Shape: [B, 256]

        # 4. 通过标准的MLP进行最终分类
        x = self.relu(self.bn1(self.fc1(real_invariant_feat)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 应用 log_softmax 得到最终预测
        log_probs = F.log_softmax(x, dim=1)
        
        # 与valid.py中的接口保持一致，返回预测结果和最后的四元数特征
        return log_probs, final_quat_vec, local_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        # REQNN 不需要特征变换的正则化损失，因为其设计保证了旋转等变性
        self.mat_diff_loss_scale = 0

    def forward(self, pred, target, trans_feat):
        # trans_feat 在我们的 REQNN 实现中应为 None
        loss = F.nll_loss(pred, target)
        return loss

if __name__ == "__main__":
    # 测试四元数PointNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 8
    num_points = 1024
    x = torch.randn(batch_size, num_points, 3).to(device)
    
    # 创建模型
    model = ContrastPointNetREQNN(k=40, normal_channel=False).to(device)
    
    # 前向传播
    pred, final_quat_vec, local_feat = model(x)
    
    print(f"输出pred形状: {pred.shape}")
    print(f"输出final_quat_vec形状: {final_quat_vec.shape}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
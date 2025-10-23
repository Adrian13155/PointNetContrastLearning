# models/reqnn_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.reqnn_util import QLinear, QMaxPooling, QReLU, QBatchNorm1d, Quaternion2Real
    

class PointNetEncoderREQNN(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetEncoderREQNN, self).__init__()
        
        # 定义四元数卷积层
        self.conv1 = QLinear(1, 64)
        self.conv2 = QLinear(64, 128)
        self.conv3 = QLinear(128, 1024)
        
        # 定义四元数批归一化层
        self.bn1 = QBatchNorm1d(64)
        self.bn2 = QBatchNorm1d(128)
        self.bn3 = QBatchNorm1d(1024)

        # 实例化论文中定义的QReLU
        self.q_relu = QReLU()
        # 实例化池化层，并指定在“点”的维度(dim=2)上操作
        self.pooling = QMaxPooling(dim=2)
        
        self.global_feat = global_feat
        # REQNN的设计使其天然具有旋转等变性，不再需要特征变换网络
        self.feature_transform = False 

    def forward(self, x):
        # 输入 x 的形状是 [Batch, Dims, Num_Points], 例如 [4, 3, 1024]
        B, D, N = x.size()
        
        # 1. 转置为 [B, N, D] 格式以进行后续处理
        x = x.transpose(1, 2)  # [B, N, D] = [B, 1024, 3]
        
        # 2. 将3D点云提升为纯四元数 (w=0, x, y, z)
        # 我们应该在最后一个维度 (D=3) 的左边填充一个0
        x_q = F.pad(x, (1, 0), 'constant', 0)  # 现在的形状是: [B, N, 4]
        
        # 3. 调整维度以匹配 QConv1d 的输入格式: [B, C_in, N, 4]
        # 我们将每个点云视为一个通道 (C_in=1)
        x_q = x_q.unsqueeze(1) # 现在的形状是: [B, 1, N, 4]
        
        # 3. 依次通过四元数卷积、BN和QReLU层
        # 使用 QReLU 替换 F.relu
        x_q = self.q_relu(self.bn1(self.conv1(x_q))) # Shape: [B, 64, N, 4]
        
        pointfeat = x_q  # 保存局部特征，用于分割等任务

        x_q = self.q_relu(self.bn2(self.conv2(x_q))) # Shape: [B, 128, N, 4]
        x_q = self.q_relu(self.bn3(self.conv3(x_q))) # 最后一层卷积后通常不加激活函数，但是这里为了旋转不变性加了

        # 将四元数局部特征转换为实数特征
        local_real_feat = x_q  

        # 4. 使用正确的实例化池化层进行等变最大池化
        # print(f"输出x_q形状: {x_q.shape}")
        x_q = self.pooling(x_q) # Shape: [B, 1024, 4]
        
        if self.global_feat:
            # 返回全局特征，并用None填充以保持接口兼容性
            return x_q, local_real_feat, None
        else:
            # 这是为分割任务准备的，将全局特征复制并与局部特征拼接
            x_q_replicated = x_q.unsqueeze(2).repeat(1, 1, N, 1)
            return torch.cat([x_q_replicated, pointfeat], 1), local_real_feat, None
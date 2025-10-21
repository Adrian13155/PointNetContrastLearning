import torch
import torch.nn as nn
import torch.nn.functional as F

# 用形为 (..., 4) 的张量来表示四元数，
# 其中最后一维按顺序存储 (q0, q1, q2, q3) 分量。

def q_conjugate(q):
    # 共轭
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def q_mul(q1, q2):
    # 四元数乘
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)

def q_norm(q):
    # 模
    return torch.sqrt(torch.sum(q * q, dim=-1))

def q_rotate(points, R):
    points_q = F.pad(points, (1, 0), 'constant', 0)   # (0, x, y, z)
    
    # R单位化
    R = R / (q_norm(R).unsqueeze(-1) + 1e-8)
    R_conj = q_conjugate(R)
    # p' = R * p * R_conj
    rotated_points_q = q_mul(q_mul(R, points_q), R_conj)

    return rotated_points_q[..., 1:]


# 2. REQNN
class QConv1d(nn.Module):
    """
    旋转等变的四元数全连接层 (等价于核大小为1的1D卷积)。
    权重是实数，没有偏置项。
    """
    def __init__(self, in_features, out_features, kernel_size=1):
        super(QConv1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重矩阵是实数
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x_q):
        # x_q shape: (..., C_in, 4)
        # weight shape: (C_out, C_in)
        # 我们希望输出 shape: (..., C_out, 4)
        # 使用 einsum 实现高效的矩阵乘法，同时保持四元数维度
        
        # (B, C, N, 4)
        if x_q.dim() == 4:
            # 修正: 'bnic' -> 'binc' 以匹配 (Batch, Channels, Points, Quat) 的维度顺序
            return torch.einsum('oi,binc->bonc', self.weight, x_q)
        # (B, C, 4)
        elif x_q.dim() == 3:
            # 'bic' -> (Batch, Channels, Quat) 这个是正确的
            return torch.einsum('oi,bic->boc', self.weight, x_q)
        else:
            raise ValueError("Input tensor must have 3 or 4 dimensions")

class QLinear(nn.Module):
    """
    旋转等变的四元数全连接层 (等价于核大小为1的1D卷积)。
    权重是实数，没有偏置项。
    """
    def __init__(self, in_features, out_features):
        super(QLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重矩阵是实数
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x_q):
        # x_q shape: (..., C_in, 4)
        # weight shape: (C_out, C_in)
        # 我们希望输出 shape: (..., C_out, 4)
        # 使用 einsum 实现高效的矩阵乘法，同时保持四元数维度
        
        # (B, C, N, 4)
        if x_q.dim() == 4:
            # 修正: 'bnic' -> 'binc' 以匹配 (Batch, Channels, Points, Quat) 的维度顺序
            return torch.einsum('oi,binc->bonc', self.weight, x_q)
        # (B, C, 4)
        elif x_q.dim() == 3:
            # 'bic' -> (Batch, Channels, Quat) 这个是正确的
            return torch.einsum('oi,bic->boc', self.weight, x_q)
        else:
            raise ValueError("Input tensor must have 3 or 4 dimensions")


class QReLU(nn.Module):
    # c = 1 为默认值
    def __init__(self, c=1.0):
        super(QReLU, self).__init__()
        self.c = c

    def forward(self, x_q):
        # x_q shape: (..., 4)
        norm = q_norm(x_q).unsqueeze(-1) 
        scale = norm / torch.max(norm, torch.tensor(self.c, device=x_q.device))
        return scale * x_q

"""
class QBatchNorm1d(nn.Module):
    
    #旋转等变的四元数批归一化。
    #公式: norm(f_v) = f_v / sqrt(E[||f_v||^2] + epsilon)
    
    def __init__(self, num_features, epsilon=1e-5):
        super(QBatchNorm1d, self).__init__()
        self.epsilon = epsilon
        # 注意：这个BN层没有可学习的仿射参数 (gamma, beta)
        # 因为它们会破坏旋转等变性

    def forward(self, x_q):
        # x_q shape: (Batch, Channels, Points, 4)
        # 计算每个四元数模的平方
        norm_sq = torch.sum(x_q * x_q, dim=-1) # Shape: (B, C, N)
        
        # 在批次和点维度上计算均值
        # E[||f_v||^2]
        mean_norm_sq = torch.mean(norm_sq, dim=[0, 2], keepdim=True) # Shape: (1, C, 1)
        
        # 计算分母
        denominator = torch.sqrt(mean_norm_sq + self.epsilon).unsqueeze(-1) # Shape: (1, C, 1, 1)
        
        # 归一化
        return x_q / denominator
"""
class QBatchNorm1d(nn.Module):
    """
    旋转等变的四元数批归一化。
    公式: norm(f_v) = f_v / sqrt(E[||f_v||^2] + epsilon)
    """
    def __init__(self, num_features, epsilon=1e-5):
        super(QBatchNorm1d, self).__init__()
        self.epsilon = epsilon
        # 注意：这个BN层没有可学习的仿射参数 (gamma, beta)
        # 因为它们会破坏旋转等变性

    def forward(self, x_q):
        # 计算每个四元数模的平方
        norm_sq = torch.sum(x_q * x_q, dim=-1) # 4D输入时 shape:(B,C,N), 3D输入时 shape:(B,C)
        
        # === 核心修改：根据输入维度决定求均值的维度 ===
        if x_q.dim() == 4:
            # 输入来自卷积层，形状为 [B, C, N, 4]
            # 在批次和点维度上计算均值
            mean_norm_sq = torch.mean(norm_sq, dim=[0, 2], keepdim=True) # Shape: (1, C, 1)
        elif x_q.dim() == 3:
            # 输入来自全连接层（全局特征），形状为 [B, C, 4]
            # 只在批次维度上计算均值
            mean_norm_sq = torch.mean(norm_sq, dim=0, keepdim=True) # Shape: (1, C)
        else:
            raise ValueError("QBatchNorm1d expects input to be 3D or 4D, but got {}D".format(x_q.dim()))
        
        # 扩展维度以进行广播除法
        while len(mean_norm_sq.shape) < len(x_q.shape):
            mean_norm_sq = mean_norm_sq.unsqueeze(-1)
        
        # 计算分母
        denominator = torch.sqrt(mean_norm_sq + self.epsilon)
        
        # 归一化
        return x_q / denominator
    
class QMaxPooling(nn.Module):
    """
    旋转等变的四元数最大池化。
    选择具有最大模的四元数特征。
    """
    def __init__(self, dim):
        super(QMaxPooling, self).__init__()
        self.dim = dim

    def forward(self, x_q):
        # x_q shape: (B, C, N, 4)
        # 计算模
        norms = q_norm(x_q) # Shape: (B, C, N)
        
        # 找到最大模的索引
        _, max_indices = torch.max(norms, dim=self.dim, keepdim=True) # Shape: (B, C, 1)
        
        # 使用索引来选择对应的四元数
        # 需要将索引扩展到四元数维度
        max_indices = max_indices.unsqueeze(-1).expand(-1, -1, -1, 4) # Shape: (B, C, 1, 4)
        
        # 从原始张量中收集元素
        # squeeze(self.dim) 移除池化维度
        return torch.gather(x_q, self.dim, max_indices).squeeze(self.dim)


class QDropout(nn.Module):
    
    # 该层会随机地将整个四元数特征（包括实部和虚部）置零。
    # 这保证了 Dropout 操作本身不会破坏特征的旋转等变性。
    def __init__(self, p=0.5):
        """
        Args:
            p (float): 需要被置零的四元数特征的比例，取值范围 [0, 1]。
        """
        super(QDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x_q):
        """
        Args:
            x_q (torch.Tensor):输入的四元数特征张量。
                               形状可以是 [B, C, 4] (用于QLinear)
                               或 [B, C, 4, N] (用于QConv1d)。

        Returns:
            torch.Tensor: 经过 Dropout 处理后的四元数特征张量。
        """
        # 如果 p=0 或者在评估模式下，则不执行 dropout
        if self.p == 0. or not self.training:
            return x_q

        # 核心：创建一个与通道维度(C)相对应的 mask
        # 形状为 [B, C]，这样可以对每个样本的每个通道独立进行 dropout
        mask_shape = (x_q.shape[0], x_q.shape[1])
        
        # F.dropout 会自动处理 mask 的生成和 scaling (乘以 1/(1-p))
        # 我们创建一个形状为 [B, C, 1] 的张量，让 dropout 作用于通道上
        # 这样就能得到一个每个通道要么是 0 要么是 1/(1-p) 的 mask
        channel_mask = F.dropout(torch.ones(mask_shape, device=x_q.device), self.p, self.training)

        # 将 mask 的维度进行扩展，使其能够通过广播机制
        # 应用于整个四元数张量 (所有4个分量以及所有空间点)
        # 例如，如果 x_q 是 [B, C, 4, N]，mask 会变为 [B, C, 1, 1]
        while len(channel_mask.shape) < len(x_q.shape):
            channel_mask = channel_mask.unsqueeze(-1)
        
        # 将 mask 应用于输入张量
        # 对于被选中的通道，其 a, i, j, k 四个分量会同时被置零
        return x_q * channel_mask

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
    
    
# ==============================================================================
# 3. Quaternion2Real 模块
# ==============================================================================

class Quaternion2Real(nn.Module):
    """
    将四元数特征转换为实数特征。
    这个操作是旋转不变的。
    f_real = ||f_v||^2 = a^2 + b^2 + c^2
    """
    def forward(self, x_q):
        # x_q shape: (..., 4)
        # 论文中提到使用纯四元数特征 f_v = 0 + ai + bj + ck
        # 所以模的平方是 a^2 + b^2 + c^2
        # 我们的实现中，特征可能不是纯四元数，所以我们计算完整的模
        return torch.sum(x_q * x_q, dim=-1)


class QuaternionToRealLocal(nn.Module):
    """
    将四元数局部特征转换为实数局部特征。
    输入: [B, C, N, 4] 的四元数特征
    输出: [B, C, N] 的实数特征
    """
    def forward(self, x_q):
        # x_q shape: [B, C, N, 4]
        # 计算每个四元数的模的平方，得到旋转不变的实数特征
        return torch.sum(x_q * x_q, dim=-1)  # [B, C, N]


class QuaternionToRealMatrix(nn.Module):
    """
    将四元数特征转换为实数矩阵特征。
    输入: [B, C, N, 4] 的四元数特征
    输出: [B, C, C] 的实数特征矩阵
    """
    def __init__(self, method='norm'):
        super(QuaternionToRealMatrix, self).__init__()
        self.method = method
        
    def forward(self, x_q):
        # x_q shape: [B, C, N, 4]
        B, C, N, _ = x_q.shape
        
        if self.method == 'norm':
            # 方法1: 计算每个四元数的模的平方
            real_features = torch.sum(x_q * x_q, dim=-1)  # [B, C, N]
            # 通过矩阵乘法得到 [B, C, C] 的特征矩阵
            real_matrix = torch.bmm(real_features, real_features.transpose(1, 2))  # [B, C, C]
            
        elif self.method == 'inner_product':
            # 方法2: 计算四元数之间的内积
            # 将四元数特征重塑为 [B, C*N, 4]
            x_flat = x_q.view(B, C*N, 4)  # [B, C*N, 4]
            # 计算内积矩阵
            inner_prod = torch.bmm(x_flat, x_flat.transpose(1, 2))  # [B, C*N, C*N]
            # 重塑为 [B, C, C] (取对角线块)
            real_matrix = inner_prod[:, :C, :C]
            
        elif self.method == 'covariance':
            # 方法3: 计算协方差矩阵
            real_features = torch.sum(x_q * x_q, dim=-1)  # [B, C, N]
            # 计算协方差矩阵
            mean = real_features.mean(dim=2, keepdim=True)  # [B, C, 1]
            centered = real_features - mean  # [B, C, N]
            real_matrix = torch.bmm(centered, centered.transpose(1, 2)) / (N - 1)  # [B, C, C]
            
        return real_matrix
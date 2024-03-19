import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.model_zoo import load_url

# 定义双向图卷积网络模块
class DualGraphConvolution(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DualGraphConvolution, self).__init__()
        # 定义双向图卷积网络的参数和层
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out

# 定义多层空间注意力机制模块
class MultiLevelSpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(MultiLevelSpatialAttention, self).__init__()
        # 定义多层空间注意力机制的参数和层
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out2 = self.conv2(out1)
        out2 = self.relu(out2)
        out3 = self.conv3(out2)
        out3 = self.relu(out3)
        out = out3 + x
        return out

# 定义融合后的ehanet网络
class EhanetWithDualGCN(nn.Module):
    def __init__(self, num_classes):
        super(EhanetWithDualGCN, self).__init__()
        # 定义ehanet网络的参数和层
        # ...

        # 添加双向图卷积网络模块
        self.dual_gcn = DualGraphConvolution(in_planes, out_planes)

        # 添加多层空间注意力机制模块
        self.spatial_attention = MultiLevelSpatialAttention(in_planes)

        # ...

    def forward(self, x):
        # ehanet网络前向传播
        # ...

        # 双向图卷积网络模块的前向传播
        out = self.dual_gcn(x)

        # 多层空间注意力机制模块的前向传播
        out = self.spatial_attention(out)

        # ...

        return out

# 创建融合后的ehanet网络实例
# model = EhanetWithDualGCN(num_classes)
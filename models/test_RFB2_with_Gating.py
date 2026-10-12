import torch

from models.common import DenseRFB2, RFB2

# 设置输入和模型参数
batch_size = 2  # 批次大小
in_channels = 768  # 输入通道数
out_channels = 256  # 输出通道数
height, width = 32, 32  # 输入特征图的高度和宽度

# 创建一个随机输入张量，形状为 (batch_size, in_channels, height, width)
x = torch.randn(batch_size, in_channels, height, width)
print(x.size())

# 创建 RFB2_with_Gating 模型
model = DenseRFB2(in_planes=in_channels, out_planes=out_channels, map_reduce=4, d=[2, 3])

# 前向传播
output = model(x)

# 打印输出的形状
print("Output shape:", output.shape)

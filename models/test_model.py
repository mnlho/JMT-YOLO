import torch
from yolo import Model  # 确保模型类导入正确
import numpy as np

# 假设你已经有了配置文件路径
cfg = 'yolov5l_city_seg.yaml'
# cfg = 'yolov5l.yaml'
device = 'cuda:0'  # 可以改为 'cuda:0'，如果你有 GPU

# 创建模型
model = Model(cfg).to(device)

# 打印模型结构，检查是否加载正确
print(model)

# 创建一个虚拟输入张量（模拟输入图像）
dummy_input = torch.randn(4, 3, 960, 960).to(device)

# 测试前向传播
outputs = model(dummy_input)
# a = np.array([tensor.cpu().detach().numpy() for tensor in outputs[0]])
a = np.array(outputs[1].cpu().detach().numpy())

softmax_outputs = torch.softmax(outputs[1], dim=1)
# 检查 logits 的范围
print(f"Logits min: {outputs[1].min().item()}, max: {outputs[1].max().item()}")
# 检查 softmax 输出的范围
print(f"Softmax output min: {softmax_outputs.min().item()}, max: {softmax_outputs.max().item()}")  # 应在 [0, 1]
# 验证概率和为 1
pixel_sum = softmax_outputs.sum(dim=1)  # 按类别维度求和
print(f"Pixel probability sum (should be all 1): {pixel_sum.unique()}")

print(outputs)



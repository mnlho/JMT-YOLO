import torch
print(torch.cuda.is_available())  # 检查CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)  #注意是双下划线
print("torch.version.cuda ", torch.version.cuda)

print(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2 )
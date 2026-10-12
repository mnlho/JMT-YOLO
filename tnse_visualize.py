import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ----------- 配置参数 ------------
SOURCE_DIR = '/home/yu/zjh/SSDA-YOLO-copy/dataset/leftImg8bit_trainvaltest/yolov5_format/images/train'
TARGET_DIR = '/home/yu/zjh/SSDA-YOLO-copy/dataset/leftImg8bit_trainval_foggyDBF/yolov5_format/images/val'
MODEL_PATH = '/home/yu/zjh/SSDA-YOLO-copy/runs/train/12_21_bestpt_maskratio_03.6/weights/best_student.pt'
IMAGE_SIZE = 960
SAMPLE_NUM = 300
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER_INDEX = 23  # 修改这里选择要提取的中间层输出
# --------------------------------

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 随机获取指定数量的图像路径
def get_images(folder, num=100):
    imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    return random.sample(imgs, min(len(imgs), num))

# 加载模型（只保留模型结构）
def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = ckpt['model'].float().fuse().eval() if 'model' in ckpt else ckpt
    model.to(DEVICE)
    return model


# 提取中间特征（支持Concat等模块）
def extract_feature(model, img_path):
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

    # Hook to capture intermediate layer output
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()

        return hook

    # Register hook to the specified layer
    layer = model.model[LAYER_INDEX]
    layer.register_forward_hook(get_features('intermediate'))

    # Forward pass
    with torch.no_grad():
        model(img)

    # Get the features and flatten them
    feature = features['intermediate']
    feature = feature.squeeze(0)  # Remove batch dimension

    # Global average pooling to reduce spatial dimensions
    feature = torch.mean(feature, dim=[1, 2])  # Average over height and width

    # Convert to numpy and normalize
    feature = feature.cpu().numpy()
    feature = feature / np.linalg.norm(feature)  # L2 normalization

    return feature


# 主流程
def run_tsne():
    print("�� 加载模型中...")
    model = load_model(MODEL_PATH)

    print("�� 模型结构:")
    for i, layer in enumerate(model.model):
        print(f"{i}: {layer}")
    print("✅ 总层数:", len(list(model.model)))

    print("�� 加载图像中...")
    source_imgs = get_images(SOURCE_DIR, SAMPLE_NUM)
    target_imgs = get_images(TARGET_DIR, SAMPLE_NUM)

    print("�� 提取特征中...")
    features, labels = [], []
    for img in source_imgs:
        features.append(extract_feature(model, img))
        labels.append(0)  # 源域

    for img in target_imgs:
        features.append(extract_feature(model, img))
        labels.append(1)  # 目标域

    print("�� 降维中 (t-SNE)...")
    features = np.stack(features)
    labels = np.array(labels)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    reduced = tsne.fit_transform(features)

    print("�� 可视化中...")
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[labels==0, 0], reduced[labels==0, 1], c='blue', label='Source', alpha=0.6)
    plt.scatter(reduced[labels==1, 0], reduced[labels==1, 1], c='red', label='Target', alpha=0.6)
    plt.legend()
    # plt.title('t-SNE Visualization of Intermediate Features')
    plt.grid(True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_result_ours.svg")
    plt.show()
    print("✅ 完成！结果保存为 tsne_result.png")

if __name__ == "__main__":
    run_tsne()

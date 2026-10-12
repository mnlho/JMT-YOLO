import numpy as np
import torch
from matplotlib import pyplot as plt

# from utils.dataset_K2C import create_dataloader


from utils.datasets import create_segmentation_dataloader



def test_dataloader(path, path_fake, imgsz=960, batch_size=8, stride=32, workers=4, rect=False, augment=False,
                    cache=False):

    # 调用 create_segmentation_dataloader 创建 DataLoader
    dataloader, dataset = create_segmentation_dataloader(
        path=path,
        path_fake=path_fake,
        imgsz=imgsz,
        batch_size=batch_size,
        stride=stride,
        workers=workers,
        rect=rect,
        augment=augment,
        cache=cache
    )

    # 打印 DataLoader 和 dataset 的基本信息
    print(f"DataLoader created successfully. Dataset length: {len(dataset)}")
    print(f"Batch size: {batch_size}, Image size: {imgsz}")

    # 迭代 DataLoader 获取一批数据
    for batch_idx, (imgs, imgs_fake, labels, paths, paths_fake, shape) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")


        # 打印每个 batch 的信息
        print(f"Real images batch shape: {imgs.shape}")
        print(f"Fake images batch shape: {imgs_fake.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print(f"Paths: {paths}")
        print(f"Fake Paths: {paths_fake}")

        # 检查数据类型
        assert isinstance(imgs, torch.Tensor), "Images should be torch.Tensor"
        assert isinstance(imgs_fake, torch.Tensor), "Fake images should be torch.Tensor"
        assert isinstance(labels, torch.Tensor), "Labels should be torch.Tensor"

        # 检查形状
        assert imgs.shape[1] == 3, f"Expected 3 channels in the image, got {imgs.shape[1]}"
        assert len(imgs.shape) == 4, f"Expected 4D tensor for images, got {imgs.shape}"
        assert imgs.shape[0] == batch_size, f"Expected batch size {batch_size}, got {imgs.shape[0]}"

        # 画出第一张图片和标签（如果 batch_size > 1，显示第一张）
        img = imgs[0].cpu().numpy().transpose(1, 2, 0)
        img_fake = imgs_fake[0].cpu().numpy().transpose(1, 2, 0)
        label = labels[0].cpu().numpy()

        # 打印标签数组的统计信息
        print("\nLabel Array:")
        print(label)  # 显示完整的标签数组

        print(f"Label unique values: {np.unique(label)}")  # 输出标签中的唯一值
        print(f"Label shape: {label.shape}")  # 显示标签形状

        # 将标签画出来
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Real Image')

        plt.subplot(1, 3, 2)
        plt.imshow(img_fake)
        plt.title('Fake Image')

        plt.subplot(1, 3, 3)
        plt.imshow(label, cmap='gray')
        plt.title('Segmentation Label')

        plt.show()

        break  # 只显示一个 batch 的数据进行测试


if __name__ == "__main__":
    # 设置图像路径和假图像路径
    real_images_path = "/home/yu/zjh/SSDA-YOLO-copy/dataset/leftImg8bit_trainvaltest/segData/images/train"  # 替换为实际路径
    fake_images_path = "/home/yu/zjh/SSDA-YOLO-copy/dataset/CityScapes2CityScapesFoggy/segData_csf/images/train"  # 替换为实际路径

    # 测试 DataLoader 配置
    test_dataloader(real_images_path, fake_images_path, imgsz=640, batch_size=4, stride=32, workers=4, rect=False,
                    augment=False, cache=False)

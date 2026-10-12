import os
import shutil
import random

def split_dataset(image_dir, label_dir, train_image_dir, test_image_dir, train_label_dir, test_label_dir, split_ratio=0.7):
    # 确保目标目录存在
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    # 确保标签文件存在
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]

    # 检查图像文件和标签文件是否一一对应
    if set(image_files) != set(label_files):
        print("Error: The image files and label files do not match!")
        return

    # 随机打乱图像文件列表
    random.shuffle(image_files)

    # 计算训练集的大小
    train_size = int(len(image_files) * split_ratio)

    # 划分训练集和测试集
    train_files = image_files[:train_size]
    test_files = image_files[train_size:]

    # 函数：复制图像和标签文件到指定目录
    def copy_files(file_list, source_image_dir, source_label_dir, dest_image_dir, dest_label_dir):
        for filename in file_list:
            # 复制图像文件
            image_path = os.path.join(source_image_dir, filename)
            shutil.copy(image_path, os.path.join(dest_image_dir, filename))

            # 复制标签文件
            label_path = os.path.join(source_label_dir, filename)
            shutil.copy(label_path, os.path.join(dest_label_dir, filename))

    # 复制训练集和测试集的图像和标签
    copy_files(train_files, image_dir, label_dir, train_image_dir, train_label_dir)
    copy_files(test_files, image_dir, label_dir, test_image_dir, test_label_dir)

    print(f"Dataset split complete. {len(train_files)} images in training set, {len(test_files)} images in test set.")

if __name__ == "__main__":
    # 设置图像和标签文件夹的路径
    image_dir = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/images"  # 图像文件所在的目录
    label_dir = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/labels"  # 标签文件所在的目录

    # 设置训练集和测试集的目标目录
    train_image_dir = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/seg_kitti_2/images/train"  # 训练集图像保存目录
    test_image_dir = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/seg_kitti_2/images/val"  # 测试集图像保存目录
    train_label_dir = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/seg_kitti_2/labels/train"  # 训练集标签保存目录
    test_label_dir = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/seg_kitti_2/labels/val"  # 测试集标签保存目录

    # 创建保存目录
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # 划分数据集并复制文件
    split_dataset(image_dir, label_dir, train_image_dir, test_image_dir, train_label_dir, test_label_dir)

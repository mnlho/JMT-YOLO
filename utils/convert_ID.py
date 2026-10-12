import os
import numpy as np
from PIL import Image

def modify_labels(kitti_label_path, output_label_path):
    # 检查路径是否是目录还是文件
    if os.path.isdir(kitti_label_path):
        print(f"Error: The path '{kitti_label_path}' is a directory, not a file.")
        return
    if not os.path.exists(kitti_label_path):
        print(f"Error: The path '{kitti_label_path}' does not exist.")
        return

    # 加载标签图像
    label = np.array(Image.open(kitti_label_path))  # 加载为 numpy 数组

    # 打印标签图像的唯一值，确保我们知道现有的类别值
    unique_values = np.unique(label)
    print(f"Original unique values in {kitti_label_path}: {unique_values}")

    # 创建一个新的标签图像，映射 KITTI 标签到 Cityscapes 标签
    label_mapped = label.copy()

    # KITTI -> Cityscapes 映射 (根据你的需求)
    label_mapped[label_mapped == 0] = 255
    label_mapped[label_mapped == 24] = 0  # 将 'person' (13) 转换为 '0' (Cityscapes: person)
    label_mapped[label_mapped == 26] = 1  # 将 'car' (10) 转换为 '1' (Cityscapes: car)

    # 保存新的标签图像
    new_label_img = Image.fromarray(label_mapped)
    new_label_img.save(output_label_path)

    # 打印映射后的唯一值
    print(f"Modified unique values in {output_label_path}: {np.unique(label_mapped)}")

def process_directory(label_dir, output_dir):
    # 列出目录中的所有文件
    files = os.listdir(label_dir)
    print(f"Files in the directory: {files}")

    # 遍历目录中的每个文件
    for filename in files:
        # 只处理以 .png 结尾的文件
        if filename.endswith(".png"):
            # 构建标签图像的完整路径
            kitti_label_path = os.path.join(label_dir, filename)
            # 构建输出标签图像的路径
            output_label_path = os.path.join(output_dir, f"{filename}")

            # 修改标签
            modify_labels(kitti_label_path, output_label_path)

if __name__ == "__main__":
    # 设置标签目录路径
    label_dir = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/data_semantics/training/output"  # 标签文件所在的目录
    output_dir = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/data_semantics/training/output_id"  # 保存修改后的标签图像的目录

    # 创建输出目录，如果不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理目录下所有标签文件
    process_directory(label_dir, output_dir)

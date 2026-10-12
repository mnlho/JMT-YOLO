#功能为遍历kitti语义分割标签目录，之保存两种类别
import numpy as np
import os
from PIL import Image
import glob

# 定义类别 ID
PERSON_ID = 24
CAR_ID = 26
BACKGROUND_ID = 0


def filter_kitti_labels(input_dir, output_dir):
    """
    过滤 KITTI 数据集的标签，只保留 person 和 car 类别，其他设置为背景
    """
    # 获取所有标签图像文件
    label_files = glob.glob(os.path.join(input_dir, "*.png"))

    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label_file in label_files:
        # 读取标签图像
        label_img = np.array(Image.open(label_file))

        # 创建新的标签图像：只保留 person 和 car，其他设置为背景
        filtered_label_img = np.zeros_like(label_img)

        # 保留 person 和 car 的类别
        filtered_label_img[label_img == PERSON_ID] = PERSON_ID
        filtered_label_img[label_img == CAR_ID] = CAR_ID

        # 保存处理后的标签图像
        output_file = os.path.join(output_dir, os.path.basename(label_file))
        Image.fromarray(filtered_label_img).save(output_file)

        print(f"Processed {label_file} -> {output_file}")


# 主程序
if __name__ == "__main__":
    input_directory = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/data_semantics/training/semantic"  # 替换为你的 KITTI 标签目录
    output_directory = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/data_semantics/training/output"  # 替换为你想保存处理后标签的目录

    filter_kitti_labels(input_directory, output_directory)
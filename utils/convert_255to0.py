
import numpy as np
from PIL import Image
import os
def replace_background_in_label(image_path, output_path, target_value=255, replace_value=0):
    """
    替换标签图像中的指定背景值。

    Args:
    - image_path (str): 输入标签图像路径。
    - output_path (str): 输出图像路径。
    - target_value (int): 需要替换的像素值（默认255）。
    - replace_value (int): 替换后的像素值（默认0）。
    """
    # 打开标签图像（以索引模式‘P’打开）
    label_image = Image.open(image_path)

    # 转换为 numpy 数组
    label_array = np.array(label_image, dtype=np.uint8)

    # 替换指定背景值
    label_array[label_array == target_value] = replace_value

    # 转换回图像并保存
    modified_label_image = Image.fromarray(label_array)
    modified_label_image.save(output_path)
    print(f"Processed label image saved to {output_path}")


def process_labels_in_directory(input_dir, output_dir, target_value=255, replace_value=0):
    """
    批量处理目录中的标签图像，替换指定背景值。

    Args:
    - input_dir (str): 输入目录，包含标签图像。
    - output_dir (str): 输出目录，保存处理后的图像。
    - target_value (int): 需要替换的像素值（默认255）。
    - replace_value (int): 替换后的像素值（默认0）。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for entry in os.scandir(input_dir):
        if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = entry.path
            output_path = os.path.join(output_dir, entry.name)

            replace_background_in_label(image_path, output_path, target_value, replace_value)


if __name__ == "__main__":
    input_directory = "/home/yu/zjh/SSDA-YOLO-copy/dataset2/VOCtrainval_11-May-2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass"
    output_directory = "/home/yu/zjh/SSDA-YOLO-copy/dataset2/VOCtrainval_11-May-2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/convert_class"

    process_labels_in_directory(input_directory, output_directory, target_value=255, replace_value=0)

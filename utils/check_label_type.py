import os
import numpy as np
import torch
from PIL import Image
from datetime import datetime


import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None' 
import matplotlib.pyplot as plt


def check_label_type(label_path, output_dir="label_check_results/007470"):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载标签图像
    if not os.path.exists(label_path):
        print(f"Error: The path '{label_path}' does not exist.")
        return

    label = np.array(Image.open(label_path))  # 加载为 numpy 数组
    
    # 获取标签文件名（不含路径）
    label_name = os.path.splitext(os.path.basename(label_path))[0]
    
    # 添加时间戳避免文件覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建结果文本文件
    result_file = os.path.join(output_dir, f"{label_name}_{timestamp}_result.txt")
    with open(result_file, 'w') as f:
        f.write(f"Label Analysis for: {label_path}\n")
        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

    # 获取标签图像的形状
    print(f"Label shape: {label.shape}")
    with open(result_file, 'a') as f:
        f.write(f"Label shape: {label.shape}\n\n")

    # 判断是否为 RGB 彩色编码（如果是三通道图像）
    if len(label.shape) == 3 and label.shape[2] == 3:
        print("The label is in RGB color-coded format.")
        unique_colors = np.unique(label.reshape(-1, 3), axis=0)
        print(f"Unique colors in the label: {len(unique_colors)} unique colors")
        
        with open(result_file, 'a') as f:
            f.write("Label Type: RGB color-coded format\n")
            f.write(f"Unique colors in the label: {len(unique_colors)} unique colors\n")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(label)
        plt.title('RGB Color Coded Label')
        output_path = os.path.join(output_dir, f"{label_name}_{timestamp}_rgb.png")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        print(f"Saved visualization to: {output_path}")

    # 判断是否为 One-Hot 编码（如果是四维数组，且每个像素的值为 0 或 1）
    elif len(label.shape) == 3 and label.shape[2] > 1:
        print("The label might be in One-Hot encoded format.")
        unique_values = np.unique(label)
        print(f"Unique values in the label (One-Hot encoding check): {unique_values}")
        
        with open(result_file, 'a') as f:
            f.write("Label Type: One-Hot encoded format\n")
            f.write(f"Unique values in the label: {unique_values}\n")
        
        if set(unique_values) <= {0, 1}:  # Only 0 and 1 are allowed in One-Hot encoding
            print("This seems to be One-Hot encoded.")
            with open(result_file, 'a') as f:
                f.write("Confirmed: One-Hot encoded\n")
        else:
            print("This doesn't seem like One-Hot encoding.")
            with open(result_file, 'a') as f:
                f.write("Note: Doesn't seem like One-Hot encoding\n")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(label.argmax(axis=-1))  # Show the class index image (max index per pixel)
        plt.title('One-Hot Encoded Label')
        output_path = os.path.join(output_dir, f"{label_name}_{timestamp}_onehot.png")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        print(f"Saved visualization to: {output_path}")

    # 判断是否为类别索引形式（单通道，0 到 N 的整数）
    elif len(label.shape) == 2:
        print("The label is in Class Index Mask format.")
        unique_values = np.unique(label)
        print(f"Unique values in the label (Class Index Mask check): {unique_values}")
        
        with open(result_file, 'a') as f:
            f.write("Label Type: Class Index Mask format\n")
            f.write(f"Unique values in the label: {unique_values}\n")
            f.write(f"Number of unique classes: {len(unique_values)}\n")
        
        if all(v >= 0 and v <= 7 for v in unique_values):  # Adjust this range based on the number of classes
            print(f"This seems to be a Class Index Mask with {len(unique_values)} unique values.")
            with open(result_file, 'a') as f:
                f.write(f"Confirmed: Class Index Mask with {len(unique_values)} unique values\n")
        else:
            print("The values in the label image don't seem to match typical class indices.")
            with open(result_file, 'a') as f:
                f.write("Note: Values don't match typical class indices (0-7)\n")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(label, cmap='tab10')  # Use a predefined colormap for better visualizing class indices
        # plt.title('Class Index Mask Label')
        output_path = os.path.join(output_dir, f"{label_name}_{timestamp}_class_index.png")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        print(f"Saved visualization to: {output_path}")

    else:
        print("The label format is not recognized.")
        with open(result_file, 'a') as f:
            f.write("Label Type: UNRECOGNIZED FORMAT\n")
    
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    # Set the path to your label image/home/yu/zjh/SSDA-YOLO-copy/dataset3/data_semantics/training/output/000000_10.png
    label_path = "tools/maskpic_sam3_careful/val/007470.png"  # Replace this with your actual label path

    # Check the label type
    check_label_type(label_path)

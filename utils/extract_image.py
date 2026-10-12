import os
import shutil

# 定义文件夹路径
txt_file_path = '/home/yu/zjh/SSDA-YOLO-copy/dataset2/VOCtrainval_11-May-2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'  # 存放图片名称的txt文件夹
image_folder = '/home/yu/zjh/SSDA-YOLO-copy/dataset2/VOCtrainval_11-May-2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'  # 存放图片的文件夹
output_folder = '/home/yu/zjh/SSDA-YOLO-copy/dataset2/VOCtrainval_11-May-2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/seg_image'  # 存放提取出来的图片的文件夹

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取txt文件中的图片名称
with open(txt_file_path, 'r') as file:
    image_names = file.read().splitlines()

# 遍历图片名称列表
for image_name in image_names:
    # 假设图片名称没有扩展名，需要手动添加（如 .jpg 或 .png）
    image_name_with_extension = image_name + '.jpg'  # 根据实际情况调整扩展名
    image_path = os.path.join(image_folder, image_name_with_extension)

    # 检查图片是否存在
    if os.path.exists(image_path):
        # 复制图片到输出文件夹
        shutil.copy(image_path, output_folder)
        print(f"Copied: {image_name_with_extension}")
    else:
        print(f"Image not found: {image_name_with_extension}")

print("Extraction completed.")
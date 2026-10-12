import os
from PIL import Image

# 设置A文件夹和B文件夹路径
folder_A = '/home/yu/zjh/Code/contrastive-unpaired-translation/VOC_Clipart_train/V2C_TRAIN_CUT/train_latest/images/fake_B'
folder_B = '/home/yu/zjh/SSDA-YOLO-copy/dataset2/voc0712'

# 遍历A文件夹中的所有图片文件
for filename in os.listdir(folder_A):
    # 检查文件是否是PNG格式图片
    if filename.endswith('.png'):
        # 构建A文件夹和B文件夹中的文件路径
        file_A = os.path.join(folder_A, filename)
        # 构建B文件夹中对应的文件路径（假设B中的文件是.jpg格式）
        file_B = os.path.join(folder_B, filename.replace('.png', '.jpg'))

        # 检查B文件夹中是否存在对应的文件
        if os.path.exists(file_B):
            # 打开A和B文件夹中的图片
            img_A = Image.open(file_A)
            img_B = Image.open(file_B)

            # 获取图片尺寸
            size_A = img_A.size
            size_B = img_B.size

            # 如果图片大小不同，则调整A文件夹中的图片大小
            if size_A != size_B:
                print(f"Size mismatch for {filename}: A({size_A}) != B({size_B})")
                img_A = img_A.resize(size_B)  # 调整A图片大小
                img_A.save(file_A)  # 保存调整后的图片
                print(f"{filename} resized to {size_B}")

print("Process completed.")


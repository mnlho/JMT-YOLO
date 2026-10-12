import os
from PIL import Image


def resize_images_in_folder(folder_path, target_width, target_height):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查是否为文件，且是否为图片文件
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            try:
                # 打开图片文件
                with Image.open(file_path) as img:
                    # 调整图片大小
                    img_resized = img.resize((target_width, target_height))

                    # 保存调整后的图片
                    img_resized.save(file_path)
                    print(f"Resized {filename} to {target_width}x{target_height}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    # 直接指定文件夹路径和目标分辨率
    folder_path = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/seg_K2C_2/labels/val"  # 请替换为你的文件夹路径
    target_width = 1242  # 目标宽度
    target_height = 375  # 目标高度

    resize_images_in_folder(folder_path, target_width, target_height)


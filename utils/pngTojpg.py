from PIL import Image
import os

def convert_jpg_to_png(folder_path):
    # 确保路径存在
    if not os.path.exists(folder_path):
        print("指定的文件夹不存在！")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):  # 只处理jpg文件
            jpg_path = os.path.join(folder_path, filename)
            png_path = os.path.join(folder_path, filename.rsplit(".", 1)[0] + ".png")

            try:
                with Image.open(jpg_path) as img:
                    img.save(png_path, "PNG")  # 以PNG格式保存
                os.remove(jpg_path)  # 删除原始JPG文件
                print(f"转换成功并删除原文件: {filename} -> {os.path.basename(png_path)}")
            except Exception as e:
                print(f"转换失败: {filename}, 错误信息: {e}")

if __name__ == "__main__":
    folder = "/home/yu/zjh/SSDA-YOLO-copy/dataset2/clipart/seg_data/labels/train"
    convert_jpg_to_png(folder)

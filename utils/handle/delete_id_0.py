import os


def remove_class_zero(folder_path):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # 检查是否为文件（避免遍历到子文件夹）
        if os.path.isfile(file_path):
            try:
                # 打开文件，读取内容
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # 过滤掉 class_id 为 0 的行
                filtered_lines = [line for line in lines if not line.startswith("0 ")]

                # 将结果写回文件
                with open(file_path, 'w') as file:
                    file.writelines(filtered_lines)

                print(f"Processed file: {file_name}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")


# 使用方法
# 将 `your_folder_path` 替换为目标文件夹的路径
your_folder_path = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/yolo_C2K_format_2/labels/val"
remove_class_zero(your_folder_path)

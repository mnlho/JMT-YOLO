import os
import argparse


def update_yolo_labels(input_dir, output_dir, backup=True):
    """
    将YOLO标签中的类别索引（class_id）统一加1。

    参数：
        input_dir (str): 原始标签文件目录（包含.txt文件）
        output_dir (str): 调整后的标签文件输出目录
        backup (bool): 是否保留原始文件备份（默认True）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有.txt文件
    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 备份原始文件（可选）
        if backup:
            backup_path = os.path.join(output_dir, f"original_{filename}")
            with open(input_path, "r") as f_in, open(backup_path, "w") as f_backup:
                f_backup.write(f_in.read())

        # 处理标签文件
        with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # 跳过无效行

                # 提取class_id并加1
                class_id = int(parts[0])
                updated_class_id = class_id + 1

                # 更新行内容
                updated_line = f"{updated_class_id} {' '.join(parts[1:])}\n"
                f_out.write(updated_line)

        print(f"Processed: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update YOLO label class indices by +1")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of original labels")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save updated labels")
    parser.add_argument("--no_backup", action="store_false", dest="backup", help="Disable backup of original files")
    args = parser.parse_args()

    update_yolo_labels(args.input_dir, args.output_dir, args.backup)
import os
import shutil


def copy_label_train_ids(image_root, label_root, output_label_dir):
    """
    将 *_labelTrainIds.png 文件提取到指定文件夹，与 yolo 格式的图像一一对应，并去掉标签名称中的 '_gtFine_labelTrainIds' 部分。

    Parameters:
    - image_root: YOLO 格式图像的根目录路径（包含 train 和 val 子文件夹）
    - label_root: *_labelTrainIds.png 文件的根目录路径（包含 train 和 val 子文件夹及城市子文件夹）
    - output_label_dir: 提取后的 *_labelTrainIds.png 文件的输出根目录路径
    """

    # 创建根输出目录 seglabels
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    # 遍历 train 和 val 子文件夹
    for subset in ["train", "val"]:
        image_subset_path = os.path.join(image_root, subset)
        label_subset_path = os.path.join(label_root, subset)
        output_subset_path = os.path.join(output_label_dir, subset)

        # 创建输出的子目录（train 或 val）
        if not os.path.exists(output_subset_path):
            os.makedirs(output_subset_path)

        # 遍历子文件夹中的图片
        for root, _, files in os.walk(image_subset_path):
            for file_name in files:
                if file_name.endswith(".jpg"):
                    # 获取图像 ID（去除后缀）
                    image_id = file_name.replace(".jpg", "")

                    # 查找标签文件：遍历所有城市子文件夹
                    found_label = False
                    for city_name in os.listdir(label_subset_path):
                        city_label_path = os.path.join(label_subset_path, city_name)
                        if os.path.isdir(city_label_path):
                            # 构造对应的 *_labelTrainIds.png 文件路径
                            label_file_name = image_id + "_gtFine_labelTrainIds.png"
                            label_file_path = os.path.join(city_label_path, label_file_name)

                            # 检查是否存在对应的标签文件
                            if os.path.exists(label_file_path):
                                # 修改目标文件名，去掉 '_gtFine_labelTrainIds'
                                output_label_name = image_id + ".png"
                                output_label_path = os.path.join(output_subset_path, output_label_name)

                                # 将标签文件复制到目标子目录，并使用修改后的名称
                                shutil.copy(label_file_path, output_label_path)
                                found_label = True
                                break

                    if not found_label:
                        print(f"未找到对应的标签文件: {image_id}_gtFine_labelTrainIds.png")


if __name__ == "__main__":
    # 设置路径
    image_root = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/yolov5_city_format_2/images"  # 替换为实际路径
    label_root = "/home/yu/zjh/SSDA-YOLO-copy/dataset/leftImg8bit_trainvaltest/gtFine"  # 替换为实际路径
    output_label_dir = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/seg_city_2/labels"  # 替换为实际输出路径（seglabels）

    # 执行标签提取
    copy_label_train_ids(image_root, label_root, output_label_dir)

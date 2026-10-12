import numpy as np
from PIL import Image

# 类别ID与名称映射（根据 KITTI 数据集定义）
label_map = {
    0: 'background',
    1: 'car'
}
# label_map = {
#     0: 'background', 1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car', 8:'cat', 9:'chair', 10:'cow', 11:'diningtable', 12:'dog',
#     13:'horse', 14:'motorbike', 15:'person', 16:'pottedplant', 17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor'
# }
# label_map = {
#     0: 'background',1:'bus', 2:'bicycle', 3:'car', 4:'motorcycle', 5:'person', 6:'rider', 7:'train', 8:'truck'
# }


def display_label_info(label_image_path):
    """
    显示分割标签图像中每个唯一 ID 和对应的类别。

    参数:
    label_image_path (str): 标签图像文件路径
    """
    # 读取标签图像
    label_img = np.array(Image.open(label_image_path))

    # 获取图像中所有唯一的 ID
    unique_ids = np.unique(label_img)

    # 打印所有唯一的 ID 及其对应的类别
    print("Unique IDs and corresponding categories in the label image:")
    for unique_id in unique_ids:
        category = label_map.get(unique_id, 'Unknown')
        print(f"ID {unique_id} corresponds to: {category}")


# 主程序
if __name__ == "__main__":
    label_image_path = "/home/yu/zjh/SSDA-YOLO-copy/tools/maskpic/train/000001.png"  # 替换为你的标签图像文件路径
    display_label_info(label_image_path)

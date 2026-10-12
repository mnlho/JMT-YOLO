import cv2
import numpy as np

# 读取图像
image_path = '/home/yu/zjh/SSDA-YOLO-copy/dataset3/yolov5_city_format_2/images/train/aachen_000000_000019.jpg'  # 图像文件路径
label_path = '/home/yu/zjh/SSDA-YOLO-copy/dataset3/yolov5_city_format_2/labels/train/aachen_000000_000019.txt'  # YOLO 格式标签文件路径

# 读取图像
image = cv2.imread(image_path)

# 获取图像的实际宽度和高度
height, width, _ = image.shape

# 读取 YOLO 标签文件
with open(label_path, 'r') as file:
    lines = file.readlines()

# 读取每一行，解析并绘制框
for line in lines:
    # YOLO 标签格式：class_id x_center y_center width height
    parts = line.strip().split()
    class_id = int(parts[0])  # 类别 ID
    x_center = float(parts[1])  # 归一化 x 中心坐标
    y_center = float(parts[2])  # 归一化 y 中心坐标
    box_width = float(parts[3])  # 归一化宽度
    box_height = float(parts[4])  # 归一化高度

    # 将归一化坐标转换为实际像素值
    x_center_pixel = int(x_center * width)
    y_center_pixel = int(y_center * height)
    box_width_pixel = int(box_width * width)
    box_height_pixel = int(box_height * height)

    # 计算左上角和右下角坐标
    x1 = int(x_center_pixel - box_width_pixel / 2)
    y1 = int(y_center_pixel - box_height_pixel / 2)
    x2 = int(x_center_pixel + box_width_pixel / 2)
    y2 = int(y_center_pixel + box_height_pixel / 2)

    # 绘制框（假设类别从 0 开始，使用简单颜色标识）
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

    # 可选：绘制类名（假设你有一个类名列表）
    # class_names = ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']
    # class_names = ["class_1", "class_2", ...]  # 类别名称列表
    class_names = ['baskground', 'car']
    cv2.putText(image, class_names[class_id], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# 显示图像
cv2.imshow('Image with Bounding Boxes', image)

# 等待用户按键，按 'q' 键退出
cv2.waitKey(0)
cv2.destroyAllWindows()

# 可选：保存带框的图像
cv2.imwrite('image_with_boxes.jpg', image)

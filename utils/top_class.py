import cv2
import numpy as np

label_color_dict = {
    "bus": (255, 56, 56),
    "bicycle": (255, 157, 151),
    "car": (255, 112, 31),
    "mcycle": (255, 178, 29),
    "person": (207, 210, 49),
    "rider": (72, 249, 10),
    "train": (146, 204, 23),
    "truck": (61, 219, 134),
}

img_h, img_w, img_c = 50, 800, 3
canvas = np.ones((img_h, img_w, img_c)) * 255  # white canvas

for id, (label, color) in enumerate(label_color_dict.items()):
    px0 = 5 + (id % 4) * 200
    py0 = 9 + (id // 4) * 25
    px1 = 65 + (id % 4) * 200
    py1 = 16 + (id // 4) * 25
    print(id, label, color, px0, py0, px1, py1)
    cv2.rectangle(canvas, (px0, py0), (px1, py1), (color[2], color[1], color[0]),
                  -1, lineType=cv2.LINE_AA)

    px = 75 + (id % 4) * 200
    py = 20 + (id // 4) * 25
    cv2.putText(canvas, label, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

cv2.imwrite("/home/yu/zjh/SSDA-YOLO-copy/label_color.png", canvas)
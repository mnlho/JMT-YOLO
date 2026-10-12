#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""在图片上绘制 YOLO 格式标注框，并保存输出图片。

Usage / 用法
-----------
1) 最简单：图片旁边有同名 .txt 标签（YOLO 格式）
     python draw_yolo_boxes.py --img /path/to/xxx.jpg

2) 指定标签与输出路径
     - 输出到目录（推荐：--out 不带扩展名时会按“目录”处理）：
         python draw_yolo_boxes.py --img label_check_results/007470/007470.png --label tools/autobox_yolov5l_coco/val/007470.txt --out label_check_results/007470
     - 输出到文件：
         python draw_yolo_boxes.py --img /path/to/xxx.jpg --label /path/to/xxx.txt --out /path/to/out.jpg

3) 同时绘制类别名（可选）
     - 逗号分隔：
         python draw_yolo_boxes.py --img /path/to/xxx.jpg --names person,car,bus
     - txt 文件（每行一个类别名）：
         python draw_yolo_boxes.py --img /path/to/xxx.jpg --names classes.txt
     - data.yaml（含 names 字段，兼容 list/dict 两种写法）：
         python draw_yolo_boxes.py --img /path/to/xxx.jpg --names data.yaml

输入标签格式
-----------
每行一个目标：
    class_id x_center y_center width height

其中 (x_center, y_center, width, height) 都是相对图片宽高归一化到 [0, 1] 的值。

输出
----
默认输出到图片同目录，文件名为：xxx_boxed.jpg（或保持原后缀）。
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import yaml


def _load_names(names_arg: Optional[str]) -> Optional[List[str]]:
    if not names_arg:
        return None

    p = Path(names_arg)
    if p.exists() and p.is_file():
        if p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            names = data.get("names") if isinstance(data, dict) else None
            if isinstance(names, list):
                return [str(x) for x in names]
            if isinstance(names, dict):
                # {0: 'person', 1: 'car', ...}
                items = sorted(((int(k), v) for k, v in names.items()), key=lambda kv: kv[0])
                return [str(v) for _, v in items]
            raise ValueError(f"YAML 文件里没有可用的 names 字段: {p}")

        # txt: 每行一个类别名
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    # 逗号分隔："person,car,bus"
    return [s.strip() for s in names_arg.split(",") if s.strip()]


def _yolo_to_xyxy(
    x_center: float, y_center: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    xc = x_center * img_w
    yc = y_center * img_h
    bw = w * img_w
    bh = h * img_h

    x1 = int(round(xc - bw / 2.0))
    y1 = int(round(yc - bh / 2.0))
    x2 = int(round(xc + bw / 2.0))
    y2 = int(round(yc + bh / 2.0))

    # clamp
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    # ensure proper ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def _color_for_class(cls_id: int) -> Tuple[int, int, int]:
    # 简单可复现的颜色（BGR for OpenCV）
    palette = [
        (0, 255, 0),
        (0, 128, 255),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (0, 0, 255),
        (0, 255, 128),
        (128, 255, 0),
    ]
    return palette[int(cls_id) % len(palette)]


def draw_boxes(
    image_path: Path,
    label_path: Path,
    out_path: Path,
    names: Optional[List[str]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
    show: bool = False,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"读取图片失败: {image_path}")

    img_h, img_w = img.shape[:2]
    lines = label_path.read_text(encoding="utf-8").splitlines()

    num = 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        x1, y1, x2, y2 = _yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h)
        color = _color_for_class(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if names is not None:
            cls_name = names[cls_id] if 0 <= cls_id < len(names) else f"cls{cls_id}"
            label = cls_name
            # 背景框 + 文字（避免看不清）
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            y_text = max(0, y1 - th - baseline - 3)
            cv2.rectangle(img, (x1, y_text), (x1 + tw + 4, y_text + th + baseline + 4), color, -1)
            cv2.putText(
                img,
                label,
                (x1 + 2, y_text + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        num += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        raise RuntimeError(f"保存失败: {out_path}")

    print(f"[OK] image={image_path}")
    print(f"[OK] label={label_path} (boxes={num})")
    print(f"[OK] saved={out_path}")

    if show:
        cv2.imshow("yolo_boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw YOLO-format bboxes on an image and save it.")
    parser.add_argument("--img", required=True, help="图片路径，例如: data/images/xxx.jpg")
    parser.add_argument(
        "--label",
        default=None,
        help="标签txt路径（YOLO格式：cls x_center y_center w h）。不填则默认同目录同名 .txt",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出路径：可以是目录(不带扩展名)或具体文件路径。不填则在图片旁边生成 *_boxed.*",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="可选：类别名。支持 1) data.yaml (含 names) 2) txt(每行一个) 3) 逗号分隔字符串",
    )
    parser.add_argument("--thickness", type=int, default=2, help="线宽")
    parser.add_argument("--font-scale", type=float, default=0.6, help="类别文字大小（仅 --names 时生效）")
    parser.add_argument("--show", action="store_true", help="可选：弹窗显示（需要桌面环境）")

    args = parser.parse_args()

    image_path = Path(args.img)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    if args.label is None:
        label_path = image_path.with_suffix(".txt")
    else:
        label_path = Path(args.label)

    if not label_path.exists():
        raise FileNotFoundError(f"标签不存在: {label_path}")

    if args.out is None:
        out_path = image_path.with_name(f"{image_path.stem}_boxed{image_path.suffix}")
    else:
        out_arg = str(args.out)
        outp = Path(out_arg)
        # OpenCV 需要通过文件扩展名确定编码器；因此 --out 不带扩展名时按“目录”处理
        if outp.suffix == "":
            outp.mkdir(parents=True, exist_ok=True)
            out_path = outp / f"{image_path.stem}_boxed{image_path.suffix}"
        else:
            out_path = outp

    names = _load_names(args.names)
    draw_boxes(
        image_path=image_path,
        label_path=label_path,
        out_path=out_path,
        names=names,
        thickness=args.thickness,
        font_scale=args.font_scale,
        show=args.show,
    )


if __name__ == "__main__":
    main()

# coding:utf-8
"""
JMT-YOLO 单图推理引擎
用于 Web 端在线推理功能
"""

import os
import sys
import time
import base64
import cv2
import numpy as np
import torch

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.datasets import letterbox

# 三种数据集的类别标签
DATASET_CLASSES = {
    'cityscapes': ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck'],
    'kitti': ['background', 'car'],
    'voc': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
}

# 每个类别的颜色 (BGR 格式)
COLORS = [
    (255, 56, 56),    # 红
    (255, 157, 151),  # 粉
    (255, 112, 31),   # 橙
    (255, 178, 29),   # 黄
    (207, 210, 49),   # 黄绿
    (72, 249, 10),    # 绿
    (146, 204, 23),   # 草绿
    (61, 219, 134),   # 青绿
    (26, 147, 52),    # 深绿
    (0, 212, 187),    # 青
    (44, 153, 168),   # 青蓝
    (0, 194, 255),    # 天蓝
    (52, 69, 147),    # 深蓝
    (100, 115, 255),  # 蓝紫
    (0, 24, 236),     # 蓝
    (132, 56, 255),   # 紫
    (82, 0, 133),     # 深紫
    (203, 56, 255),   # 紫红
    (255, 149, 200),  # 粉红
    (255, 55, 199),   # 玫红
    (255, 111, 221),  # 浅粉
]


class YOLODetector:
    """YOLO 检测器封装类"""
    
    def __init__(self, weights: str, device: str = '', img_size: int = 640):
        """
        初始化检测器
        
        Args:
            weights: 模型权重路径
            device: 设备 ('0', '0,1', 'cpu')
            img_size: 推理图片尺寸
        """
        self.device = select_device(device)
        self.model = None
        self.weights = None
        self.img_size = img_size
        self.stride = 32
        self.num_params = 0
        
        if weights:
            self.load_model(weights, img_size)
    
    def load_model(self, weights: str, img_size: int = 640):
        """加载模型权重"""
        if self.weights == weights and self.model is not None:
            return  # 模型已加载，无需重复加载
        
        print(f"[Inference Engine] Loading model: {weights}")
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        self.weights = weights
        
        # 计算模型参数量
        self.num_params = sum(p.numel() for p in self.model.parameters())
        
        # 半精度推理 (仅 CUDA)
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        
        # 预热
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))
        
        print(f"[Inference Engine] Model loaded. Params: {self.num_params / 1e6:.2f}M")
    
    def detect(self, img_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45, 
               dataset: str = 'cityscapes') -> dict:
        """
        对单张图片进行检测
        
        Args:
            img_path: 图片路径
            conf_thres: 置信度阈值
            iou_thres: NMS IoU 阈值
            dataset: 数据集类型 ('cityscapes', 'kitti', 'voc')
        
        Returns:
            dict: 检测结果，包含 result_image_base64, inference_time, fps, objects 等
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 获取类别标签
        class_names = DATASET_CLASSES.get(dataset, DATASET_CLASSES['cityscapes'])
        
        # 读取图片
        img0 = cv2.imread(img_path)
        if img0 is None:
            raise ValueError(f"Failed to read image: {img_path}")
        
        # 预处理
        img = letterbox(img0, self.img_size, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 推理 (与 ssda_yolov5_test.py 第 174-190 行保持一致)
        t1 = time.time()
        with torch.no_grad():
            # JMT-YOLO 模型输出格式:
            # out, train_out = model(img, augment=augment)
            # out = (nms_out, det_out)
            # nms_out 是 tensor，用于 NMS
            model_output = self.model(img, augment=False)
            
            # 解包模型输出
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                out = model_output[0]  # inference output
                # train_out = model_output[1]  # training output (不需要)
            else:
                out = model_output
            
            # 从 out 中提取 nms_out
            if isinstance(out, tuple) and len(out) >= 1:
                nms_out = out[0]  # nms_out tensor
            else:
                nms_out = out
            
            # 确保 nms_out 是 tensor（防止嵌套 tuple）
            while isinstance(nms_out, (tuple, list)) and len(nms_out) > 0:
                nms_out = nms_out[0]
        
        # NMS (与 ssda_yolov5_test.py 第 190 行保持一致)
        pred = non_max_suppression(nms_out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)
        t2 = time.time()
        
        inference_time = (t2 - t1) * 1000  # ms
        fps = 1000.0 / inference_time if inference_time > 0 else 0
        
        # 处理检测结果
        objects = []
        det = pred[0]  # 单张图片的检测结果
        
        if len(det):
            # 缩放坐标到原图尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            for *xyxy, conf, cls in reversed(det):
                cls_id = int(cls)
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
                confidence = float(conf)
                bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                
                objects.append({
                    'class': cls_name,
                    'class_id': cls_id,
                    'conf': confidence,
                    'bbox': bbox
                })
                
                # 绘制检测框
                color = COLORS[cls_id % len(COLORS)]
                label = f'{cls_name} {confidence:.2f}'
                self._draw_box(img0, bbox, label, color)
        
        # 将结果图片转为 Base64
        _, buffer = cv2.imencode('.jpg', img0, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'result_image_base64': result_image_base64,
            'inference_time': round(inference_time, 2),
            'fps': round(fps, 1),
            'model_params': f'{self.num_params / 1e6:.1f}M',
            'objects': objects,
            'num_objects': len(objects)
        }
    
    def _draw_box(self, img: np.ndarray, bbox: list, label: str, color: tuple):
        """在图片上绘制检测框和标签"""
        x1, y1, x2, y2 = bbox
        
        # 绘制边框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 标签位置
        label_y1 = max(y1 - text_h - 8, 0)
        label_y2 = y1
        
        cv2.rectangle(img, (x1, label_y1), (x1 + text_w + 4, label_y2), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y2 - 4), font, font_scale, (255, 255, 255), thickness)


# 全局检测器实例 (用于模型缓存)
_detector_instance = None


def get_detector(weights: str = None, device: str = '', img_size: int = 640) -> YOLODetector:
    """获取检测器实例（单例模式，支持模型切换）"""
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = YOLODetector(weights, device, img_size)
    elif weights and _detector_instance.weights != weights:
        _detector_instance.load_model(weights, img_size)
    
    return _detector_instance


def run_inference(img_path: str, weights: str, dataset: str = 'cityscapes',
                  conf_thres: float = 0.25, iou_thres: float = 0.45,
                  img_size: int = 640, device: str = '') -> dict:
    """
    运行推理的便捷函数
    
    Args:
        img_path: 图片路径
        weights: 模型权重路径
        dataset: 数据集类型
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值
        img_size: 推理图片尺寸
        device: 设备
    
    Returns:
        dict: 检测结果
    """
    detector = get_detector(weights, device, img_size)
    return detector.detect(img_path, conf_thres, iou_thres, dataset)


if __name__ == '__main__':
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--source', type=str, required=True, help='Image path')
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'kitti', 'voc'])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    
    result = run_inference(
        img_path=args.source,
        weights=args.weights,
        dataset=args.dataset,
        conf_thres=args.conf_thres,
        img_size=args.img_size,
        device=args.device
    )
    
    print(f"Inference time: {result['inference_time']} ms")
    print(f"FPS: {result['fps']}")
    print(f"Objects detected: {result['num_objects']}")
    for obj in result['objects']:
        print(f"  - {obj['class']}: {obj['conf']:.2f}")

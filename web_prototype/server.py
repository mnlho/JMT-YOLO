# coding:utf-8
"""
JMT-YOLO Web 后端服务
提供训练日志读取、在线推理等 API
"""

import os
import sys
import time
import shutil
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 配置部分
WORKSPACE_DIR = PROJECT_ROOT
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# 默认路径配置
DEFAULT_RESULTS_FILE = os.path.join(WORKSPACE_DIR, "runs/train/12_21_bestpt_maskratio_03.6/results.txt")
DEFAULT_WEIGHTS = os.path.join(WORKSPACE_DIR, "runs/train/12_21_bestpt_maskratio_03.6/weights/best_student.pt")

# 三种数据集的类别标签
DATASET_CLASSES = {
    'cityscapes': ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck'],
    'kitti': ['background', 'car'],
    'voc': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
}

app = FastAPI(title="JMT-YOLO API", version="1.0.0")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainConfig(BaseModel):
    epoch: int
    batch: int


@app.get("/")
def root():
    """API 根路径"""
    return {"message": "JMT-YOLO API Server", "version": "1.0.0"}


@app.get("/train/logs")
def get_training_logs(results_file: str = DEFAULT_RESULTS_FILE):
    """
    读取训练日志文件 (results.txt) 并解析为 JSON
    
    Args:
        results_file: results.txt 文件路径
    """
    data = []
    
    if not os.path.exists(results_file):
        return {"error": f"File not found: {results_file}", "data": []}
    
    try:
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 17:
                continue
            
            try:
                epoch_part = parts[0].split('/')
                epoch = int(epoch_part[0]) + 1
                total_epochs = int(epoch_part[1]) + 1
                
                data.append({
                    "epoch": epoch,
                    "totalEpochs": total_epochs,
                    "gpuMemory": parts[1],
                    "trainBoxLoss": float(parts[2]),
                    "trainObjLoss": float(parts[3]),
                    "trainClsLoss": float(parts[4]),
                    "trainTotalLoss": float(parts[5]),
                    "consLoss": float(parts[6]),
                    "labelsCount": int(parts[7]),
                    "imgSize": int(parts[8]),
                    "precision": float(parts[9]),
                    "recall": float(parts[10]),
                    "mAP50": float(parts[11]),
                    "mAP75": float(parts[12]),
                    "mAP5095": float(parts[13]),
                    "valBoxLoss": float(parts[14]),
                    "valObjLoss": float(parts[15]),
                    "valClsLoss": float(parts[16]),
                })
            except (ValueError, IndexError) as e:
                continue
                
    except Exception as e:
        return {"error": str(e), "data": []}
    
    return data


@app.post("/inference")
async def run_inference(
    file: UploadFile = File(...),
    weights: str = Form(DEFAULT_WEIGHTS),
    dataset: str = Form("cityscapes"),
    conf_thres: float = Form(0.25),
    img_size: int = Form(640),
    device: str = Form("")
):
    """
    执行目标检测推理
    
    Args:
        file: 上传的图片文件
        weights: 模型权重路径
        dataset: 数据集类型 (cityscapes, kitti, voc)
        conf_thres: 置信度阈值
        img_size: 推理图片尺寸
        device: 设备 (空字符串=自动选择, '0'=GPU 0, 'cpu'=CPU)
    
    Returns:
        检测结果，包含带框图片的 Base64、推理时间、检测目标列表等
    """
    # 验证数据集类型
    if dataset not in DATASET_CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid dataset: {dataset}. Must be one of {list(DATASET_CLASSES.keys())}")
    
    # 验证权重文件
    if not os.path.exists(weights):
        raise HTTPException(status_code=400, detail=f"Weights file not found: {weights}")
    
    # 保存上传的图片
    timestamp = int(time.time() * 1000)
    file_ext = os.path.splitext(file.filename)[1] or '.jpg'
    temp_filename = f"upload_{timestamp}{file_ext}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 导入推理引擎并执行检测
        from inference_engine import run_inference as detect
        
        result = detect(
            img_path=temp_path,
            weights=weights,
            dataset=dataset,
            conf_thres=conf_thres,
            img_size=img_size,
            device=device
        )
        
        return {
            "success": True,
            "inferenceTime": result['inference_time'],
            "fps": result['fps'],
            "modelParams": result['model_params'],
            "objects": result['objects'],
            "numObjects": result['num_objects'],
            "resultImageBase64": result['result_image_base64']
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[Server Error] Inference failed:\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.get("/models")
def list_models():
    """
    列出可用的模型权重文件
    """
    models = []
    runs_dir = os.path.join(WORKSPACE_DIR, "runs/train")
    
    if os.path.exists(runs_dir):
        for exp_name in os.listdir(runs_dir):
            weights_dir = os.path.join(runs_dir, exp_name, "weights")
            if os.path.exists(weights_dir):
                for weight_file in os.listdir(weights_dir):
                    if weight_file.endswith('.pt'):
                        weight_path = os.path.join(weights_dir, weight_file)
                        models.append({
                            "name": f"{exp_name}/{weight_file}",
                            "path": weight_path,
                            "size": os.path.getsize(weight_path) / (1024 * 1024)  # MB
                        })
    
    return {"models": models}


@app.get("/datasets")
def list_datasets():
    """
    列出支持的数据集类型及其类别
    """
    return {
        "datasets": [
            {
                "id": "cityscapes",
                "name": "Cityscapes → Foggy",
                "classes": DATASET_CLASSES['cityscapes'],
                "num_classes": len(DATASET_CLASSES['cityscapes'])
            },
            {
                "id": "kitti",
                "name": "KITTI → CityScapes", 
                "classes": DATASET_CLASSES['kitti'],
                "num_classes": len(DATASET_CLASSES['kitti'])
            },
            {
                "id": "voc",
                "name": "VOC → Clipart",
                "classes": DATASET_CLASSES['voc'],
                "num_classes": len(DATASET_CLASSES['voc'])
            }
        ]
    }


@app.get("/model/download")
def download_model(weights: str = DEFAULT_WEIGHTS):
    """
    下载模型权重文件
    """
    if not os.path.exists(weights):
        raise HTTPException(status_code=404, detail=f"Model not found: {weights}")
    
    return FileResponse(
        path=weights,
        filename=os.path.basename(weights),
        media_type='application/octet-stream'
    )


@app.delete("/temp/cleanup")
def cleanup_temp():
    """
    清理临时文件
    """
    try:
        for f in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return {"success": True, "message": "Temp files cleaned up"}
    except Exception as e:
        return {"success": False, "message": str(e)}


class DatasetAnalysisRequest(BaseModel):
    source_path: str
    target_path: str


@app.post("/data/analyze")
def analyze_dataset(request: DatasetAnalysisRequest):
    """
    分析源域和目标域数据集
    
    Args:
        source_path: 源域数据集路径 (应包含 images/ 和 labels/ 子目录)
        target_path: 目标域数据集路径
    
    Returns:
        数据集统计信息
    """
    def count_files(base_path: str, subdir: str, extensions: list):
        """统计指定目录下的文件数量"""
        full_path = os.path.join(base_path, subdir)
        if not os.path.exists(full_path):
            # 尝试直接在 base_path 下查找
            full_path = base_path
            if not os.path.exists(full_path):
                return 0
        
        count = 0
        for root, dirs, files in os.walk(full_path):
            for f in files:
                if any(f.lower().endswith(ext) for ext in extensions):
                    count += 1
        return count
    
    def parse_labels(base_path: str):
        """解析标签文件，统计各类别目标数量"""
        labels_path = os.path.join(base_path, "labels")
        if not os.path.exists(labels_path):
            labels_path = base_path
        
        class_counts = {}
        total_objects = 0
        
        if os.path.exists(labels_path):
            for root, dirs, files in os.walk(labels_path):
                for f in files:
                    if f.endswith('.txt'):
                        try:
                            with open(os.path.join(root, f), 'r') as label_file:
                                for line in label_file:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                        total_objects += 1
                        except:
                            continue
        
        return class_counts, total_objects
    
    try:
        # 统计源域
        source_images = count_files(request.source_path, "images", ['.jpg', '.jpeg', '.png', '.bmp'])
        source_labels = count_files(request.source_path, "labels", ['.txt'])
        source_class_counts, source_total = parse_labels(request.source_path)
        
        # 统计目标域
        target_images = count_files(request.target_path, "images", ['.jpg', '.jpeg', '.png', '.bmp'])
        target_labels = count_files(request.target_path, "labels", ['.txt'])
        target_class_counts, target_total = parse_labels(request.target_path)
        
        # 获取类别名称 (默认使用 cityscapes)
        class_names = DATASET_CLASSES['cityscapes']
        
        return {
            "success": True,
            "source": {
                "path": request.source_path,
                "imageCount": source_images,
                "labelCount": source_labels,
                "totalObjects": source_total,
                "avgObjectsPerImage": round(source_total / max(source_images, 1), 2),
                "classDistribution": [
                    {"classId": cid, "name": class_names[cid] if cid < len(class_names) else f"class_{cid}", "count": cnt}
                    for cid, cnt in sorted(source_class_counts.items())
                ]
            },
            "target": {
                "path": request.target_path,
                "imageCount": target_images,
                "labelCount": target_labels,
                "totalObjects": target_total,
                "avgObjectsPerImage": round(target_total / max(target_images, 1), 2),
                "classDistribution": [
                    {"classId": cid, "name": class_names[cid] if cid < len(class_names) else f"class_{cid}", "count": cnt}
                    for cid, cnt in sorted(target_class_counts.items())
                ]
            },
            "classes": class_names,
            "numClasses": len(class_names)
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[Server Error] Dataset analysis failed:\n{error_detail}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print(f"[Server] Workspace: {WORKSPACE_DIR}")
    print(f"[Server] Default weights: {DEFAULT_WEIGHTS}")
    print(f"[Server] Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

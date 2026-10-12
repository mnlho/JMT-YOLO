# JMT-YOLO 跨域目标检测原型系统技术报告

## 1. 系统概述

### 1.1 系统定位

本系统是面向跨域场景的域自适应目标检测原型系统，旨在为 JMT-YOLO 算法提供完整的可视化交互平台。系统采用 B/S 架构，支持数据分析、模型训练监控和在线推理三大核心功能。

### 1.2 技术栈

| 层次 | 技术选型 | 版本 |
|------|---------|------|
| **前端框架** | React + TypeScript | React 19, TS 5.6 |
| **UI 组件库** | Ant Design | 6.x |
| **图表库** | ECharts | 6.x |
| **状态管理** | Zustand | 5.x |
| **构建工具** | Vite | 7.x |
| **后端框架** | FastAPI | 0.115+ |
| **深度学习** | PyTorch | 1.x / 2.x |
| **图像处理** | OpenCV | 4.x |

### 1.3 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户界面层                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│  │   数据分析模块   │ │   训练监控模块   │ │   在线推理模块   │        │
│  │ DataProcessing │ │    Training     │ │    Inference    │        │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘        │
└───────────┼────────────────────┼────────────────────┼────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           服务接口层                                  │
│                    FastAPI RESTful API Server                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │
│  │/data/analyze │ │ /train/logs  │ │  /inference  │ │  /models   │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           算法核心层                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     JMT-YOLO 检测模型                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │   Backbone  │→│    Neck     │→│  Detection Head     │  │   │
│  │  │  CSPDarknet │  │ PANet+DPLA │  │  + Domain Adapter   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 功能模块详细设计

### 2.1 数据处理模块套件 (DataProcessing Suite)

本模块包含数据分析、SAM 辅助标注和风格迁移三大功能，为跨域目标检测提供全流程的数据准备与优化工具。

#### 2.1.1 数据分析 (DataAnalysis.tsx)

**功能描述**:
- 支持配置源域和目标域数据集路径。
- 自动分析统计信息：图像张数、标注目标数、平均目标数等。
- 可视化展示类别分布对比（双系列柱状图）。
- 域差异分析：五维雷达图（亮度、对比度、清晰度、饱和度、纹理）。
- 样本对比：直观展示源域与目标域在视觉上的差异。

#### 2.1.2 SAM 辅助标注 (SAMSegmentation.tsx)

**功能描述**:
- 集成 Segment Anything Model (SAM) 进行高质量分割掩码生成。
- **Prompt 模式**:
    - **Box Prompt**: 从现有检测框直接生成分割掩码（推荐）。
    - **Text Prompt**: 通过类别文本提示生成（如 "car"）。
    - **Auto**: 自动执行 YOLO 检测并级联 SAM 进行全图分割。
- **实时反馈**: 展示原始图、引导框与生成的分割结果对比。
- **统计分析**: 实时计算单张图像的检测目标数与掩码质量分布。

#### 2.1.3 风格迁移 (StyleTransfer.tsx)

**功能描述**:
- 基于 CUT 算法实现跨域图像风格转换（如 Cityscapes → KITTI）。
- **迁移方向**: 支持 S→T、T→S 以及双向同步迁移。
- **批处理控制**: 可灵活配置分辨率、批大小 (Batch Size) 等推理参数。
- **对比展示**: 左右镜像展示源域图像与迁移后的生成图像，便于效果验证。
- **任务管理**: 支持任务队列化执行，实时监控处理进度与系统日志。

---

### 2.2 训练监控模块 (Training.tsx)

#### 功能描述

- 配置训练超参数 (Epochs, Batch, 学习率等)
- 配置域自适应参数 (蒸馏系数λ, 一致性损失系数α, Masking Ratio等)
- 实时展示训练曲线 (Train Loss, Val Loss, mAP)
- 滚动显示训练日志
- 持久化训练状态 (跨页面保持)

#### 训练参数配置

| 参数 | 默认值 | 范围 | 说明 |
|------|-------|------|------|
| Epochs | 200 | 1-1000 | 训练轮数 |
| Batch Size | 2 | 1-64 | 批量大小 |
| Image Size | 960 | 320-1280 | 输入图像尺寸 |
| λ (蒸馏系数) | 0.005 | 0-0.2 | 教师-学生蒸馏损失权重 |
| α (一致性系数) | 2.0 | 0-5 | 一致性约束损失权重 |
| Teacher EMA α | 0.99 | 0.9-0.999 | 教师模型 EMA 更新系数 |
| Mask Ratio | 0.6 | 0-1 | 随机遮挡比例 |
| Mask Block Size | 64 | 16-128 | 遮挡块大小 (像素) |

#### 训练数据格式 (results.txt 17列)

```
Epoch/Total | GPU_Mem | box_loss | obj_loss | cls_loss | total_loss | cons_loss | labels | img_size | P | R | mAP@.5 | mAP@.75 | mAP@.5:.95 | val_box | val_obj | val_cls
```

#### 状态管理 (Zustand)

```typescript
interface TrainingState {
  isTraining: boolean;
  currentEpoch: number;
  totalEpochs: number;
  currentMap50: number;
  currentMap5095: number;
  precision: number;
  recall: number;
  trainTotalLoss: number;
  valTotalLoss: number;
  consLoss: number;
  gpuMemory: string;
  historyData: TrainingDataPoint[];
  logs: string[];
}
```

#### 可视化图表

1. **Train Loss 曲线**: Box/Obj/Cls 三条损失曲线
2. **Val Loss 曲线**: 验证集损失
3. **mAP & Metrics 曲线**: mAP@0.5, mAP@0.75, mAP@0.5:0.95, Precision, Recall

---

### 2.3 在线推理模块 (Inference.tsx)

#### 功能描述

- 支持拖拽/点击上传多张图片
- 选择不同的数据集配置 (Cityscapes/KITTI/VOC)
- 选择不同的模型权重
- 调整置信度阈值和图像尺寸
- 展示检测结果 (带框图片、推理时间、检测目标列表)
- 详情抽屉查看边界框坐标

#### 推理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| dataset | string | cityscapes | 数据集类型 |
| weights | string | best_student.pt | 模型权重路径 |
| conf_thres | float | 0.25 | 置信度阈值 |
| img_size | int | 960 | 推理图像尺寸 |

#### 检测结果数据结构

```typescript
interface DetectionResult {
  fileUrl: string;              // 结果图片 URL (Base64)
  fileName: string;             // 原文件名
  inferenceTime: number;        // 推理耗时 (ms)
  imgSize: number;              // 图像尺寸
  modelParams: string;          // 模型参数量 (如 "46.5M")
  numObjects: number;           // 检测目标数
  objects: Array<{
    class: string;              // 类别名称
    class_id: number;           // 类别 ID
    conf: number;               // 置信度
    bbox: [x1, y1, x2, y2];     // 边界框坐标
  }>;
}
```

---

## 3. 后端 API 接口文档

### 3.1 接口列表

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | API 状态检查 |
| GET | `/train/logs` | 获取训练日志 |
| POST | `/inference` | 执行目标检测 |
| GET | `/models` | 列出可用模型 |
| GET | `/datasets` | 列出支持的数据集 |
| GET | `/model/download` | 下载模型权重 |
| POST | `/data/analyze` | 分析数据集 |
| DELETE | `/temp/cleanup` | 清理临时文件 |

### 3.2 核心接口详情

#### POST /inference

**请求参数** (FormData):

```
file: File          # 上传的图片文件
weights: string     # 模型权重路径
dataset: string     # 数据集类型 (cityscapes/kitti/voc)
conf_thres: float   # 置信度阈值
img_size: int       # 推理图像尺寸
```

**响应示例**:

```json
{
  "success": true,
  "inferenceTime": 45.32,
  "fps": 22.1,
  "modelParams": "46.5M",
  "numObjects": 12,
  "objects": [
    {"class": "car", "class_id": 2, "conf": 0.92, "bbox": [120, 340, 450, 520]},
    {"class": "person", "class_id": 4, "conf": 0.87, "bbox": [80, 200, 150, 480]}
  ],
  "resultImageBase64": "/9j/4AAQSkZJRgABAQAAAQ..."
}
```

#### GET /train/logs

**请求参数**:

```
results_file: string  # results.txt 文件路径 (可选)
```

**响应示例**:

```json
[
  {
    "epoch": 1,
    "totalEpochs": 200,
    "gpuMemory": "8.5G",
    "trainBoxLoss": 0.0452,
    "trainObjLoss": 0.0312,
    "trainClsLoss": 0.0089,
    "trainTotalLoss": 0.0853,
    "consLoss": 1.23e-05,
    "precision": 0.4521,
    "recall": 0.3892,
    "mAP50": 0.4123,
    "mAP75": 0.2156,
    "mAP5095": 0.2234
  }
]
```

#### POST /data/analyze

**请求体**:

```json
{
  "source_path": "/path/to/source/dataset",
  "target_path": "/path/to/target/dataset"
}
```

**响应示例**:

```json
{
  "success": true,
  "source": {
    "path": "/path/to/source",
    "imageCount": 2975,
    "labelCount": 2975,
    "totalObjects": 50588,
    "avgObjectsPerImage": 17.01,
    "classDistribution": [
      {"classId": 0, "name": "bus", "count": 386},
      {"classId": 2, "name": "car", "count": 26963}
    ]
  },
  "target": { /* 同 source */ },
  "classes": ["bus", "bicycle", "car", ...],
  "numClasses": 8
}
```

---

## 4. 推理引擎实现 (inference_engine.py)

### 4.1 核心类: YOLODetector

```python
class YOLODetector:
    """YOLO 检测器封装类 - 单例模式支持模型缓存"""
    
    def __init__(self, weights, device='', img_size=640):
        self.device = select_device(device)
        self.model = None
        self.weights = None
        
    def load_model(self, weights, img_size=640):
        """加载模型权重 (支持热切换)"""
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        # 半精度推理优化
        if self.device.type != 'cpu':
            self.model.half()
    
    def detect(self, img_path, conf_thres=0.25, iou_thres=0.45, dataset='cityscapes'):
        """执行检测并返回带框图片的 Base64"""
        # 1. 图像预处理 (letterbox resize)
        # 2. 模型推理
        # 3. NMS 后处理
        # 4. 坐标映射回原图
        # 5. 绘制检测框
        # 6. 编码为 Base64
        return result
```

### 4.2 模型输出处理

JMT-YOLO 模型输出格式特殊，需要正确解包：

```python
# JMT-YOLO 输出: (out, train_out)
# 其中 out = (nms_out, det_out)
model_output = self.model(img)
out = model_output[0]      # inference output
nms_out = out[0]           # tensor for NMS

# 防止嵌套 tuple
while isinstance(nms_out, (tuple, list)):
    nms_out = nms_out[0]

pred = non_max_suppression(nms_out, conf_thres, iou_thres)
```

### 4.3 检测框绘制

```python
def _draw_box(self, img, bbox, label, color):
    """在图片上绘制检测框和标签"""
    x1, y1, x2, y2 = bbox
    
    # 绘制边框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 绘制标签背景
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 1)
    cv2.rectangle(img, (x1, y1-text_h-8), (x1+text_w+4, y1), color, -1)
    cv2.putText(img, label, (x1+2, y1-4), font, 0.6, (255,255,255), 1)
```

---

## 5. 支持的跨域数据集配置

| 适应任务 | 源域 | 目标域 | 类别数 |
|---------|------|-------|-------|
| Cityscapes → Foggy | Cityscapes (晴朗) | Foggy Cityscapes (雾天) | 8 |
| KITTI → Cityscapes | KITTI | Cityscapes | 1 (car) |
| VOC → Clipart | Pascal VOC 07+12 | Clipart1k | 20 |

### 类别标签定义

```python
DATASET_CLASSES = {
    'cityscapes': ['bus', 'bicycle', 'car', 'motorcycle', 
                   'person', 'rider', 'train', 'truck'],
    'kitti': ['background', 'car'],
    'voc': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
}
```

---

## 6. 前端状态管理 (appStore.ts)

### 6.1 全局状态定义

```typescript
interface AppState {
  // 模式切换
  isLiveMode: boolean;
  toggleMode: () => void;
  
  // 训练状态 (持久化)
  training: TrainingState;
  setTraining: (state: Partial<TrainingState>) => void;
  addLog: (log: string) => void;
  addHistoryPoint: (point: TrainingDataPoint) => void;
  resetTraining: () => void;
}
```

### 6.2 训练数据点类型

```typescript
interface TrainingDataPoint {
  epoch: number;
  totalEpochs: number;
  gpuMemory: string;
  trainBoxLoss: number;
  trainObjLoss: number;
  trainClsLoss: number;
  trainTotalLoss: number;
  consLoss: number;
  labelsCount: number;
  imgSize: number;
  precision: number;
  recall: number;
  mAP50: number;
  mAP75: number;
  mAP5095: number;
  valBoxLoss: number;
  valObjLoss: number;
  valClsLoss: number;
}
```

---

## 7. 部署与运行

### 7.1 环境要求

- Python 3.8+
- Node.js 18+
- CUDA 11.x (GPU 推理)
- 8GB+ GPU 显存 (推荐)

### 7.2 依赖安装

**后端依赖**:

```bash
pip install fastapi uvicorn python-multipart
pip install torch torchvision  # PyTorch
pip install opencv-python numpy
```

**前端依赖**:

```bash
npm install
```

### 7.3 一键启动脚本

```bash
# 启动系统
cd web_prototype
./start.sh

# 停止系统
./stop.sh
```

### 7.4 手动启动

```bash
# 后端 (端口 8000)
conda activate zjh_SSDA_web
python server.py

# 前端 (端口 5173)
npm run dev
```

### 7.5 访问地址

- 前端界面: http://localhost:5173
- 后端 API: http://localhost:8000
- API 文档: http://localhost:8000/docs

---

## 8. 系统性能指标

| 指标 | 数值 |
|------|------|
| 模型参数量 | 46.5M |
| 单图推理时间 (GPU) | 30-50ms |
| 单图推理时间 (CPU) | 200-400ms |
| mAP@0.5 (Cityscapes→Foggy) | 51.2% |
| mAP@0.5:0.95 (Cityscapes→Foggy) | 27.8% |
| 前端首屏加载时间 | <2s |
| API 响应时间 (非推理) | <50ms |

---

## 9. 文件结构

```
web_prototype/
├── src/
│   ├── pages/
│   │   ├── DataProcessing/      # 数据处理套件目录
│   │   │   ├── DataAnalysis.tsx   # 数据分析子页面
│   │   │   ├── SAMSegmentation.tsx # SAM 辅助标注子页面
│   │   │   ├── StyleTransfer.tsx  # 风格迁移子页面
│   │   │   └── index.tsx          # 入口路由
│   │   ├── Training.tsx         # 训练监控页面
│   │   └── Inference.tsx        # 在线推理页面
│   ├── layout/
│   │   └── MainLayout.tsx       # 主布局组件
│   ├── store/
│   │   └── appStore.ts          # Zustand 状态管理
│   ├── api/
│   │   └── client.ts            # API 客户端
│   ├── utils/
│   │   └── mockData.ts          # Mock 数据工具
│   ├── App.tsx                  # 路由配置
│   └── main.tsx                 # 入口文件
├── public/
│   └── samples/                 # 示例图片
├── server.py                    # FastAPI 后端服务
├── inference_engine.py          # 推理引擎
├── start.sh                     # 启动脚本
├── stop.sh                      # 停止脚本
├── package.json                 # 前端依赖
└── TECHNICAL_REPORT.md          # 本技术报告
```

---

## 10. 参考文献

1. YOLOv5: https://github.com/ultralytics/yolov5
2. Cityscapes Dataset: https://www.cityscapes-dataset.com/
3. Foggy Cityscapes: https://www.cityscapes-dataset.com/foggydownload/
4. React: https://react.dev/
5. Ant Design: https://ant.design/
6. FastAPI: https://fastapi.tiangolo.com/
7. ECharts: https://echarts.apache.org/

---

*文档生成时间: 2026-01-28*

*版本: v1.0.0*

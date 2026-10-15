# JMT-YOLO
JMT-YOLO

# JMT-YOLO / SPL

本仓库包含基于YOLO的教师–学生域适应目标检测代码，以及SPL的DAAT自适应伪标签阈值、DPLA 动态伪标签分配模块。仓库同时提供统一设置下的训练、伪标签质量分析、效率测量、多随机种子统计和模块消融工具。

以下命令在仓库根目录执行。示例使用 Cityscapes → Foggy Cityscapes（`c2f`）；数据集、风格转换图像和预训练权重需自行准备。
<img width="990" height="557" alt="屏幕截图 2026-10-15 113652" src="https://github.com/user-attachments/assets/d8a30505-1edb-4dde-bab5-860316eb098d" />

1. 环境安装

实际实验环境：Ubuntu、NVIDIA RTX 3090、Python 3.8、PyTorch 2.4。CUDA 版本需与安装的 PyTorch 及显卡驱动匹配。

```bash
conda create -n jmt_yolo python=3.8 -y
conda activate jmt_yolo

# 先安装与本机 CUDA 环境匹配的 PyTorch / torchvision，再安装其余依赖
python -m pip install -r requirements.txt
python -m pip install kornia==0.6.12

python -c "import torch, kornia; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`requirements.txt` 为原项目依赖列表，并未锁定所有包的版本。复现实验时请额外保存环境：

```bash
python -m pip freeze > environment_pip.txt
```

2. 数据与权重准备

将 YOLOv5-L 预训练权重放到 `weights/yolov5l.pt`。下面的路径仅为示例，可替换为实际位置：

```bash
export DATA_ROOT=/path/to/dataset
export WEIGHTS=/path/to/yolov5l.pt
```

`c2f` 数据配置包含以下内容：

|          配置键         | 内容                                                     
| ----------------------- | -------------------------------------------------------- 
| `train_source_real`     | 源域原始训练图像                                         
| `train_source_fake`     | 源域转换为目标域风格的训练图像                           
| `train_target_real`     | 目标域原始训练图像                                       
| `train_target_fake`     | 目标域转换为源域风格的训练图像                           
| `train_source_seg_real` | 与源域图像对应的分割图像树
| `test_target_real`      | 目标域验证图像                                           

检测图像、检测标注及分割掩码采用平行目录，例如：

```text
dataset/
├── leftImg8bit_trainvaltest/
│   ├── yolov5_format/
│   │   ├── images/train/<image_id>.png
│   │   └── labels/train/<image_id>.txt
│   └── segData/
│       ├── images/train/<image_id>.png
│       └── labels/train/<image_id>.png
├── CityScapes2CityScapesFoggy/...
├── leftImg8bit_trainval_foggyDBF/...
└── CityScapesFoggy2CityScapes/...
```
检测标注为 YOLO 格式：每行 `class_id x_center y_center width height`，坐标归一化到 `[0, 1]`。图像文件名不含扩展名的部分作为配对 ID；同一集合内 ID 必须唯一，成对图像及源域掩码必须对应。路径中需保留 `images` 目录，程序通过替换为 `labels` 查找标注。

`c2f` 检测类别顺序为：`bus, bicycle, car, motorcycle, person, rider, train, truck`。分割类别数为 9；需使用符合配置的类别编码。

生成配置并检查数据：

```bash
python revision_run.py prepare --task c2f --data-root "$DATA_ROOT"

# 检查并按实际数据位置修改 revision_configs/c2f.yaml 中的全部路径
python revision_run.py check --data revision_configs/c2f.yaml --weights "$WEIGHTS"
```

3. 致谢

本项目基于YOLO及仓库保留的相关组件。各组件的作者信息和许可声明以对应源文件及上游项目为准。

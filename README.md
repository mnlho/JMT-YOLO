# SPL
SPL

#SPL

   This repository contains teacher-student domain adaptation object detection code based on YOLO, as well as the DAAT adaptive pseudo-label threshold of SPL and the DPLA dynamic pseudo-label assignment module.

<img width="990" height="557" alt="屏幕截图 2026-10-15 113652" src="https://github.com/user-attachments/assets/d8a30505-1edb-4dde-bab5-860316eb098d" />

   The following commands are executed in the root directory of the repository. The example uses Cityscapes → Foggy Cityscapes (`c2f`); the dataset, style-converted images, and pre-trained weights need to be prepared by yourself.

1. Environment Installation

   Actual experimental environment: Ubuntu, NVIDIA RTX 3090, Python 3.8, PyTorch 2.4. The CUDA version needs to match the installed PyTorch and graphics card driver.

```bash
conda create -n jmt_yolo python=3.8 -y
conda activate jmt_yolo

# First, install PyTorch / torchvision that matches the CUDA environment on this machine, and then install the remaining dependencies
python -m pip install -r requirements.txt
python -m pip install kornia==0.6.12

python -c "import torch, kornia; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`requirements.txt` is the dependency list of the original project, and it does not lock the versions of all packages. Please save the environment separately when reproducing the experiment:

```bash
python -m pip freeze > environment_pip.txt
```

2. Data and weight preparation
   
   Place the pre-trained weights of YOLOv5-L into `weights/yolov5l.pt`. The path below is just an example and can be replaced with the actual location:

```bash
export DATA_ROOT=/path/to/dataset
export WEIGHTS=/path/to/yolov5l.pt
```

   The `c2f` data configuration includes the following content:

|    Configuration Key    | Content                                                     
| ----------------------- | -------------------------------------------------------- 
| `train_source_real`     | Original training images in the source domain                                         
| `train_source_fake`     | Training images converted from the source domain to the target domain style                           
| `train_target_real`     | Original training images in the target domain                                       
| `train_target_fake`     | Training images converted from the target domain to the source domain style                           
| `train_source_seg_real` | Segmentation image tree corresponding to the source domain image
| `test_target_real`      | Validation images in the target domain
      
   The detection images, detection annotations, and segmentation masks are organized in parallel directories, for example:

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
   The detection annotations are in YOLO format: each line consists of `class_id x_center y_center width height`, with coordinates normalized to `[0, 1]`. The part of the image filename without the extension serves as the pairing ID; IDs within the same set must be unique, and paired images and source domain masks must correspond. The `images` directory needs to be retained in the path, and the program finds annotations by replacing it with `labels`.

   The detection category order for `c2f` is: `bus, bicycle, car, motorcycle, person, rider, train, truck`. The number of segmentation categories is 9; category codes that conform to the configuration are required.

   Generate configuration and check data:

```bash
python revision_run.py prepare --task c2f --data-root "$DATA_ROOT"

# Check and modify all paths in revision_configs/c2f.yaml according to the actual data location

python revision_run.py check --data revision_configs/c2f.yaml --weights "$WEIGHTS"
```
<img width="1477" height="823" alt="image" src="https://github.com/user-attachments/assets/3a79231f-fe2c-4747-8f6c-b6a68fd5d3cc" />


3. Acknowledgements

   This project is based on YOLO and related components retained from the repository. The author information and license notices for each component are subject to the corresponding source files and upstream projects.

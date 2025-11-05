# YOLOv5 Vending — Vending Machine Detection & Recognition

> This repository adapts YOLOv5 for vending machine scenarios. It includes dataset organization examples, training and inference scripts, export utilities, and deployment tips to help you quickly train and deploy a vending-machine object detection model.

## Contents
- Project Overview
- Features
- Requirements
- Installation
- Quick Start
  - Inference (image / video / webcam)
  - Training
- Dataset & Annotation
- Export & Deployment
- Evaluation
- Tips & Best Practices
- Contributing
- License
- Contact

---

## Project Overview
This project provides scripts and documentation to train and deploy YOLOv5-based detectors tailored for vending machine environments: detecting items (drinks, snacks), slot/stock status (empty/full), coin slots, and other vending-specific classes. It is optimized for both offline batch processing and real-time camera applications.

## Features
- YOLOv5-based training and inference (supports yolov5s/m/l)
- Example configs for YOLO (TXT) and COCO (JSON) dataset formats
- Configurable training hyperparameters and augmentations
- Export to ONNX / TensorRT (when supported)
- Single-image, batch, video, and webcam inference
- Tips for edge-device deployment (Jetson, Coral, etc.)

## Requirements
- Python 3.8+
- PyTorch 1.10+ (or appropriate version for your CUDA)
- CUDA 10.2 / 11.x for GPU acceleration (optional)
- Other Python packages listed in requirements.txt
- Optional: TensorRT, ONNX, OpenCV with contrib modules for some pipelines

## Installation
```bash
git clone https://github.com/Ivy-forever18/yolov5-vending.git
cd yolov5-vending

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt

# If requirements.txt is missing, install core deps manually:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# pip install -r requirements.txt
```

## Quick Start

### Inference (single image)
```bash
python detect.py --weights yolov5s.pt --source data/images/test.jpg --conf 0.25 --save-conf --project runs/detect --name vending_demo
```

### Inference (video / webcam)
```bash
# Webcam
python detect.py --weights yolov5s.pt --source 0 --img 640 --conf 0.25

# Video file
python detect.py --weights yolov5s.pt --source data/videos/test.mp4 --img 640 --conf 0.25
```

### Training
Prepare a data config (see Dataset & Annotation) and run:
```bash
python train.py --img 640 --batch 16 --epochs 50 --data data/vending.yaml --weights yolov5s.pt --name vending_exp
```
Common options:
- --img: input image size (e.g., 640)
- --batch: batch size (depends on GPU memory)
- --epochs: number of training epochs
- --data: path to data config YAML
- --weights: initial weights (use pretrained yolov5s.pt or a checkpoint)

## Dataset & Annotation

Supported formats:
1. YOLO TXT (one .txt per image: "class x_center y_center width height" normalized to [0,1])
2. COCO JSON (standard COCO format)

Recommended directory structure for YOLO format:
```
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Example data/vending.yaml:
```yaml
train: ../dataset/images/train
val:   ../dataset/images/val

nc: 5
names: ['drink','snack','coin_slot','empty_slot','other']
```

Label tips:
- Use consistent class names.
- Ensure labels match images one-to-one.
- Keep a validation set for reliable mAP measurement.

## Export & Deployment

Export to ONNX:
```bash
python export.py --weights runs/train/vending_exp/weights/best.pt --img 640 --include onnx
```

Export to TensorRT (if supported and dependencies installed):
```bash
python export.py --weights runs/train/vending_exp/weights/best.pt --img 640 --include engine
```

Deployment notes:
- For edge devices prioritize lightweight models (yolov5s) or quantized/pruned models.
- Test exported models on target hardware and validate accuracy and latency.
- Consider batching and input resizing appropriately for real-time pipelines.

## Evaluation
Use built-in validation scripts to compute mAP, precision, and recall:
```bash
python val.py --weights runs/train/vending_exp/weights/best.pt --data data/vending.yaml --img 640
```

## Tips & Best Practices
- Use Mosaic, MixUp, and appropriate augmentations to improve generalization.
- Handle class imbalance by oversampling under-represented classes or using loss techniques (e.g., focal loss).
- Tune learning rate with linear scaling rules when changing batch size.
- For production, consider model quantization, pruning, or knowledge distillation to reduce size and latency.

## Contributing
- Open an issue to discuss major changes or report bugs.
- Fork the repo → create a feature branch → open a pull request with clear description and reproduction steps.
- Include dataset subset or clear instructions when reporting dataset/label problems.

## License
Add a LICENSE file to declare the repository license (e.g., MIT). This repository currently does not include an explicit license — please add one if you plan to share or redistribute.

## Contact
Repository: https://github.com/Ivy-forever18/yolov5-vending  
For help, open an issue with reproduction steps and logs.

---

# 中文（Chinese）

# YOLOv5 Vending — 自动售货机检测与识别

> 本仓库基于 YOLOv5，面向自动售货机场景，包含数据集组织示例、训练/推理脚本、模型导出与部署建议，方便快速训练与落地部署检测模型。

## 目录
- 项目简介
- 功能
- 环境要求
- 安装
- 快速开始
  - 推理（图片 / 视频 / 摄像头）
  - 训练
- 数据集与标注
- 导出与部署
- 评估
- 建议与最佳实践
- 贡献
- 许可
- 联系

---

## 项目简介
本项目提供用于自动售货机场景的 YOLOv5 检测器训练与部署脚本，目标包括识别饮料、零食、货道是否空缺、投币口等类目，支持离线批处理和实时摄像头检测。

## 功能
- 基于 YOLOv5 的训练与推理（支持 yolov5s/m/l）
- 提供 YOLO(TXT) 与 COCO(JSON) 格式示例配置
- 可配置的训练超参与增强策略
- 支持导出 ONNX / TensorRT（在支持的环境下）
- 支持单张图、批处理、视频与摄像头推理
- 针对边缘设备（Jetson、Coral 等）的部署建议

## 环境要求
- Python 3.8+
- PyTorch 1.10+（根据 CUDA 版本选择对应包）
- CUDA 10.2 / 11.x（GPU 可选）
- requirements.txt 中列出的其它依赖
- 可选：TensorRT、ONNX、OpenCV（含 contrib）

## 安装
```bash
git clone https://github.com/Ivy-forever18/yolov5-vending.git
cd yolov5-vending

# （可选）创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt

# 若无 requirements.txt，可手动安装核心依赖：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# pip install -r requirements.txt
```

## 快速开始

### 推理（单张图片）
```bash
python detect.py --weights yolov5s.pt --source data/images/test.jpg --conf 0.25 --save-conf --project runs/detect --name vending_demo
```

### 推理（视频 / 摄像头）
```bash
# 摄像头
python detect.py --weights yolov5s.pt --source 0 --img 640 --conf 0.25

# 视频文件
python detect.py --weights yolov5s.pt --source data/videos/test.mp4 --img 640 --conf 0.25
```

### 训练
准备好数据配置文件（参见数据集与标注），然后运行：
```bash
python train.py --img 640 --batch 16 --epochs 50 --data data/vending.yaml --weights yolov5s.pt --name vending_exp
```
常用参数：
- --img: 输入尺寸（如 640）
- --batch: 批大小（受显存限制）
- --epochs: 训练轮数
- --data: 数据配置 YAML 路径
- --weights: 初始权重（可使用预训练 yolov5s.pt 或 checkpoint）

## 数据集与标注

支持格式：
1. YOLO TXT（每张图片对应一个 .txt，格式："class x_center y_center width height"，坐标归一化到 [0,1]）
2. COCO JSON（标准 COCO 格式）

建议目录结构（YOLO 格式）：
```
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

示例 data/vending.yaml：
```yaml
train: ../dataset/images/train
val:   ../dataset/images/val

nc: 5
names: ['drink','snack','coin_slot','empty_slot','other']
```

标注建议：
- 类名保持一致并明确含义
- 确保每张图片的标签文件与图片一一对应
- 保留验证集用于可靠的 mAP 测量

## 导出与部署

导出 ONNX：
```bash
python export.py --weights runs/train/vending_exp/weights/best.pt --img 640 --include onnx
```

导出 TensorRT（在安装相关依赖且支持时）：
```bash
python export.py --weights runs/train/vending_exp/weights/best.pt --img 640 --include engine
```

部署建议：
- 边缘设备优先使用轻量模型（yolov5s）或量化/剪枝模型
- 在目标硬件上验证准确率与延迟
- 实时系统中合理设置批大小与输入尺寸

## 评估
使用内置评估脚本计算 mAP、精度和召回率：
```bash
python val.py --weights runs/train/vending_exp/weights/best.pt --data data/vending.yaml --img 640
```

## 建议与最佳实践
- 使用 Mosaic、MixUp 等增强以提升泛化能力
- 对类不平衡情况考虑过采样或使用 focal loss
- 改变批大小时按线性缩放规则调整学习率
- 生产环境可使用量化、蒸馏或剪枝以降低模型体积与延迟

## 贡献
- 提交 Issue 讨论改动或报告 Bug
- Fork → 新分支 → 提交 PR，PR 中请包含清晰说明和复现步骤
- 报告数据或标注问题时请提供示例或说明如何重现

## 许可
请添加 LICENSE 文件以声明许可（例如 MIT）。当前仓库未显式包含许可文件 — 若计划开源或分发，请补充许可。

## 联系
仓库地址: https://github.com/Ivy-forever18/yolov5-vending  
如需帮助，请在 Issue 中附上复现步骤与日志。

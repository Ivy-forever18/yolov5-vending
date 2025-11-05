> 本仓库基于 YOLOv5，针对自动售货机场景进行了数据集组织、训练与推理流程的封装，便于快速部署和二次开发。
>
> This repository is based on YOLOv5 and provides dataset organization, training and inference workflows tailored for vending machine use-cases, enabling quick deployment and further development.

---

目录 / Contents
- 项目简介 / Project Overview
- 功能亮点 / Features
- 环境与依赖 / Requirements
- 快速开始 / Quick Start
- 训练 / Training
- 推理 / Inference
- 数据集与标注格式 / Dataset & Annotation
- 导出与部署 / Export & Deployment
- 评估 / Evaluation
- 贡献 / Contributing
- 许可 / License
- 联系方式 / Contact

---

项目简介 / Project Overview
- 本项目目标是构建一个鲁棒的自动售货机物品检测模型（例如识别饮料、零食、货道状态等）。适用于视频/摄像头实时检测和离线图片批量处理。
- This project aims to build a robust object detection model for vending machines (e.g. detecting drinks, snacks, slot/stock status). It is suitable for real-time camera/video detection and batch image processing.

功能亮点 / Features
- 基于 YOLOv5 的训练与推理流程（支持 yolov5s/yolov5m/yolov5l 等）
- 支持 COCO/YOLO 格式数据集，含示例配置文件
- 训练超参数/数据增强可配置
- 支持导出 ONNX / TensorRT 等模型用于部署
- 支持单张图像与视频流推理
- Built on YOLOv5; supports yolov5s/m/l models
- Works with COCO/YOLO dataset formats and includes example config
- Configurable training params and augmentations
- Export to ONNX / TensorRT for deployment
- Single-image and video/stream inference supported

环境与依赖 / Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA 10.2 / 11.x (如需 GPU)
- 推荐使用虚拟环境（venv / conda）
- Python 3.8+
- PyTorch 1.10+
- CUDA 10.2 / 11.x (for GPU)
- Use virtual environment (venv / conda) recommended

示例安装 / Example installation
```bash
# 克隆仓库
git clone https://github.com/Ivy-forever18/yolov5-vending.git
cd yolov5-vending

# 创建并激活虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
# 若无 requirements.txt，可参考：
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
# pip install -r requirements.txt
```

快速开始 / Quick Start

推理示例（单张图片） / Inference (single image)
```bash
# 使用预训练权重进行推理
python detect.py --weights yolov5s.pt --source data/images/test.jpg --conf 0.25 --save-conf
```

推理示例（视频 / 摄像头） / Inference (video / webcam)
```bash
# 使用摄像头
python detect.py --weights yolov5s.pt --source 0 --img 640 --conf 0.25

# 处理视频文件
python detect.py --weights yolov5s.pt --source data/videos/test.mp4 --img 640 --conf 0.25
```

训练 / Training
- 准备好数据集并创建 data/vending.yaml（或类似文件）指定训练/验证路径与类别。
- 训练命令示例：
```bash
python train.py --img 640 --batch 16 --epochs 50 --data data/vending.yaml --weights yolov5s.pt --name vending_exp
```
- 常见参数：
  - --img: 输入大小
  - --batch: 批大小（受显存限制）
  - --epochs: 训练轮数
  - --data: 数据配置文件（包含类名与路径）
  - --weights: 初始权重（可用 yolov5s.pt 或上次训练的 best.pt）

数据集与标注格式 / Dataset & Annotation
- 支持两种主流格式：
  1. YOLO TXT（每张图片对应 .txt，格式：class x_center y_center width height，归一化到 [0,1]）
  2. COCO JSON（标准 COCO 格式）
- 推荐目录结构（YOLO 格式）：
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
- data/vending.yaml 示例：
```yaml
train: ../dataset/images/train
val:   ../dataset/images/val

nc: 5
names: ['drink','snack','coin_slot','empty_slot','other']
```

导出与部署 / Export & Deployment
- 导出 ONNX：
```bash
python export.py --weights runs/train/vending_exp/weights/best.pt --img 640 --include onnx
```
- 导出 TensorRT（如果仓库支持或你安装了相关依赖）：
```bash
python export.py --weights runs/train/vending_exp/weights/best.pt --img 640 --include engine
```
- 部署建议：在边缘设备优先使用轻量模型（yolov5s 或经过剪枝/量化的模型），在服务器端可使用更大模型以提高精度。

评估 / Evaluation
- 使用 val.py 或 detect/metrics 脚本评估检测性能（mAP, precision, recall）
- 示例：
```bash
python val.py --weights runs/train/vending_exp/weights/best.pt --data data/vending.yaml --img 640
```

调优建议 / Tips
- 数据增强：适量使用 Mosaic、HSV、随机翻转等可提升鲁棒性
- 类别不均衡：可考虑对小样本类别做上采样或使用 Focal Loss
- 学习率与批大小：遵循线性缩放规则调整学习率
- 如果需要更小更快的模型，考虑使用 pruning / quantization / knowledge distillation

贡献 / Contributing
- 欢迎提交 Issue 或 Pull Request
- 提交前请先创建 Issue 讨论需求或 Bug
- Fork → 新分支(feature/xxx) → 提交 PR

许可 / License
- 本仓库默认未声明特定许可证（请根据需要添加 LICENSE 文件）
- If you plan to publish this project, add a license file (e.g., MIT) to clarify reuse terms.

致谢 / Acknowledgements
- 感谢 YOLOv5 原作者及社区贡献者，许多代码与结构参考自 ultralytics/yolov5。
- Thanks to YOLOv5 authors and community; many scripts and structures are inspired by ultralytics/yolov5.

联系方式 / Contact
- 仓库：https://github.com/Ivy-forever18/yolov5-vending
- 如需帮助，请在 Issue 中描述问题并附上复现步骤与日志。
- Repo: https://github.com/Ivy-forever18/yolov5-vending
- For help, open an Issue with reproduction steps and logs.

---

快速检查清单 / Quick checklist
- [ ] data/vending.yaml 配置完成
- [ ] 数据按 YOLO/COCO 格式组织
- [ ] 安装依赖并验证 detect.py 能运行
- [ ] 进行少量训练调试超参数
- [ ] 导出并在目标设备验证模型效果

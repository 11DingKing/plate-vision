# 模型训练与微调指南

本目录包含车牌识别系统的模型训练和微调脚本。

## 目录结构

```
training/
├── README.md                    # 本文档
├── config.yaml                  # 训练配置文件
├── train_detector.py            # YOLOv8 车牌检测模型微调
├── train_char_classifier.py     # CNN 字符分类器训练
├── prepare_data.py              # 数据准备和增强脚本
└── evaluate.py                  # 模型评估脚本
```

## 环境要求

```bash
# 安装训练依赖
pip install ultralytics>=8.0.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install albumentations>=1.3.0
pip install tensorboard>=2.12.0
```

## 数据集准备

### 1. 车牌检测数据集

推荐数据集：
- CCPD (Chinese City Parking Dataset) - 约 25 万张中国车牌图像
- CLPD (Chinese License Plate Dataset)

数据目录结构：
```
data/
├── detection/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
```

### 2. 字符识别数据集

数据目录结构：
```
data/
├── characters/
│   ├── provinces/      # 省份简称 (京、津、沪...)
│   │   ├── 京/
│   │   ├── 津/
│   │   └── ...
│   ├── letters/        # 字母 (A-Z, 不含I和O)
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
│   └── digits/         # 数字 (0-9)
│       ├── 0/
│       ├── 1/
│       └── ...
```

## 快速开始

### 1. 准备数据

```bash
# 下载并准备 CCPD 数据集
python prepare_data.py --dataset ccpd --data_dir /path/to/ccpd --output_dir ./data

# 数据增强
python prepare_data.py --augment --input_dir ./data --output_dir ./data_augmented
```

### 2. 训练车牌检测模型

```bash
# 基础训练
python train_detector.py --data_dir ./data/detection --epochs 100

# 从预训练模型微调
python train_detector.py --data_dir ./data/detection --pretrained yolov8n.pt --epochs 50

# 针对特殊车牌微调 (黑牌、白牌)
python train_detector.py --data_dir ./data/detection_special --pretrained ./models/plate_detector.pt --epochs 30
```

### 3. 训练字符分类器

```bash
# 训练字符分类器
python train_char_classifier.py --data_dir ./data/characters --epochs 50

# 针对污损/模糊字符微调
python train_char_classifier.py --data_dir ./data/characters_hard --pretrained ./models/char_classifier.pt --epochs 20
```

### 4. 评估模型

```bash
# 评估检测模型
python evaluate.py --task detection --model ./models/plate_detector.pt --data_dir ./data/detection/val

# 评估字符分类器
python evaluate.py --task classification --model ./models/char_classifier.pt --data_dir ./data/characters/val
```

## 训练配置

编辑 `config.yaml` 修改训练参数：

```yaml
# 检测模型配置
detection:
  model: yolov8n.pt           # 基础模型
  epochs: 100                  # 训练轮数
  batch_size: 16              # 批次大小
  img_size: 640               # 输入图像尺寸
  lr0: 0.01                   # 初始学习率
  
# 字符分类器配置
classification:
  model: resnet18             # 骨干网络
  epochs: 50
  batch_size: 64
  img_size: [40, 20]          # 高x宽
  lr: 0.001
```

## 导出模型

```bash
# 导出 ONNX 格式 (用于部署)
python train_detector.py --export onnx --model ./models/plate_detector.pt

# 导出 TensorRT 格式 (GPU 加速)
python train_detector.py --export tensorrt --model ./models/plate_detector.pt
```

## 性能指标

训练完成后，模型应达到以下指标：

| 任务 | 指标 | 目标值 |
|------|------|--------|
| 车牌检测 | mAP@0.5 | ≥ 95% |
| 车牌检测 (复杂场景) | mAP@0.5 | ≥ 90% |
| 字符识别 | Top-1 准确率 | ≥ 99% |
| 字符识别 (污损/模糊) | Top-1 准确率 | ≥ 95% |

## 常见问题

### Q: 训练时显存不足？

降低 `batch_size` 或使用混合精度训练：
```bash
python train_detector.py --batch_size 8 --amp
```

### Q: 如何针对特定场景优化？

1. 收集目标场景的数据
2. 对现有模型进行微调（较少的 epochs）
3. 使用数据增强模拟目标场景条件

### Q: 模型推理速度慢？

1. 使用 TensorRT 加速
2. 使用更小的模型 (yolov8n vs yolov8s)
3. 降低输入图像分辨率

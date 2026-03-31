#!/usr/bin/env python3
"""
车牌检测模型训练脚本

基于 YOLOv8 进行车牌检测模型的训练和微调。
支持从头训练、预训练模型微调、特殊场景微调等功能。
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch


def load_config(config_path: str = "config.yaml") -> dict:
    """加载训练配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device_config: str = "auto") -> str:
    """获取训练设备"""
    if device_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_config


def create_data_yaml(data_dir: str, output_path: str) -> str:
    """创建 YOLO 格式的数据配置文件"""
    data_config = {
        "path": os.path.abspath(data_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "plate"  # 单类别：车牌
        },
        "nc": 1
    }
    
    # 如果需要多类别检测（按车牌颜色分类）
    # data_config = {
    #     "path": os.path.abspath(data_dir),
    #     "train": "images/train",
    #     "val": "images/val",
    #     "names": {
    #         0: "blue_plate",
    #         1: "yellow_plate", 
    #         2: "green_plate",
    #         3: "white_plate",
    #         4: "black_plate"
    #     },
    #     "nc": 5
    # }
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True)
    
    return output_path


def train_detector(
    data_dir: str,
    output_dir: str,
    config: dict,
    pretrained: str = None,
    resume: str = None,
    epochs: int = None,
    batch_size: int = None,
    img_size: int = None,
    device: str = None,
    amp: bool = True,
    name: str = None
):
    """
    训练车牌检测模型
    
    Args:
        data_dir: 数据目录路径
        output_dir: 输出目录路径
        config: 训练配置
        pretrained: 预训练模型路径
        resume: 恢复训练的检查点路径
        epochs: 训练轮数 (覆盖配置)
        batch_size: 批次大小 (覆盖配置)
        img_size: 图像尺寸 (覆盖配置)
        device: 训练设备
        amp: 是否使用混合精度
        name: 实验名称
    """
    from ultralytics import YOLO
    
    det_config = config.get("detection", {})
    hw_config = config.get("hardware", {})
    
    # 参数优先级: 命令行 > 配置文件 > 默认值
    epochs = epochs or det_config.get("epochs", 100)
    batch_size = batch_size or det_config.get("batch_size", 16)
    img_size = img_size or det_config.get("img_size", 640)
    device = device or get_device(hw_config.get("device", "auto"))
    amp = amp if amp is not None else hw_config.get("amp", True)
    
    # 创建数据配置文件
    data_yaml = create_data_yaml(data_dir, os.path.join(output_dir, "data.yaml"))
    
    # 加载模型
    if resume:
        print(f"从检查点恢复训练: {resume}")
        model = YOLO(resume)
    elif pretrained:
        print(f"从预训练模型微调: {pretrained}")
        model = YOLO(pretrained)
    else:
        base_model = det_config.get("base_model", "yolov8n.pt")
        print(f"使用基础模型: {base_model}")
        model = YOLO(base_model)
    
    # 实验名称
    if name is None:
        name = f"plate_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 数据增强配置
    aug_config = det_config.get("augmentation", {})
    
    # 训练参数
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": device,
        "amp": amp,
        "project": output_dir,
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": 42,
        
        # 优化器参数
        "optimizer": det_config.get("optimizer", "AdamW"),
        "lr0": det_config.get("lr0", 0.01),
        "lrf": det_config.get("lrf", 0.01),
        "momentum": det_config.get("momentum", 0.937),
        "weight_decay": det_config.get("weight_decay", 0.0005),
        "warmup_epochs": det_config.get("warmup_epochs", 3.0),
        "warmup_momentum": det_config.get("warmup_momentum", 0.8),
        
        # 数据增强
        "hsv_h": aug_config.get("hsv_h", 0.015),
        "hsv_s": aug_config.get("hsv_s", 0.7),
        "hsv_v": aug_config.get("hsv_v", 0.4),
        "degrees": aug_config.get("degrees", 15.0),
        "translate": aug_config.get("translate", 0.1),
        "scale": aug_config.get("scale", 0.5),
        "shear": aug_config.get("shear", 5.0),
        "perspective": aug_config.get("perspective", 0.0005),
        "flipud": aug_config.get("flipud", 0.0),
        "fliplr": aug_config.get("fliplr", 0.0),
        "mosaic": aug_config.get("mosaic", 0.5),
        "mixup": aug_config.get("mixup", 0.1),
        
        # 损失函数
        "box": det_config.get("loss", {}).get("box", 7.5),
        "cls": det_config.get("loss", {}).get("cls", 0.5),
        "dfl": det_config.get("loss", {}).get("dfl", 1.5),
        
        # 验证和保存
        "val": True,
        "save": True,
        "save_period": det_config.get("save_period", 10),
        "patience": det_config.get("patience", 50),
        
        # 多线程
        "workers": hw_config.get("workers", 4),
    }
    
    print("\n" + "="*60)
    print("开始训练车牌检测模型")
    print("="*60)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}/{name}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {img_size}")
    print(f"训练设备: {device}")
    print(f"混合精度: {amp}")
    print("="*60 + "\n")
    
    # 开始训练
    results = model.train(**train_args)
    
    # 验证最佳模型
    print("\n验证最佳模型...")
    best_model_path = os.path.join(output_dir, name, "weights", "best.pt")
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=data_yaml, device=device)
        
        print("\n" + "="*60)
        print("训练完成！最佳模型指标:")
        print("="*60)
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print(f"最佳模型路径: {best_model_path}")
        print("="*60)
    
    return results


def export_model(
    model_path: str,
    export_format: str = "onnx",
    output_dir: str = None,
    img_size: int = 640,
    simplify: bool = True,
    dynamic: bool = False
):
    """
    导出模型为部署格式
    
    Args:
        model_path: 模型路径
        export_format: 导出格式 (onnx, torchscript, tensorrt)
        output_dir: 输出目录
        img_size: 输入图像尺寸
        simplify: 是否简化 ONNX 模型
        dynamic: 是否使用动态输入尺寸
    """
    from ultralytics import YOLO
    
    print(f"\n导出模型: {model_path}")
    print(f"导出格式: {export_format}")
    
    model = YOLO(model_path)
    
    export_args = {
        "format": export_format,
        "imgsz": img_size,
        "simplify": simplify,
        "dynamic": dynamic,
    }
    
    if export_format == "onnx":
        export_args["opset"] = 12
    
    exported_path = model.export(**export_args)
    
    print(f"模型已导出: {exported_path}")
    return exported_path


def main():
    parser = argparse.ArgumentParser(description="车牌检测模型训练")
    
    # 基本参数
    parser.add_argument("--data_dir", type=str, default="./data/detection",
                        help="数据目录路径")
    parser.add_argument("--output_dir", type=str, default="./runs/detection",
                        help="输出目录路径")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径")
    
    # 模型参数
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练模型路径 (用于微调)")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批次大小")
    parser.add_argument("--img_size", type=int, default=None,
                        help="输入图像尺寸")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备 (cuda, cpu, mps)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="使用混合精度训练")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="禁用混合精度训练")
    parser.add_argument("--name", type=str, default=None,
                        help="实验名称")
    
    # 导出参数
    parser.add_argument("--export", type=str, default=None,
                        choices=["onnx", "torchscript", "tensorrt"],
                        help="导出模型格式")
    parser.add_argument("--model", type=str, default=None,
                        help="要导出的模型路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 导出模式
    if args.export:
        if not args.model:
            print("错误: 导出模式需要指定 --model 参数")
            sys.exit(1)
        export_model(
            model_path=args.model,
            export_format=args.export,
            output_dir=args.output_dir,
            img_size=args.img_size or 640
        )
    else:
        # 训练模式
        train_detector(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config,
            pretrained=args.pretrained,
            resume=args.resume,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            amp=args.amp,
            name=args.name
        )


if __name__ == "__main__":
    main()

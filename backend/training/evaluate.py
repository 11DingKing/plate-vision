#!/usr/bin/env python3
"""
模型评估脚本

评估车牌检测和字符识别模型的性能
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np
import torch
import cv2
from tqdm import tqdm


def evaluate_detector(
    model_path: str,
    data_dir: str,
    img_size: int = 640,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    device: str = None
) -> Dict:
    """
    评估车牌检测模型
    
    Args:
        model_path: 模型路径
        data_dir: 验证数据目录
        img_size: 图像尺寸
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值
        device: 设备
        
    Returns:
        评估指标字典
    """
    from ultralytics import YOLO
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n评估检测模型: {model_path}")
    print(f"数据目录: {data_dir}")
    print(f"设备: {device}")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 创建临时数据配置
    data_yaml = {
        "path": os.path.abspath(data_dir),
        "val": "images/val" if Path(data_dir, "images/val").exists() else "images",
        "names": {0: "plate"},
        "nc": 1
    }
    
    yaml_path = Path(data_dir) / "eval_data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    
    # 评估
    results = model.val(
        data=str(yaml_path),
        imgsz=img_size,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=True
    )
    
    # 清理临时文件
    yaml_path.unlink()
    
    metrics = {
        "mAP@0.5": float(results.box.map50),
        "mAP@0.5:0.95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1": 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-10)
    }
    
    print("\n" + "="*50)
    print("检测模型评估结果")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*50)
    
    return metrics


def evaluate_classifier(
    model_path: str,
    data_dir: str,
    batch_size: int = 64,
    device: str = None
) -> Dict:
    """
    评估字符分类器
    
    Args:
        model_path: 模型路径
        data_dir: 验证数据目录
        batch_size: 批次大小
        device: 设备
        
    Returns:
        评估指标字典
    """
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    # 导入训练脚本中的类
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_char_classifier import CharDataset, CharClassifier, ALL_CHARS
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    print(f"\n评估字符分类器: {model_path}")
    print(f"数据目录: {data_dir}")
    print(f"设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((40, 20)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    dataset = CharDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"样本数量: {len(dataset)}")
    
    # 加载模型
    model = CharClassifier(num_classes=len(ALL_CHARS), backbone="resnet18", pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 评估
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="评估中"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 按类别统计
            for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                label_char = ALL_CHARS[label] if label < len(ALL_CHARS) else "?"
                class_total[label_char] = class_total.get(label_char, 0) + 1
                if pred == label:
                    class_correct[label_char] = class_correct.get(label_char, 0) + 1
    
    # 计算整体指标
    accuracy = 100.0 * correct / total
    
    # 计算每类准确率
    class_acc = {}
    for char in class_total:
        class_acc[char] = 100.0 * class_correct.get(char, 0) / class_total[char]
    
    # 找出表现最差的类别
    worst_classes = sorted(class_acc.items(), key=lambda x: x[1])[:10]
    
    metrics = {
        "accuracy": accuracy,
        "total_samples": total,
        "correct_samples": correct,
        "num_classes": len(class_total),
        "worst_classes": worst_classes
    }
    
    print("\n" + "="*50)
    print("字符分类器评估结果")
    print("="*50)
    print(f"总体准确率: {accuracy:.2f}%")
    print(f"总样本数: {total}")
    print(f"正确样本数: {correct}")
    print(f"类别数: {len(class_total)}")
    print("\n表现最差的 10 个类别:")
    for char, acc in worst_classes:
        print(f"  {char}: {acc:.2f}% ({class_correct.get(char, 0)}/{class_total[char]})")
    print("="*50)
    
    return metrics


def evaluate_end_to_end(
    detector_path: str,
    classifier_path: str,
    test_images_dir: str,
    ground_truth_file: str = None,
    device: str = None
) -> Dict:
    """
    端到端评估完整的车牌识别系统
    
    Args:
        detector_path: 检测模型路径
        classifier_path: 分类模型路径 (可选)
        test_images_dir: 测试图像目录
        ground_truth_file: 真实标签文件 (可选)
        device: 设备
        
    Returns:
        评估指标
    """
    from ultralytics import YOLO
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n端到端评估")
    print(f"检测模型: {detector_path}")
    print(f"测试目录: {test_images_dir}")
    
    # 加载检测模型
    detector = YOLO(detector_path)
    
    # 收集测试图像
    test_dir = Path(test_images_dir)
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    print(f"测试图像数量: {len(image_files)}")
    
    # 统计
    total_images = len(image_files)
    detected_plates = 0
    processing_times = []
    
    for img_path in tqdm(image_files, desc="处理中"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 检测
        import time
        start_time = time.time()
        results = detector(img, verbose=False)
        elapsed = (time.time() - start_time) * 1000
        processing_times.append(elapsed)
        
        # 统计检测到的车牌
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                detected_plates += len(result.boxes)
    
    # 计算指标
    avg_time = np.mean(processing_times) if processing_times else 0
    fps = 1000 / avg_time if avg_time > 0 else 0
    
    metrics = {
        "total_images": total_images,
        "detected_plates": detected_plates,
        "avg_plates_per_image": detected_plates / total_images if total_images > 0 else 0,
        "avg_processing_time_ms": avg_time,
        "fps": fps
    }
    
    print("\n" + "="*50)
    print("端到端评估结果")
    print("="*50)
    print(f"测试图像数: {total_images}")
    print(f"检测到的车牌数: {detected_plates}")
    print(f"平均每张图像车牌数: {metrics['avg_plates_per_image']:.2f}")
    print(f"平均处理时间: {avg_time:.2f} ms")
    print(f"处理速度: {fps:.1f} FPS")
    print("="*50)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    
    parser.add_argument("--task", type=str, required=True,
                        choices=["detection", "classification", "e2e"],
                        help="评估任务类型")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据目录")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批次大小 (分类任务)")
    parser.add_argument("--img_size", type=int, default=640,
                        help="图像尺寸 (检测任务)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU 阈值")
    
    args = parser.parse_args()
    
    if args.task == "detection":
        evaluate_detector(
            model_path=args.model,
            data_dir=args.data_dir,
            img_size=args.img_size,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
    elif args.task == "classification":
        evaluate_classifier(
            model_path=args.model,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=args.device
        )
    elif args.task == "e2e":
        evaluate_end_to_end(
            detector_path=args.model,
            classifier_path=None,
            test_images_dir=args.data_dir,
            device=args.device
        )


if __name__ == "__main__":
    main()

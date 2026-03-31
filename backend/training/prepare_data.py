#!/usr/bin/env python3
"""
数据准备脚本

功能：
1. CCPD 数据集转换为 YOLO 格式
2. 字符数据集提取和整理
3. 数据增强
4. 数据集划分
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm


# 省份简称映射
PROVINCE_MAP = {
    "皖": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4,
    "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9,
    "苏": 10, "浙": 11, "京": 12, "闽": 13, "赣": 14,
    "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19,
    "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
    "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29,
    "新": 30
}

PROVINCE_LIST = list(PROVINCE_MAP.keys())


def parse_ccpd_filename(filename: str) -> Dict:
    """
    解析 CCPD 数据集文件名
    
    CCPD 文件名格式:
    [区域]-[倾斜角度]-[边界框坐标]-[四个顶点坐标]-[车牌号]-[亮度]-[模糊度].jpg
    
    例如: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    """
    try:
        parts = filename.replace(".jpg", "").split("-")
        if len(parts) < 7:
            return None
        
        # 解析边界框 (原始格式: x1&y1_x2&y2)
        bbox_str = parts[2]
        bbox_parts = bbox_str.split("_")
        x1, y1 = map(int, bbox_parts[0].split("&"))
        x2, y2 = map(int, bbox_parts[1].split("&"))
        
        # 解析四个顶点
        vertices_str = parts[3]
        vertices = []
        for v in vertices_str.split("_"):
            vx, vy = map(int, v.split("&"))
            vertices.append((vx, vy))
        
        # 解析车牌号
        plate_str = parts[4]
        plate_indices = list(map(int, plate_str.split("_")))
        
        # 转换为车牌字符
        chars = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
        chars += "ABCDEFGHJKLMNPQRSTUVWXYZ"
        chars += "0123456789"
        
        plate_number = ""
        for idx in plate_indices:
            if idx < len(chars):
                plate_number += chars[idx]
        
        return {
            "bbox": (x1, y1, x2, y2),
            "vertices": vertices,
            "plate_number": plate_number,
            "area_idx": int(parts[0]),
            "tilt": int(parts[1].split("_")[0]) if "_" in parts[1] else int(parts[1]),
            "brightness": int(parts[5]) if len(parts) > 5 else 0,
            "blur": int(parts[6]) if len(parts) > 6 else 0
        }
    except Exception as e:
        return None


def convert_ccpd_to_yolo(
    ccpd_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
):
    """
    将 CCPD 数据集转换为 YOLO 格式
    
    Args:
        ccpd_dir: CCPD 数据集目录
        output_dir: 输出目录
        split_ratio: (train, val, test) 比例
    """
    ccpd_path = Path(ccpd_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # 收集所有图像文件
    image_files = []
    for subdir in ["ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db", 
                   "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"]:
        subdir_path = ccpd_path / subdir
        if subdir_path.exists():
            image_files.extend(list(subdir_path.glob("*.jpg")))
    
    if not image_files:
        # 如果没有子目录，直接搜索根目录
        image_files = list(ccpd_path.glob("**/*.jpg"))
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 划分数据集
    n_train = int(len(image_files) * split_ratio[0])
    n_val = int(len(image_files) * split_ratio[1])
    
    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:]
    }
    
    # 转换
    for split_name, files in splits.items():
        print(f"\n处理 {split_name} 集 ({len(files)} 张)...")
        
        for img_path in tqdm(files):
            # 解析文件名
            info = parse_ccpd_filename(img_path.name)
            if info is None:
                continue
            
            # 读取图像获取尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # 转换边界框为 YOLO 格式 (归一化的中心点坐标和宽高)
            x1, y1, x2, y2 = info["bbox"]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # 确保坐标在 [0, 1] 范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # 新文件名
            new_name = f"{img_path.stem}"
            
            # 复制图像
            dst_img_path = output_path / "images" / split_name / f"{new_name}.jpg"
            shutil.copy(img_path, dst_img_path)
            
            # 写入标签文件 (类别 0 = plate)
            dst_label_path = output_path / "labels" / split_name / f"{new_name}.txt"
            with open(dst_label_path, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"\n转换完成！输出目录: {output_dir}")
    print(f"  训练集: {len(splits['train'])} 张")
    print(f"  验证集: {len(splits['val'])} 张")
    print(f"  测试集: {len(splits['test'])} 张")


def extract_characters_from_ccpd(
    ccpd_dir: str,
    output_dir: str,
    max_samples_per_char: int = 5000
):
    """
    从 CCPD 数据集中提取字符图像
    
    Args:
        ccpd_dir: CCPD 数据集目录
        output_dir: 输出目录
        max_samples_per_char: 每个字符的最大样本数
    """
    ccpd_path = Path(ccpd_dir)
    output_path = Path(output_dir)
    
    # 收集所有图像
    image_files = list(ccpd_path.glob("**/*.jpg"))
    print(f"找到 {len(image_files)} 张图像")
    
    # 字符计数
    char_counts = {}
    
    for img_path in tqdm(image_files):
        # 解析文件名
        info = parse_ccpd_filename(img_path.name)
        if info is None:
            continue
        
        plate_number = info["plate_number"]
        if len(plate_number) < 7:
            continue
        
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 裁剪车牌区域
        x1, y1, x2, y2 = info["bbox"]
        plate_img = img[y1:y2, x1:x2]
        
        if plate_img.size == 0:
            continue
        
        # 简单的字符分割 (均匀分割)
        h, w = plate_img.shape[:2]
        num_chars = len(plate_number)
        char_width = w // num_chars
        
        for i, char in enumerate(plate_number):
            # 检查是否需要更多样本
            if char_counts.get(char, 0) >= max_samples_per_char:
                continue
            
            # 提取字符
            x_start = i * char_width
            x_end = (i + 1) * char_width if i < num_chars - 1 else w
            char_img = plate_img[:, x_start:x_end]
            
            if char_img.size == 0:
                continue
            
            # 创建字符目录
            char_dir = output_path / char
            char_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存字符图像
            count = char_counts.get(char, 0)
            char_path = char_dir / f"{char}_{count:05d}.jpg"
            cv2.imwrite(str(char_path), char_img)
            
            char_counts[char] = count + 1
    
    # 打印统计
    print("\n字符提取统计:")
    for char, count in sorted(char_counts.items()):
        print(f"  {char}: {count}")


def augment_images(
    input_dir: str,
    output_dir: str,
    augment_factor: int = 3
):
    """
    对图像进行数据增强
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        augment_factor: 增强倍数
    """
    try:
        import albumentations as A
    except ImportError:
        print("请安装 albumentations: pip install albumentations")
        return
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 定义增强变换
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Rotate(limit=10, p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=0.5),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
    ])
    
    # 处理每个子目录
    for subdir in input_path.iterdir():
        if not subdir.is_dir():
            continue
        
        dst_subdir = output_path / subdir.name
        dst_subdir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"增强 {subdir.name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # 复制原始图像
            shutil.copy(img_path, dst_subdir / img_path.name)
            
            # 生成增强图像
            for i in range(augment_factor - 1):
                augmented = transform(image=img)
                aug_img = augmented["image"]
                
                aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(dst_subdir / aug_name), aug_img)
    
    print(f"\n数据增强完成！输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="数据准备脚本")
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # CCPD 转换命令
    ccpd_parser = subparsers.add_parser("ccpd", help="转换 CCPD 数据集")
    ccpd_parser.add_argument("--data_dir", type=str, required=True,
                            help="CCPD 数据集目录")
    ccpd_parser.add_argument("--output_dir", type=str, required=True,
                            help="输出目录")
    ccpd_parser.add_argument("--split", type=str, default="0.8,0.1,0.1",
                            help="数据集划分比例 (train,val,test)")
    
    # 字符提取命令
    char_parser = subparsers.add_parser("extract_chars", help="提取字符图像")
    char_parser.add_argument("--data_dir", type=str, required=True,
                            help="CCPD 数据集目录")
    char_parser.add_argument("--output_dir", type=str, required=True,
                            help="输出目录")
    char_parser.add_argument("--max_per_char", type=int, default=5000,
                            help="每个字符的最大样本数")
    
    # 数据增强命令
    aug_parser = subparsers.add_parser("augment", help="数据增强")
    aug_parser.add_argument("--input_dir", type=str, required=True,
                           help="输入目录")
    aug_parser.add_argument("--output_dir", type=str, required=True,
                           help="输出目录")
    aug_parser.add_argument("--factor", type=int, default=3,
                           help="增强倍数")
    
    args = parser.parse_args()
    
    if args.command == "ccpd":
        split_ratio = tuple(map(float, args.split.split(",")))
        convert_ccpd_to_yolo(args.data_dir, args.output_dir, split_ratio)
    elif args.command == "extract_chars":
        extract_characters_from_ccpd(args.data_dir, args.output_dir, args.max_per_char)
    elif args.command == "augment":
        augment_images(args.input_dir, args.output_dir, args.factor)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

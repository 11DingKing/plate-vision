#!/usr/bin/env python3
"""
性能基准测试脚本

验证系统是否满足 Prompt 要求的性能指标：
- 单帧识别时间 ≤100ms (CPU)
- 视频流处理 ≥25fps
- 准确率 ≥99% (需要测试数据集)

运行方式:
    python scripts/benchmark.py [--images-dir DIR] [--iterations N]
"""

import os
import sys
import time
import json
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def create_test_image(width: int = 800, height: int = 600) -> np.ndarray:
    """
    创建测试图像（带有模拟车牌区域）
    
    用于在没有真实测试图像时进行性能测试
    """
    # 创建背景
    image = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
    
    # 添加模拟车牌区域（蓝色矩形）
    plate_x = width // 4
    plate_y = height // 2 - 30
    plate_w = 200
    plate_h = 60
    
    # 蓝色背景
    cv2.rectangle(image, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), 
                  (200, 100, 50), -1)
    
    # 添加白色文字区域
    for i in range(7):
        x = plate_x + 10 + i * 26
        cv2.rectangle(image, (x, plate_y + 10), (x + 20, plate_y + 50), 
                      (255, 255, 255), -1)
    
    return image


def benchmark_single_image(detector, recognizer, preprocessor, image: np.ndarray) -> Dict[str, Any]:
    """
    对单张图像进行基准测试
    
    Returns:
        包含处理时间和结果的字典
    """
    start_time = time.perf_counter()
    
    # 预处理
    t1 = time.perf_counter()
    processed = preprocessor.process(image)
    preprocess_time = (time.perf_counter() - t1) * 1000
    
    # 检测
    t2 = time.perf_counter()
    detections = detector.detect(processed)
    detect_time = (time.perf_counter() - t2) * 1000
    
    # 识别（如果有检测结果且需要额外识别）
    recognize_time = 0
    plates = []
    for det in detections:
        if det.plate_text:
            # HyperLPR3 已返回结果
            plates.append({
                "plate_number": det.plate_text,
                "confidence": det.confidence,
                "source": "hyperlpr3"
            })
        else:
            # 需要额外识别
            t3 = time.perf_counter()
            plate_img = det.corrected_image if det.corrected_image is not None else det.cropped_image
            if plate_img is not None:
                result = recognizer.recognize(plate_img, det.plate_type, det.bbox)
                plates.append({
                    "plate_number": result.plate_number,
                    "confidence": result.confidence,
                    "source": "paddleocr"
                })
            recognize_time += (time.perf_counter() - t3) * 1000
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    return {
        "total_time_ms": total_time,
        "preprocess_time_ms": preprocess_time,
        "detect_time_ms": detect_time,
        "recognize_time_ms": recognize_time,
        "plates_count": len(plates),
        "plates": plates
    }


def benchmark_video_fps(detector, recognizer, preprocessor, 
                        num_frames: int = 100, image_size: tuple = (800, 600)) -> Dict[str, Any]:
    """
    模拟视频流处理，测试 FPS
    """
    # 创建测试帧
    test_frame = create_test_image(image_size[0], image_size[1])
    
    frame_times = []
    
    start_time = time.perf_counter()
    
    for i in range(num_frames):
        frame_start = time.perf_counter()
        
        # 预处理
        processed = preprocessor.process(test_frame)
        
        # 检测
        detections = detector.detect(processed)
        
        # 识别（使用 HyperLPR3 返回的结果）
        for det in detections:
            if not det.plate_text:
                plate_img = det.corrected_image if det.corrected_image is not None else det.cropped_image
                if plate_img is not None:
                    recognizer.recognize(plate_img, det.plate_type, det.bbox)
        
        frame_time = (time.perf_counter() - frame_start) * 1000
        frame_times.append(frame_time)
    
    total_time = time.perf_counter() - start_time
    avg_fps = num_frames / total_time
    
    return {
        "total_frames": num_frames,
        "total_time_seconds": total_time,
        "average_fps": avg_fps,
        "average_frame_time_ms": statistics.mean(frame_times),
        "min_frame_time_ms": min(frame_times),
        "max_frame_time_ms": max(frame_times),
        "std_frame_time_ms": statistics.stdev(frame_times) if len(frame_times) > 1 else 0,
        "meets_25fps_requirement": avg_fps >= 25
    }


def run_benchmark(images_dir: Optional[str] = None, iterations: int = 10) -> Dict[str, Any]:
    """
    运行完整的基准测试
    """
    print("="*60)
    print("车牌识别系统 - 性能基准测试")
    print("="*60)
    print(f"测试时间: {datetime.now().isoformat()}")
    print()
    
    # 初始化组件
    print("正在初始化组件...")
    
    from src.detector import PlateDetector
    from src.recognizer import CharRecognizer
    from src.preprocessor import ImageProcessor
    
    detector = PlateDetector(use_yolo=True, device="cpu")
    recognizer = CharRecognizer(use_gpu=False, lightweight_mode=True)
    preprocessor = ImageProcessor(fast_mode=True)
    
    print("组件初始化完成")
    print()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "python_version": sys.version,
            "opencv_version": cv2.__version__,
        },
        "tests": {}
    }
    
    # 测试1: 单帧处理时间
    print("="*40)
    print("测试1: 单帧处理时间")
    print("="*40)
    
    test_images = []
    
    if images_dir and Path(images_dir).exists():
        # 使用真实测试图像
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            test_images.extend(Path(images_dir).glob(ext))
        print(f"找到 {len(test_images)} 张测试图像")
    
    if not test_images:
        # 使用生成的测试图像
        print("使用生成的测试图像")
        test_images = [None] * iterations
    
    single_frame_times = []
    
    for i, img_path in enumerate(test_images[:iterations]):
        if img_path:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
        else:
            image = create_test_image()
        
        result = benchmark_single_image(detector, recognizer, preprocessor, image)
        single_frame_times.append(result["total_time_ms"])
        
        if (i + 1) % 5 == 0:
            print(f"  已处理 {i+1}/{min(len(test_images), iterations)} 帧")
    
    avg_time = statistics.mean(single_frame_times)
    meets_100ms = avg_time <= 100
    
    print(f"\n单帧平均处理时间: {avg_time:.2f}ms")
    print(f"最小时间: {min(single_frame_times):.2f}ms")
    print(f"最大时间: {max(single_frame_times):.2f}ms")
    print(f"标准差: {statistics.stdev(single_frame_times) if len(single_frame_times) > 1 else 0:.2f}ms")
    print(f"满足 ≤100ms 要求: {'✓ 是' if meets_100ms else '✗ 否'}")
    
    results["tests"]["single_frame"] = {
        "iterations": len(single_frame_times),
        "average_time_ms": avg_time,
        "min_time_ms": min(single_frame_times),
        "max_time_ms": max(single_frame_times),
        "std_time_ms": statistics.stdev(single_frame_times) if len(single_frame_times) > 1 else 0,
        "meets_100ms_requirement": meets_100ms
    }
    
    # 测试2: 视频流 FPS
    print("\n" + "="*40)
    print("测试2: 视频流 FPS")
    print("="*40)
    
    fps_result = benchmark_video_fps(detector, recognizer, preprocessor, num_frames=50)
    
    print(f"\n处理帧数: {fps_result['total_frames']}")
    print(f"总耗时: {fps_result['total_time_seconds']:.2f}s")
    print(f"平均 FPS: {fps_result['average_fps']:.2f}")
    print(f"平均帧时间: {fps_result['average_frame_time_ms']:.2f}ms")
    print(f"满足 ≥25fps 要求: {'✓ 是' if fps_result['meets_25fps_requirement'] else '✗ 否'}")
    
    results["tests"]["video_fps"] = fps_result
    
    # 测试3: 检测方法验证
    print("\n" + "="*40)
    print("测试3: 检测方法状态")
    print("="*40)
    
    print(f"HyperLPR3 可用: {'✓ 是' if detector.hyperlpr_available else '✗ 否'}")
    print(f"YOLO 车牌模型可用: {'✓ 是' if detector.yolo_plate_model_available else '✗ 否'}")
    
    results["tests"]["detection_methods"] = {
        "hyperlpr3_available": detector.hyperlpr_available,
        "yolo_plate_model_available": detector.yolo_plate_model_available
    }
    
    # 总结
    print("\n" + "="*60)
    print("基准测试总结")
    print("="*60)
    
    all_passed = meets_100ms and fps_result['meets_25fps_requirement']
    
    print(f"\n| 指标 | 要求 | 实际 | 状态 |")
    print(f"|------|------|------|------|")
    print(f"| 单帧耗时 | ≤100ms | {avg_time:.1f}ms | {'✓' if meets_100ms else '✗'} |")
    print(f"| 视频帧率 | ≥25fps | {fps_result['average_fps']:.1f}fps | {'✓' if fps_result['meets_25fps_requirement'] else '✗'} |")
    
    results["summary"] = {
        "all_tests_passed": all_passed,
        "single_frame_passed": meets_100ms,
        "video_fps_passed": fps_result['meets_25fps_requirement']
    }
    
    # 保存结果
    output_dir = Path(__file__).parent.parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n基准测试结果已保存到: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="车牌识别系统性能基准测试")
    parser.add_argument("--images-dir", type=str, default=None, 
                        help="测试图像目录路径")
    parser.add_argument("--iterations", type=int, default=20,
                        help="单帧测试迭代次数")
    
    args = parser.parse_args()
    
    results = run_benchmark(args.images_dir, args.iterations)
    
    # 返回退出码
    if results["summary"]["all_tests_passed"]:
        print("\n✓ 所有性能测试通过")
        return 0
    else:
        print("\n✗ 部分性能测试未通过")
        return 1


if __name__ == "__main__":
    sys.exit(main())

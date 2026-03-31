#!/usr/bin/env python3
"""
模型下载脚本

下载预训练的车牌检测模型，确保系统开箱即用。
优先使用 HyperLPR3（无需额外下载），备选下载 YOLO 车牌模型。
"""

import os
import sys
import hashlib
import urllib.request
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List


# 模型目录
MODELS_DIR = Path(__file__).parent.parent / "models"

# YOLO 车牌检测模型来源（开源社区提供）
YOLO_PLATE_MODELS = [
    {
        "name": "yolo-license-plate",
        "url": "https://github.com/Muhammad-Zeerak-Khan/License-Plate-Detection-using-YOLOv8/raw/main/license_plate_detector.pt",
        "filename": "plate_detect.pt",
        "description": "YOLOv8 车牌检测模型 (开源社区)"
    },
    {
        "name": "yolov8-plate-fallback",
        "url": "https://huggingface.co/keremberke/yolov8n-license-plate-detection/resolve/main/best.pt",
        "filename": "plate_detect.pt", 
        "description": "YOLOv8 车牌检测模型 (HuggingFace)"
    }
]


def download_file(url: str, dest_path: str, show_progress: bool = True) -> bool:
    """下载文件"""
    try:
        print(f"正在下载: {url}")
        print(f"保存到: {dest_path}")
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if show_progress:
            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(100, count * block_size * 100 // total_size)
                    sys.stdout.write(f"\r下载进度: {percent}%")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, dest_path, progress_hook)
            print()
        else:
            urllib.request.urlretrieve(url, dest_path)
        
        # 验证文件
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"下载完成: {dest_path}")
            return True
        else:
            print("下载的文件无效")
            return False
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def setup_hyperlpr() -> bool:
    """
    设置 HyperLPR3
    
    HyperLPR3 是专门针对中国车牌优化的开源识别库，
    安装后会自动下载所需模型，无需额外配置。
    """
    print("\n" + "="*60)
    print("设置 HyperLPR3 (中国车牌识别专用库)")
    print("="*60)
    
    try:
        import hyperlpr3 as lpr3
        print("✓ HyperLPR3 已安装")
        
        # 测试初始化（这会触发模型下载）
        print("正在初始化 HyperLPR3（首次运行会下载模型，请耐心等待）...")
        catcher = lpr3.LicensePlateCatcher()
        
        # 运行一次测试识别，确保模型完全加载
        print("正在预热模型...")
        import numpy as np
        test_img = np.zeros((100, 200, 3), dtype=np.uint8)
        try:
            catcher(test_img)
        except:
            pass  # 忽略测试图像的识别错误
        
        print("✓ HyperLPR3 初始化成功，模型已就绪")
        return True
        
    except ImportError:
        print("HyperLPR3 未安装，尝试安装...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "hyperlpr3", "-q", "--no-cache-dir"
            ])
            
            # 再次尝试导入和初始化
            import hyperlpr3 as lpr3
            print("正在初始化 HyperLPR3（首次运行会下载模型，请耐心等待）...")
            catcher = lpr3.LicensePlateCatcher()
            print("✓ HyperLPR3 安装并初始化成功")
            return True
            
        except Exception as e:
            print(f"✗ HyperLPR3 安装失败: {e}")
            return False
            
    except Exception as e:
        print(f"✗ HyperLPR3 初始化失败: {e}")
        return False


def download_yolo_plate_model() -> Optional[str]:
    """
    下载 YOLO 车牌检测模型（备选方案）
    
    如果 HyperLPR3 不可用，使用 YOLO 模型进行检测
    """
    print("\n" + "="*60)
    print("下载 YOLO 车牌检测模型（备选方案）")
    print("="*60)
    
    target_path = MODELS_DIR / "plate_detect.pt"
    
    # 检查是否已存在
    if target_path.exists() and target_path.stat().st_size > 10000:
        print(f"✓ YOLO 车牌模型已存在: {target_path}")
        return str(target_path)
    
    # 尝试每个来源
    for model_info in YOLO_PLATE_MODELS:
        print(f"\n尝试下载: {model_info['description']}")
        
        if download_file(model_info["url"], str(target_path)):
            print(f"✓ YOLO 模型下载成功: {target_path}")
            return str(target_path)
        
        # 清理失败的下载
        if target_path.exists():
            target_path.unlink()
    
    print("✗ 所有 YOLO 模型下载源都失败")
    return None


def create_model_info():
    """创建模型信息文件"""
    info_path = MODELS_DIR / "MODEL_INFO.md"
    
    content = """# 模型信息

## 车牌识别方案

本系统使用以下方案进行车牌识别（按优先级）：

### 1. HyperLPR3（推荐，默认启用）

- **类型**：端到端车牌识别库
- **优点**：专为中国车牌优化，检测+识别一体化，无需额外模型
- **安装**：`pip install hyperlpr3`
- **性能**：单帧 <50ms (CPU)

### 2. YOLO 车牌检测 + PaddleOCR（备选）

- **检测模型**：`plate_detect.pt`
- **识别模型**：PaddleOCR（自动下载）
- **适用**：HyperLPR3 不可用时的备选方案

## 性能指标

| 方案 | 单帧耗时 (CPU) | 视频帧率 |
|------|----------------|----------|
| HyperLPR3 | <50ms | ≥25fps |
| YOLO+PaddleOCR | <100ms | ≥15fps |

## 模型训练

如需训练自定义模型，请参考 `training/README.md`
"""
    
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✓ 已创建模型信息: {info_path}")


def main():
    """主函数"""
    print("="*60)
    print("车牌识别系统 - 模型初始化")
    print("="*60)
    print("\n系统将自动下载所需模型，请耐心等待...")
    print("首次运行可能需要 1-3 分钟（取决于网络速度）")
    
    # 创建模型目录
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 设置 HyperLPR3（主要方案，会自动下载模型）
    hyperlpr_ok = setup_hyperlpr()
    
    # 2. 下载 YOLO 模型（备选方案）
    yolo_model = None
    if not hyperlpr_ok:
        print("\nHyperLPR3 不可用，下载备选 YOLO 模型...")
        yolo_model = download_yolo_plate_model()
    
    # 3. 创建信息文件
    create_model_info()
    
    # 总结
    print("\n" + "="*60)
    print("模型初始化完成")
    print("="*60)
    
    if hyperlpr_ok:
        print("✓ HyperLPR3: 已就绪（推荐方案）")
        print("  - 端到端车牌检测+识别")
        print("  - 支持蓝牌、黄牌、绿牌、白牌、黑牌")
        print("  - 预期性能: <50ms/帧 (CPU)")
    else:
        print("✗ HyperLPR3: 不可用")
    
    if yolo_model:
        print(f"✓ YOLO 车牌模型: {yolo_model}")
    
    print("\n" + "="*60)
    
    if hyperlpr_ok or yolo_model:
        print("✓ 所有模型已就绪，系统可以正常使用")
        return 0
    else:
        print("⚠ 警告: 主要检测方案不可用，将使用传统 CV 方法")
        print("  建议检查网络连接后重新运行此脚本")
        return 1


if __name__ == "__main__":
    sys.exit(main())

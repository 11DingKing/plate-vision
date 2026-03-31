#!/usr/bin/env python3
"""
字符分类器训练脚本

基于 CNN 训练车牌字符分类器，支持:
- 省份简称 (31类)
- 英文字母 (24类，不含I和O)
- 数字 (10类)
- 特殊字符 (3类: 港、澳、学等)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import cv2
from tqdm import tqdm


# 字符集定义
PROVINCES = [
    "京", "津", "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘",
    "皖", "鲁", "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋",
    "蒙", "陕", "吉", "闽", "贵", "粤", "青", "藏", "川", "宁", "琼"
]

LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z"
]

DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

SPECIAL_CHARS = ["港", "澳", "学"]

# 完整字符集
ALL_CHARS = PROVINCES + LETTERS + DIGITS + SPECIAL_CHARS
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALL_CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(ALL_CHARS)}


class CharDataset(Dataset):
    """字符数据集"""
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        char_list: List[str] = None
    ):
        """
        Args:
            data_dir: 数据目录，结构为 data_dir/char_name/images
            transform: 数据变换
            char_list: 要使用的字符列表
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.char_list = char_list or ALL_CHARS
        
        self.samples = []
        self.char_to_idx = {char: idx for idx, char in enumerate(self.char_list)}
        
        self._load_samples()
    
    def _load_samples(self):
        """加载所有样本"""
        for char in self.char_list:
            char_dir = self.data_dir / char
            if not char_dir.exists():
                continue
            
            for img_path in char_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((str(img_path), self.char_to_idx[char]))
        
        print(f"加载了 {len(self.samples)} 个样本，{len(self.char_list)} 个类别")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            # 返回一个空白图像
            image = np.zeros((40, 20, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CharClassifier(nn.Module):
    """字符分类器模型"""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True
    ):
        super().__init__()
        
        self.backbone_name = backbone
        
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            # 修改第一层以适应小图像
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()  # 移除 maxpool
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif backbone == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            # 修改第一层
            self.backbone.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            
        else:
            # 自定义轻量级 CNN
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)


def get_transforms(config: dict, is_train: bool = True) -> transforms.Compose:
    """获取数据变换"""
    img_height = config.get("img_height", 40)
    img_width = config.get("img_width", 20)
    aug_config = config.get("augmentation", {})
    
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_height, img_width)),
            transforms.RandomRotation(aug_config.get("random_rotation", 5)),
            transforms.RandomAffine(
                degrees=0,
                scale=tuple(aug_config.get("random_scale", [0.9, 1.1]))
            ),
            transforms.ColorJitter(
                brightness=aug_config.get("brightness", 0.2),
                contrast=aug_config.get("contrast", 0.2)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{running_loss / (pbar.n + 1):.4f}",
            "acc": f"{100. * correct / total:.2f}%"
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train_classifier(
    data_dir: str,
    output_dir: str,
    config: dict,
    pretrained_path: str = None,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    device: str = None,
    name: str = None
):
    """
    训练字符分类器
    """
    cls_config = config.get("classification", {})
    hw_config = config.get("hardware", {})
    
    # 参数
    epochs = epochs or cls_config.get("epochs", 50)
    batch_size = batch_size or cls_config.get("batch_size", 64)
    lr = lr or cls_config.get("lr", 0.001)
    backbone = cls_config.get("backbone", "resnet18")
    num_classes = cls_config.get("num_classes", len(ALL_CHARS))
    val_split = cls_config.get("val_split", 0.1)
    
    # 设备
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    
    # 实验名称
    if name is None:
        name = f"char_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    save_dir = Path(output_dir) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("开始训练字符分类器")
    print("="*60)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {save_dir}")
    print(f"骨干网络: {backbone}")
    print(f"类别数量: {num_classes}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"训练设备: {device}")
    print("="*60 + "\n")
    
    # 数据集
    train_transform = get_transforms(cls_config, is_train=True)
    val_transform = get_transforms(cls_config, is_train=False)
    
    full_dataset = CharDataset(data_dir, transform=train_transform)
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 为验证集使用不带增强的变换
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=hw_config.get("workers", 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=hw_config.get("workers", 4),
        pin_memory=True
    )
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    
    # 模型
    model = CharClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=(pretrained_path is None)
    )
    
    if pretrained_path:
        print(f"加载预训练权重: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=cls_config.get("weight_decay", 0.0001)
    )
    
    # 学习率调度器
    scheduler_type = cls_config.get("scheduler", "CosineAnnealingLR")
    if scheduler_type == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cls_config.get("T_max", epochs),
            eta_min=cls_config.get("eta_min", 1e-6)
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练循环
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best.pt")
            print(f"  -> 保存最佳模型 (acc: {best_acc:.2f}%)")
        
        # 定期保存检查点
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
            }, save_dir / f"checkpoint_epoch{epoch}.pt")
    
    # 保存最终模型
    torch.save(model.state_dict(), save_dir / "last.pt")
    
    # 保存训练历史
    with open(save_dir / "history.yaml", "w") as f:
        yaml.dump(history, f)
    
    # 保存字符映射
    with open(save_dir / "char_mapping.yaml", "w", encoding="utf-8") as f:
        yaml.dump({
            "char_to_idx": CHAR_TO_IDX,
            "idx_to_char": IDX_TO_CHAR
        }, f, allow_unicode=True)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存路径: {save_dir}")
    print("="*60)
    
    return history


def main():
    parser = argparse.ArgumentParser(description="字符分类器训练")
    
    parser.add_argument("--data_dir", type=str, default="./data/characters",
                        help="数据目录路径")
    parser.add_argument("--output_dir", type=str, default="./runs/classification",
                        help="输出目录路径")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练模型路径")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备")
    parser.add_argument("--name", type=str, default=None,
                        help="实验名称")
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    
    # 训练
    train_classifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        pretrained_path=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        name=args.name
    )


if __name__ == "__main__":
    main()

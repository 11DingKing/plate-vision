#!/bin/bash
# Docker 容器启动脚本
# 确保所有模型在服务启动前下载完成

set -e

echo "============================================"
echo "车牌识别系统 - 启动中"
echo "============================================"

# 创建必要目录
mkdir -p /app/logs /app/output /app/models

# 初始化/下载模型
echo ""
echo "正在初始化模型..."
python /app/scripts/download_models.py

# 启动服务
echo ""
echo "启动 API 服务..."
exec python -m src.main

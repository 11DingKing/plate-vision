#!/bin/bash
# 车牌识别系统启动脚本

set -e

echo "========================================"
echo "车牌识别系统启动"
echo "========================================"

# 切换到项目目录
cd "$(dirname "$0")/.."

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 创建必要目录
mkdir -p models logs output

# 初始化模型
echo ""
echo "正在初始化模型..."
python3 scripts/download_models.py

# 启动服务
echo ""
echo "启动服务..."
echo "API 文档: http://localhost:8000/docs"
echo "健康检查: http://localhost:8000/api/v1/health"
echo ""

python3 -m src.main

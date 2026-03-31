# ============================================
# 车牌识别系统后端 Dockerfile
# 支持多平台: linux/amd64, linux/arm64
# ============================================

# 使用官方 Python 镜像 (支持多平台)
FROM --platform=$BUILDPLATFORM python:3.10-slim AS builder

# 构建参数
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# 复制依赖文件
COPY requirements.txt .

# 安装依赖到指定目录
RUN pip install --upgrade pip setuptools wheel && \
    pip install --target=/build/deps -r requirements.txt

# ============================================
# 运行时镜像
# ============================================
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Shanghai \
    PYTHONPATH=/app/deps

WORKDIR /app

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 从构建阶段复制依赖
COPY --from=builder /build/deps /app/deps

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p logs output models data/samples scripts

# 设置启动脚本权限
RUN chmod +x scripts/entrypoint.sh

# 暴露端口
EXPOSE 8000

# 健康检查（增加启动等待时间，等待模型下载）
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# 使用启动脚本（自动下载模型）
ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]

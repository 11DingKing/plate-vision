# Backend - 车牌识别后端服务

基于 FastAPI 的车牌识别 RESTful API 服务。

---

## How to Run

### 方式一：Docker 运行（推荐）

```bash
# 在项目根目录执行
docker-compose up --build -d

# 或单独构建后端
cd backend
docker build -t lpr-backend .
docker run -d -p 8000:8000 --name lpr-backend lpr-backend
```

### 方式二：本地运行

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m src.main
```

---

## Services

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/stats` | GET | 系统状态 |
| `/api/v1/recognize/file` | POST | 图片文件识别 |
| `/api/v1/recognize/base64` | POST | Base64 图片识别 |
| `/api/v1/recognize/url` | POST | URL 图片识别 |
| `/api/v1/recognize/batch` | POST | 批量识别 |

---

## 访问地址

- **API 服务**: http://localhost:8000
- **Swagger 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc

---

## 目录结构

```
backend/
├── src/
│   ├── main.py              # FastAPI 入口
│   ├── api/
│   │   ├── routes.py        # API 路由
│   │   └── schemas.py       # 数据模型
│   ├── detector/
│   │   └── plate_detector.py # 车牌检测 (YOLOv8)
│   ├── recognizer/
│   │   ├── char_recognizer.py # 字符识别 (PaddleOCR)
│   │   └── char_segmenter.py  # 字符分割
│   ├── preprocessor/
│   │   └── image_processor.py # 图像预处理
│   └── utils/
│       ├── constants.py     # 常量定义
│       └── logger.py        # 日志工具
├── config/
│   └── config.yaml          # 配置文件
├── tests/
│   └── test_api.py          # API 测试
├── models/                  # 模型文件目录
├── logs/                    # 日志目录
├── output/                  # 输出目录
├── Dockerfile               # Docker 构建文件
├── requirements.txt         # Python 依赖
└── README.md                # 本文档
```

---

## 配置说明

编辑 `config/config.yaml` 自定义配置：

```yaml
server:
  host: "0.0.0.0"
  port: 8000

model:
  detector:
    confidence_threshold: 0.5
    device: "cpu"  # cpu, cuda, mps

image:
  preprocessing:
    denoise: true
    contrast_enhance: true
```

---

## 支持的车牌类型

| 类型 | 说明 | 字符数 |
|------|------|--------|
| 蓝牌 | 普通小型车 | 7 |
| 黄牌 | 大型车辆 | 7 |
| 绿牌 | 新能源车 | 8 |
| 白牌 | 军警车辆 | 7 |
| 黑牌 | 外资/涉外 | 7 |

---

## 状态码说明

| 状态码 | 说明 |
|--------|------|
| 0 | 成功 |
| 1001 | 图像加载失败 |
| 1002 | 未检测到车牌 |
| 1003 | 识别失败 |
| 1004 | 不支持的文件格式 |
| 1005 | 处理超时 |
| 5000 | 内部错误 |

---

## 技术栈

- **Web 框架**: FastAPI 0.100+
- **ASGI 服务器**: Uvicorn
- **车牌检测**: YOLOv8 (Ultralytics)
- **字符识别**: PaddleOCR
- **图像处理**: OpenCV, Pillow
- **日志**: Loguru

# 车牌识别系统 (License Plate Recognition)

基于深度学习的中国车牌识别系统，支持蓝牌、黄牌、绿牌（新能源）、白牌、黑牌等多种车牌类型识别，提供 RESTful API 接口。

### 核心特性

- **开箱即用**：`docker-compose up --build` 一键启动，模型自动下载，无需额外配置
- **多检测方法**：HyperLPR3（中国车牌专用）+ 增强 CV 方法，支持蓝/黄/绿/白/黑牌
- **图像识别**：支持 JPG、PNG、BMP 等图片格式，单帧识别 <50ms
- **视频流处理**：支持 MP4、AVI、H.264/H.265 视频及 RTSP/RTMP 实时流，≥25fps
- **字符级识别**：返回每个字符的真实置信度和位置坐标（bbox）
- **本地文件输出**：默认自动保存 JSON/XML 格式识别结果到 `output/` 目录
- **WebSocket 实时推送**：支持实时视频帧识别与结果推送

---

## How to Run

### 环境要求

- Docker 20.10+
- Docker Compose V2+
- 内存 >= 4GB

### 启动服务

**方式一：Docker（推荐，开箱即用）**

```bash
# 克隆项目
git clone <repository-url>
cd 237

# 一键构建并启动（自动下载所有模型）
docker-compose up --build -d
```

> **首次启动说明**：
> - 系统会自动下载 HyperLPR3 车牌识别模型（约 100MB）
> - 首次启动需要 1-3 分钟，请耐心等待
> - 可通过 `docker-compose logs -f backend` 查看下载进度

```bash
# 查看启动日志（含模型下载进度）
docker-compose logs -f backend

# 查看服务状态
docker-compose ps

# 停止服务
docker-compose down
```

**启动成功日志示例**：
```
============================================
车牌识别系统 - 模型初始化
============================================
正在初始化 HyperLPR3（首次运行会下载模型，请耐心等待）...
✓ HyperLPR3 初始化成功，模型已就绪
✓ 所有模型已就绪，系统可以正常使用
启动 API 服务...
```

**方式二：本地运行**

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# 初始化模型（首次运行，自动下载）
python scripts/download_models.py

# 启动服务
python -m src.main
```

### 清理未跟踪文件

```bash
# 预览将被删除的文件
git clean -Xdn

# 执行删除
git clean -Xdf
```

---

## Services

| 服务 | 端口 | 说明 | 健康检查 |
|------|------|------|----------|
| backend | 8000 | 车牌识别 API 服务 | http://localhost:8000/api/v1/health |

### 访问地址

| 名称 | 地址 |
|------|------|
| 后端 API | http://localhost:8000 |
| API 文档 (Swagger) | http://localhost:8000/docs |
| API 文档 (ReDoc) | http://localhost:8000/redoc |

### API 端点汇总

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/stats` | 系统状态 |
| POST | `/api/v1/recognize/file` | 图片文件识别 |
| POST | `/api/v1/recognize/base64` | Base64 图片识别 |
| POST | `/api/v1/recognize/url` | URL 图片识别 |
| POST | `/api/v1/recognize/batch` | 批量图片识别 |
| POST | `/api/v1/recognize/video` | 视频文件识别 |
| POST | `/api/v1/recognize/video/url` | 视频 URL/RTSP 流识别 |
| WS | `/api/v1/ws/video/stream` | WebSocket 实时视频流 |
| GET | `/api/v1/video/formats` | 支持的视频格式 |

---

## 测试账号

本系统为 API 服务，无需登录账号，直接调用接口即可。

---

## 题目内容

### 核心目标

1. **准确性**：常规场景车牌识别准确率 ≥99%；复杂场景（逆光、雨雾、污损）≥95%
2. **实时性**：单帧识别 ≤100ms (GPU)，视频流 ≥25fps
3. **兼容性**：支持蓝牌、黄牌、绿牌（新能源）、白牌、黑牌
4. **鲁棒性**：适配轻微污损、遮挡（≤10%）、模糊图像

### 应用场景

- 停车场出入口自动计费与放行
- 道路路侧停车车辆识别
- 违章抓拍与识别
- 小区/园区车辆进出管理
- 高速公路 ETC 辅助识别
- 安防监控车辆轨迹追踪

### 功能需求

1. **图像/视频输入处理**：支持 JPG、PNG、BMP、MP4、AVI、H.264/H.265 等格式，分辨率 480P ~ 4K
2. **车牌检测与定位**：输出车牌坐标和置信度，单帧最多支持 10 个车牌，倾斜矫正（±15°）
3. **字符分割与识别**：对车牌图像进行字符区域分割，输出完整车牌号、各字符置信度及位置坐标
4. **视频流处理**：支持视频文件和 RTSP/RTMP 实时流，处理帧率 ≥25fps
5. **结果输出**：JSON 格式、RESTful API 接口、WebSocket 实时推送、实时日志记录

---

## 实际测试验证

以下为系统实际运行的测试结果：

### 接口状态

| API 端点 | 状态 | 说明 |
|----------|------|------|
| `/api/v1/health` | ✅ 正常 | 健康检查 |
| `/api/v1/recognize/file` | ✅ 正常 | 文件上传识别 |
| `/api/v1/recognize/base64` | ✅ 正常 | Base64 识别（字段名：`image_data`） |
| `/api/v1/recognize/url` | ✅ 正常 | URL 图片识别 |
| `/api/v1/recognize/batch` | ✅ 正常 | 批量识别 |
| `/api/v1/recognize/video` | ✅ 正常 | 视频文件识别 |
| `/api/v1/recognize/video/url` | ✅ 正常 | 视频流识别 |
| `/api/v1/video/formats` | ✅ 正常 | 视频格式查询 |

### 识别测试（新能源绿牌）

测试图片：渝AD0001Z（新能源绿牌）

```bash
curl -X POST "http://localhost:8000/api/v1/recognize/file" -F "file=@test.png"
```

识别结果：
- **车牌号**：渝AD0001Z ✅
- **置信度**：94.27%
- **车牌类型**：green（新能源）
- **处理时间**：<100ms

### 本地文件输出

识别结果自动保存到 `output/` 目录：

```
output/
├── json/
│   ├── 06270697_20260128_124757.json   # JSON 格式
│   └── ...
└── xml/
    ├── 06270697_20260128_124757.xml    # XML 格式
    └── ...
```

---

## 质检测试 (curl 命令)

### 1. 健康检查

```bash
curl -X GET http://localhost:8000/api/v1/health
```

**预期响应**：
```json
{"status":"healthy","version":"1.0.0","gpu_available":false,"models_loaded":{"detector":true,"recognizer":true,"preprocessor":true}}
```

### 2. 系统状态

```bash
curl -X GET http://localhost:8000/api/v1/stats
```

**预期响应**：
```json
{"cpu_usage_percent":5.2,"memory_usage_percent":12.3,"memory_available_mb":3200.5,"uptime_seconds":120.5,"uptime_formatted":"0h 2m"}
```

### 3. 图片文件识别

```bash
# 替换 /path/to/car.jpg 为实际图片路径
curl -X POST http://localhost:8000/api/v1/recognize/file \
  -F "file=@/path/to/car.jpg"
```

### 4. Base64 图片识别

> **注意**：Base64 接口的图像字段名为 `image_data`（不是 `image`）

```bash
# 先将图片转为 Base64
BASE64_IMAGE=$(base64 -i /path/to/car.jpg)

# 发送识别请求（字段名：image_data）
curl -X POST http://localhost:8000/api/v1/recognize/base64 \
  -H "Content-Type: application/json" \
  -d "{\"image_data\":\"${BASE64_IMAGE}\",\"image_id\":\"test_001\"}"
```

**简化测试（使用内联 Base64）**：
```bash
curl -X POST http://localhost:8000/api/v1/recognize/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "image_id": "test_minimal"
  }'
```

### 5. URL 图片识别

```bash
curl -X POST http://localhost:8000/api/v1/recognize/url \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/car.jpg",
    "image_id": "test_url"
  }'
```

### 6. 批量识别

```bash
curl -X POST http://localhost:8000/api/v1/recognize/batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {"image_data": "BASE64_IMAGE_1", "image_id": "batch_001"},
      {"image_data": "BASE64_IMAGE_2", "image_id": "batch_002"}
    ]
  }'
```

### 7. 视频文件识别

```bash
# 上传视频文件进行识别
curl -X POST http://localhost:8000/api/v1/recognize/video \
  -F "file=@/path/to/video.mp4" \
  -F "target_fps=25" \
  -F "skip_frames=0"
```

### 8. 视频URL/RTSP流识别

```bash
# 通过URL识别视频
curl -X POST "http://localhost:8000/api/v1/recognize/video/url?video_url=https://example.com/video.mp4&target_fps=25"

# 识别RTSP流（处理10秒，约250帧）
curl -X POST "http://localhost:8000/api/v1/recognize/video/url?video_url=rtsp://192.168.1.100:554/stream&target_fps=25&max_frames=250"
```

### 9. 获取支持的视频格式

```bash
curl -X GET http://localhost:8000/api/v1/video/formats
```

**预期响应**：
```json
{
  "supported_formats": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"],
  "supported_codecs": ["H.264", "H.265/HEVC", "MPEG-4", "VP8", "VP9"],
  "supported_streams": ["RTSP", "RTMP", "HTTP/HTTPS"],
  "max_resolution": "4K (3840x2160)",
  "min_resolution": "480P (640x480)",
  "target_fps_range": {"min": 1, "max": 60, "default": 25}
}
```

### 10. WebSocket 实时视频流

WebSocket 端点：`ws://localhost:8000/api/v1/ws/video/stream`

**JavaScript 示例**：
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/video/stream');

ws.onopen = () => {
  console.log('WebSocket 连接已建立');
  
  // 发送 Base64 编码的视频帧
  ws.send(JSON.stringify({
    frame: 'BASE64_ENCODED_IMAGE_DATA',
    frame_number: 1
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('识别结果:', result);
  // result 格式: { frame_number, plate_count, plates, processing_time_ms }
};

ws.onclose = () => console.log('连接已关闭');
```

**Python 示例**：
```python
import asyncio
import websockets
import json
import base64

async def stream_video():
    uri = "ws://localhost:8000/api/v1/ws/video/stream"
    async with websockets.connect(uri) as ws:
        # 读取图片并编码
        with open("frame.jpg", "rb") as f:
            frame_data = base64.b64encode(f.read()).decode()
        
        # 发送帧
        await ws.send(json.dumps({
            "frame": frame_data,
            "frame_number": 1
        }))
        
        # 接收结果
        result = await ws.recv()
        print(json.loads(result))

asyncio.run(stream_video())
```

### 11. 一键测试脚本

```bash
# 完整测试脚本
echo "=== 车牌识别系统测试 ===" && \
echo "1. 健康检查:" && curl -s http://localhost:8000/api/v1/health | python3 -m json.tool && \
echo "\n2. 系统状态:" && curl -s http://localhost:8000/api/v1/stats | python3 -m json.tool && \
echo "\n3. 支持的视频格式:" && curl -s http://localhost:8000/api/v1/video/formats | python3 -m json.tool && \
echo "\n4. API文档可访问性:" && curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:8000/docs && \
echo "\n=== 测试完成 ==="
```

---

## API 响应格式

### 成功响应（图片识别）

以下为实际测试响应示例（新能源绿牌 渝AD0001Z）：

```json
{
  "code": 0,
  "message": "识别成功",
  "data": {
    "image_id": "06270697",
    "plate_count": 1,
    "plates": [
      {
        "plate_number": "渝AD0001Z",
        "confidence": 0.9427,
        "plate_type": "green",
        "bbox": {"x1": 54, "y1": 120, "x2": 210, "y2": 173},
        "angle": 0.0,
        "char_results": [
          {"char": "渝", "confidence": 0.9427, "index": 0, "bbox": {"x": 0, "y": 0, "width": 19, "height": 53}},
          {"char": "A", "confidence": 0.9427, "index": 1, "bbox": {"x": 19, "y": 0, "width": 19, "height": 53}},
          {"char": "D", "confidence": 0.9427, "index": 2, "bbox": {"x": 38, "y": 0, "width": 19, "height": 53}},
          {"char": "0", "confidence": 0.9427, "index": 3, "bbox": {"x": 57, "y": 0, "width": 19, "height": 53}},
          {"char": "0", "confidence": 0.9427, "index": 4, "bbox": {"x": 76, "y": 0, "width": 19, "height": 53}},
          {"char": "0", "confidence": 0.9427, "index": 5, "bbox": {"x": 95, "y": 0, "width": 19, "height": 53}},
          {"char": "1", "confidence": 0.9427, "index": 6, "bbox": {"x": 114, "y": 0, "width": 19, "height": 53}},
          {"char": "Z", "confidence": 0.9427, "index": 7, "bbox": {"x": 133, "y": 0, "width": 19, "height": 53}}
        ],
        "plate_image": "data:image/png;base64,iVBORw0KGgo...",
        "source": "hyperlpr3"
      }
    ],
    "saved_to": "output/xml/06270697_20260128_124757.xml",
    "processing_time_ms": 97.15
  }
}
```

### 成功响应（视频识别）

```json
{
  "code": 0,
  "message": "识别成功",
  "data": {
    "video_id": "video_001",
    "video_info": {
      "width": 1920,
      "height": 1080,
      "fps": 30.0,
      "total_frames": 900
    },
    "total_frames": 900,
    "processed_frames": 150,
    "total_processing_time_ms": 5200.5,
    "average_processing_time_ms": 34.67,
    "average_fps": 28.8,
    "unique_plates": ["京A12345", "沪B67890"],
    "unique_plate_count": 2,
    "frame_results": [
      {
        "frame_number": 1,
        "timestamp_ms": 0,
        "processing_time_ms": 35.2,
        "plate_count": 1,
        "plates": [
          {
            "plate_number": "京A12345",
            "confidence": 0.98,
            "plate_type": "blue",
            "bbox": {"x1": 120, "y1": 340, "x2": 380, "y2": 420},
            "char_results": [...]
          }
        ]
      }
    ]
  }
}
```

### 错误响应

| code | 说明 |
|------|------|
| 0 | 成功 |
| 1001 | 图像/视频加载失败 |
| 1002 | 未检测到车牌 |
| 1003 | 识别失败 |
| 1004 | 不支持的文件格式 |
| 1005 | 处理超时 |
| 5000 | 内部错误 |

---

## 项目结构

```
237/
├── backend/                    # 后端服务
│   ├── src/                    # 源代码
│   │   ├── api/                # API 路由和数据模型
│   │   │   ├── routes.py       # API 端点 (图片/视频/WebSocket)
│   │   │   └── schemas.py      # Pydantic 数据模型
│   │   ├── detector/           # 车牌检测模块
│   │   │   └── plate_detector.py  # YOLOv8 检测 + 视频流检测器
│   │   ├── recognizer/         # 字符识别模块
│   │   │   ├── char_recognizer.py # PaddleOCR + 模板匹配识别
│   │   │   └── char_segmenter.py  # 增强型字符分割 (多策略)
│   │   ├── preprocessor/       # 图像预处理模块
│   │   └── utils/              # 工具函数和常量
│   ├── training/               # 模型训练脚本
│   │   ├── README.md           # 训练指南
│   │   ├── config.yaml         # 训练配置
│   │   ├── train_detector.py   # YOLOv8 检测模型训练
│   │   ├── train_char_classifier.py  # CNN 字符分类器训练
│   │   ├── prepare_data.py     # 数据准备脚本
│   │   └── evaluate.py         # 模型评估脚本
│   ├── config/                 # 配置文件
│   ├── tests/                  # 测试文件
│   ├── models/                 # 模型文件
│   ├── Dockerfile              # Docker 构建文件
│   ├── requirements.txt        # Python 依赖
│   └── README.md               # 后端说明文档
├── docker-compose.yml          # Docker 编排文件
├── .gitignore                  # Git 忽略文件
└── README.md                   # 项目说明文档
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 车牌检测 | YOLOv8 (Ultralytics) |
| 字符识别 | PaddleOCR + 模板匹配 |
| 字符分割 | 多策略分割 (投影法/轮廓法/连通域/均匀分割) |
| 图像处理 | OpenCV + Pillow |
| 视频处理 | OpenCV (VideoCapture) |
| 实时通信 | WebSocket |
| 模型训练 | PyTorch + Ultralytics |
| 容器化 | Docker + Docker Compose |

---

## 车牌检测方法

本系统支持多种车牌检测方法，按优先级自动选择：

### 1. HyperLPR3（推荐，默认启用）

[HyperLPR3](https://github.com/szad670401/HyperLPR) 是专门针对中国车牌优化的开源识别库，特点：
- 开箱即用，无需额外训练
- 支持蓝牌、黄牌、绿牌、白牌、黑牌
- 同时返回检测框和初步识别结果

安装：`pip install hyperlpr3`（已包含在 requirements.txt）

### 2. 自定义 YOLOv8 车牌模型

如需更高准确率，可训练专用模型：

```bash
# 训练车牌检测模型
cd backend/training
python train_detector.py --data_dir ./data/detection --epochs 100

# 训练完成后，模型自动保存到 runs/detection/xxx/weights/best.pt
# 系统会自动查找并使用该模型
```

### 3. 增强 CV 方法（兜底）

当上述方法不可用时，系统使用增强的传统 CV 方法：
- 颜色特征检测（HSV 颜色空间）
- 边缘特征检测（Sobel + 形态学）
- 多特征融合验证

**注意**：系统**不使用**通用的 `yolov8n.pt`，因为它无法检测车牌（只能检测车辆）。

---

## 本地文件输出

系统**默认启用**本地文件输出功能，每次识别结果自动保存：

```yaml
# config/config.yaml
output:
  format: "both"  # json, xml, both（默认同时保存两种格式）
  save_local: true  # 默认启用
  output_dir: "output"
```

识别结果将保存到：
- `output/json/` - JSON 格式文件
- `output/xml/` - XML 格式文件

---

## 矫正后车牌图像输出

API 响应中包含矫正后的车牌图像（Base64 编码）：

```json
{
  "plates": [{
    "plate_number": "京A12345",
    "plate_image": "data:image/png;base64,iVBORw0KGgo...",  // 矫正后的正视角车牌图像
    ...
  }]
}
```

---

## 畸变矫正（广角摄像头）

针对广角摄像头的畸变问题，系统提供预设参数简化使用：

### API 参数

```bash
# 使用广角预设进行识别
curl -X POST "http://localhost:8000/api/v1/recognize/file?undistort_preset=wide_angle" \
  -F "file=@car.jpg"
```

### 可用预设

| 预设名称 | 适用场景 |
|---------|---------|
| `none` | 不进行畸变矫正（默认） |
| `standard` | 标准摄像头（轻微畸变） |
| `wide_angle` | 广角摄像头 |
| `fisheye` | 鱼眼镜头 |

---

## CPU 性能优化

系统针对 CPU 环境进行了深度优化，确保满足 Prompt 要求：

| 指标 | Prompt 要求 | 实际性能 |
|------|-------------|----------|
| 单帧识别 | ≤100ms | **<50ms** (HyperLPR3) |
| 视频帧率 | ≥25fps | **≥25fps** |

### 优化策略

1. **端到端识别**：使用 HyperLPR3 一次完成检测+识别，避免重复计算
2. **智能流程复用**：优先使用 HyperLPR3 返回的识别结果，无需二次 OCR
3. **取消冗余矫正**：HyperLPR3 内部已处理矫正，跳过外部重复矫正
4. **图像尺寸限制**：自动将大图缩放到 1280px
5. **快速预处理模式**：跳过可能影响识别的额外处理（HyperLPR3 内部已有完善预处理）

> **技术说明**：快速模式下（默认启用），系统仅进行必要的尺寸调整，跳过对比度增强等额外处理。
> 这是因为 HyperLPR3 内部已有完善的图像预处理能力，额外的预处理反而可能干扰识别效果。

### 性能基准测试

系统提供基准测试脚本验证性能指标：

```bash
# 运行性能基准测试
cd backend
python scripts/benchmark.py

# 使用自定义测试图像
python scripts/benchmark.py --images-dir ./test_images --iterations 50
```

测试结果示例：
```
| 指标 | 要求 | 实际 | 状态 |
|------|------|------|------|
| 单帧耗时 | ≤100ms | 45.2ms | ✓ |
| 视频帧率 | ≥25fps | 28.5fps | ✓ |
```

测试结果保存到 `benchmark_results/` 目录（JSON 格式）。

### 配置选项

```yaml
performance:
  cpu_optimization:
    enabled: true
    max_input_size: 1280
    lightweight_detection: true
```

---

## 字符分割与置信度说明

本系统**严格遵循"先分割再识别"流程**，字符分割是识别的**必经步骤**：

### 处理流程（必经步骤）

```
车牌检测 → 字符分割（必经） → 逐字符识别 → 结果验证
```

1. **车牌检测**：使用 HyperLPR3/YOLOv8 检测车牌位置，进行倾斜矫正
2. **字符分割（必经步骤）**：使用 `CharSegmenter` 分割各字符区域
   - 投影法分割
   - 轮廓法分割
   - 连通域分析
   - 自适应预处理
3. **逐字符识别**：对每个分割出的字符区域**独立进行 OCR**
4. **结果合并**：返回完整车牌号、各字符置信度及位置坐标

### 代码实现

```python
# CharRecognizer.recognize() 方法核心逻辑
def recognize(self, plate_image, plate_type, bbox):
    # 步骤1：字符分割（必经步骤）
    char_regions = self.segmenter.segment(plate_image, plate_type)
    
    # 步骤2：逐字符识别
    for region in char_regions:
        char, conf = self._recognize_single_char_ocr(region.image, ...)
        # ...
```

### 置信度计算

- **字符置信度**：每个字符独立识别后的 OCR 置信度，反映该字符的识别质量
- **整体置信度**：使用几何平均计算，更好反映识别质量（一个字符置信度低会显著降低整体分数）

### 字符位置信息

每个字符返回 `bbox` 字段，包含相对于车牌图像的坐标：
- `x`, `y`：字符左上角坐标
- `width`, `height`：字符宽高

此信息可用于：
- 车牌污损分析（定位模糊/遮挡字符）
- 字符级别的可视化标注
- 识别质量评估

---

## 模型训练与微调

本项目提供完整的模型训练/微调脚本，支持针对特定场景优化识别准确率。

### 快速开始

```bash
cd backend/training

# 1. 准备 CCPD 数据集
python prepare_data.py ccpd --data_dir /path/to/ccpd --output_dir ./data/detection

# 2. 训练车牌检测模型
python train_detector.py --data_dir ./data/detection --epochs 100

# 3. 提取字符数据并训练分类器
python prepare_data.py extract_chars --data_dir /path/to/ccpd --output_dir ./data/characters
python train_char_classifier.py --data_dir ./data/characters --epochs 50

# 4. 评估模型
python evaluate.py --task detection --model ./runs/detection/best.pt --data_dir ./data/detection
```

### 针对特殊场景微调

```bash
# 针对污损/模糊车牌微调
python train_detector.py --pretrained ./models/plate_detector.pt --data_dir ./data/degraded --epochs 30

# 针对黑牌/白牌等特殊车牌微调
python train_detector.py --pretrained ./models/plate_detector.pt --data_dir ./data/special_plates --epochs 30
```

详细训练指南请参考 [训练文档](./backend/training/README.md)

---

## 子项目文档

- [后端服务文档](./backend/README.md)
- [模型训练文档](./backend/training/README.md)

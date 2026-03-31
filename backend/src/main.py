"""
车牌识别系统 - 主入口
FastAPI应用程序
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.api.routes import router
from src.utils.logger import setup_logger, get_logger

# 加载配置
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}

# 初始化日志
log_config = config.get("logging", {})
setup_logger(
    log_level=log_config.get("level", "INFO"),
    log_dir=log_config.get("log_dir", "logs"),
    log_format=log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
)

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("=" * 50)
    logger.info("车牌识别系统启动中...")
    logger.info(f"配置文件: {CONFIG_PATH}")
    logger.info("=" * 50)
    
    # 预加载模型（可选）
    # from src.api.routes import get_detector, get_recognizer
    # get_detector()
    # get_recognizer()
    # logger.info("模型预加载完成")
    
    yield
    
    # 关闭时
    logger.info("车牌识别系统关闭")


# 创建FastAPI应用
app = FastAPI(
    title="车牌识别系统 API",
    description="""
## 功能说明

本系统提供完整的车牌识别功能，支持：

- 🚗 **多种车牌类型**：蓝牌、黄牌、绿牌（新能源）、白牌、黑牌
- 📷 **多种输入方式**：文件上传、Base64、URL
- 🎬 **视频流处理**：支持MP4、AVI、H.264/H.265等格式，≥25fps实时处理
- 🔄 **批量处理**：支持最多10张图片同时识别
- ⚡ **实时处理**：单帧识别时间 ≤100ms (CPU)，使用HyperLPR3可达<50ms
- 📊 **字符级置信度**：返回每个字符的真实识别置信度和位置信息
- 🔧 **开箱即用**：HyperLPR3端到端识别，无需额外下载模型

## 图片识别接口

1. 使用 `/api/v1/recognize/file` 上传图片文件
2. 使用 `/api/v1/recognize/base64` 发送Base64编码的图片
3. 使用 `/api/v1/recognize/url` 提供图片URL

## 视频流处理接口

1. 使用 `/api/v1/recognize/video` 上传视频文件
2. 使用 `/api/v1/recognize/video/url` 通过URL处理视频或RTSP流
3. 使用 `/api/v1/ws/video/stream` WebSocket实时视频流处理

## 返回格式

```json
{
    "code": 0,
    "message": "识别成功",
    "data": {
        "plate_count": 1,
        "plates": [
            {
                "plate_number": "京A12345",
                "confidence": 0.98,
                "plate_type": "blue",
                "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 260},
                "char_results": [
                    {"char": "京", "confidence": 0.99, "index": 0, "bbox": {"x": 5, "y": 2, "width": 25, "height": 36}},
                    {"char": "A", "confidence": 0.98, "index": 1, "bbox": {"x": 35, "y": 2, "width": 20, "height": 36}}
                ]
            }
        ]
    }
}
```
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # 自定义文档路径
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 - 返回简介页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>车牌识别系统</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 16px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .feature-list {
                list-style: none;
                padding: 0;
            }
            .feature-list li {
                padding: 12px 0;
                border-bottom: 1px solid #eee;
            }
            .feature-list li:last-child {
                border-bottom: none;
            }
            .btn {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                text-decoration: none;
                margin: 10px 10px 10px 0;
                transition: transform 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
            }
            .endpoints {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
            }
            code {
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 车牌识别系统</h1>
            
            <ul class="feature-list">
                <li>✅ 支持蓝牌、黄牌、绿牌（新能源）、白牌、黑牌</li>
                <li>✅ 支持图片文件上传、Base64、URL三种输入方式</li>
                <li>✅ <strong>视频流处理</strong>：支持MP4、AVI、H.264/H.265，≥25fps实时处理</li>
                <li>✅ <strong>字符级置信度</strong>：返回每个字符的真实识别置信度和位置</li>
                <li>✅ 支持批量识别（最多10张）</li>
                <li>✅ 单帧识别时间 < 200ms</li>
                <li>✅ 支持倾斜车牌自动矫正（±15°）</li>
                <li>✅ WebSocket实时视频流识别</li>
            </ul>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/docs" class="btn">📚 API文档</a>
                <a href="/redoc" class="btn">📖 ReDoc</a>
                <a href="/api/v1/health" class="btn">💚 健康检查</a>
            </div>
            
            <div class="endpoints">
                <h3>图片识别 API</h3>
                <p><code>POST /api/v1/recognize/file</code> - 上传图片文件识别</p>
                <p><code>POST /api/v1/recognize/base64</code> - Base64图片识别</p>
                <p><code>POST /api/v1/recognize/url</code> - URL图片识别</p>
                <p><code>POST /api/v1/recognize/batch</code> - 批量识别</p>
                
                <h3>视频流识别 API</h3>
                <p><code>POST /api/v1/recognize/video</code> - 上传视频文件识别</p>
                <p><code>POST /api/v1/recognize/video/url</code> - URL/RTSP视频流识别</p>
                <p><code>WS /api/v1/ws/video/stream</code> - WebSocket实时视频流</p>
                <p><code>GET /api/v1/video/formats</code> - 支持的视频格式</p>
                
                <h3>系统 API</h3>
                <p><code>GET /api/v1/health</code> - 健康检查</p>
                <p><code>GET /api/v1/stats</code> - 系统状态</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """自定义Swagger UI"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="车牌识别系统 - API文档",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )


def main():
    """主函数"""
    import uvicorn
    
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    workers = server_config.get("workers", 1)
    reload = server_config.get("reload", False)
    
    logger.info(f"启动服务器: http://{host}:{port}")
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()

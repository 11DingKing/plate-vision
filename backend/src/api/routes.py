"""
API路由定义
"""

import io
import base64
import time
import uuid
import tempfile
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .schemas import (
    RecognitionResponse, ImageRecognitionRequest, UrlRecognitionRequest,
    HealthResponse, BatchRecognitionRequest, BatchRecognitionResponse,
    PlateDetectionResult, BoundingBox, CharResultSchema, PlateTypeEnum,
    VideoRecognitionRequest, VideoRecognitionResponse, FrameResult, CharBoundingBox
)
from ..detector import PlateDetector
from ..detector.plate_detector import VideoPlateDetector
from ..recognizer import CharRecognizer
from ..preprocessor import ImageProcessor
from ..utils.logger import get_logger
from ..utils.constants import (
    ResponseCode, RESPONSE_MESSAGES, SUPPORTED_IMAGE_FORMATS, 
    SUPPORTED_VIDEO_FORMATS
)
from ..utils.output_saver import save_recognition_result, get_output_saver

# 创建路由器
router = APIRouter()
logger = get_logger("api")

# 全局组件实例（延迟初始化）
_detector: Optional[PlateDetector] = None
_video_detector: Optional[VideoPlateDetector] = None
_recognizer: Optional[CharRecognizer] = None
_preprocessor: Optional[ImageProcessor] = None
_start_time: float = time.time()


def get_detector() -> PlateDetector:
    """获取检测器实例"""
    global _detector
    if _detector is None:
        _detector = PlateDetector(use_yolo=True, device="cpu")
    return _detector


def get_recognizer() -> CharRecognizer:
    """获取识别器实例"""
    global _recognizer
    if _recognizer is None:
        _recognizer = CharRecognizer(use_gpu=False)
    return _recognizer


def get_preprocessor() -> ImageProcessor:
    """获取预处理器实例"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImageProcessor()
    return _preprocessor


def get_video_detector(target_fps: int = 25) -> VideoPlateDetector:
    """获取视频检测器实例"""
    global _video_detector
    if _video_detector is None:
        _video_detector = VideoPlateDetector(
            target_fps=target_fps,
            use_yolo=True,
            device="cpu"
        )
    return _video_detector


@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查接口
    
    返回服务状态和模型加载情况
    """
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        gpu_available = False
    
    models_loaded = {
        "detector": _detector is not None,
        "recognizer": _recognizer is not None,
        "preprocessor": _preprocessor is not None
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gpu_available=gpu_available,
        models_loaded=models_loaded
    )


@router.post("/recognize/file", response_model=RecognitionResponse, tags=["识别"])
async def recognize_file(
    file: UploadFile = File(..., description="上传的图像文件"),
    preprocess: bool = Query(True, description="是否进行图像预处理"),
    undistort_preset: Optional[str] = Query(
        None, 
        description="畸变矫正预设：none/standard/wide_angle/fisheye（针对广角摄像头）"
    )
):
    """
    上传图像文件进行车牌识别
    
    支持的格式：JPG, PNG, BMP, WEBP
    """
    start_time = time.time()
    image_id = str(uuid.uuid4())[:8]
    
    # 验证文件格式
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}"
        )
    
    try:
        # 读取图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return RecognitionResponse(
                code=ResponseCode.IMAGE_LOAD_ERROR,
                message=RESPONSE_MESSAGES[ResponseCode.IMAGE_LOAD_ERROR],
                data={"image_id": image_id}
            )
        
        # 执行识别
        result = await _process_image(image, image_id, preprocess, undistort_preset)
        
        # 计算处理时间
        processing_time = (time.time() - start_time) * 1000
        if result.data:
            result.data["processing_time_ms"] = round(processing_time, 2)
        
        return result
        
    except Exception as e:
        logger.error(f"处理图像失败: {e}")
        return RecognitionResponse(
            code=ResponseCode.INTERNAL_ERROR,
            message=str(e),
            data={"image_id": image_id}
        )


@router.post("/recognize/base64", response_model=RecognitionResponse, tags=["识别"])
async def recognize_base64(request: ImageRecognitionRequest):
    """
    Base64编码图像识别
    
    传入Base64编码的图像数据进行识别
    """
    start_time = time.time()
    image_id = request.image_id or str(uuid.uuid4())[:8]
    
    try:
        # 解码Base64
        image_data = base64.b64decode(request.image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return RecognitionResponse(
                code=ResponseCode.IMAGE_LOAD_ERROR,
                message=RESPONSE_MESSAGES[ResponseCode.IMAGE_LOAD_ERROR],
                data={"image_id": image_id}
            )
        
        # 执行识别
        result = await _process_image(image, image_id, preprocess=True)
        
        # 计算处理时间
        processing_time = (time.time() - start_time) * 1000
        if result.data:
            result.data["processing_time_ms"] = round(processing_time, 2)
        
        return result
        
    except Exception as e:
        logger.error(f"处理Base64图像失败: {e}")
        return RecognitionResponse(
            code=ResponseCode.INTERNAL_ERROR,
            message=str(e),
            data={"image_id": image_id}
        )


@router.post("/recognize/url", response_model=RecognitionResponse, tags=["识别"])
async def recognize_url(request: UrlRecognitionRequest):
    """
    URL图像识别
    
    传入图像URL进行识别
    """
    import httpx
    
    start_time = time.time()
    image_id = request.image_id or str(uuid.uuid4())[:8]
    
    try:
        # 下载图像
        async with httpx.AsyncClient() as client:
            response = await client.get(request.image_url, timeout=10.0)
            response.raise_for_status()
        
        # 解码图像
        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return RecognitionResponse(
                code=ResponseCode.IMAGE_LOAD_ERROR,
                message=RESPONSE_MESSAGES[ResponseCode.IMAGE_LOAD_ERROR],
                data={"image_id": image_id}
            )
        
        # 执行识别
        result = await _process_image(image, image_id, preprocess=True)
        
        # 计算处理时间
        processing_time = (time.time() - start_time) * 1000
        if result.data:
            result.data["processing_time_ms"] = round(processing_time, 2)
        
        return result
        
    except httpx.HTTPError as e:
        return RecognitionResponse(
            code=ResponseCode.IMAGE_LOAD_ERROR,
            message=f"下载图像失败: {e}",
            data={"image_id": image_id}
        )
    except Exception as e:
        logger.error(f"处理URL图像失败: {e}")
        return RecognitionResponse(
            code=ResponseCode.INTERNAL_ERROR,
            message=str(e),
            data={"image_id": image_id}
        )


@router.post("/recognize/batch", response_model=BatchRecognitionResponse, tags=["识别"])
async def recognize_batch(request: BatchRecognitionRequest):
    """
    批量识别
    
    最多支持10张图像同时识别
    """
    results = []
    success_count = 0
    
    for img_request in request.images:
        result = await recognize_base64(img_request)
        results.append(result)
        if result.code == 0:
            success_count += 1
    
    return BatchRecognitionResponse(
        code=0,
        message="批量识别完成",
        total=len(request.images),
        success_count=success_count,
        results=results
    )


async def _process_image(
    image: np.ndarray,
    image_id: str,
    preprocess: bool = True,
    undistort_preset: Optional[str] = None
) -> RecognitionResponse:
    """
    处理单张图像（优化版）
    
    性能优化：
    - 如果 HyperLPR3 已返回识别结果，直接使用，避免重复识别
    - 只有当检测器未返回识别结果时，才调用 CharRecognizer
    - 目标性能：单帧 ≤100ms (CPU)
    
    Args:
        image: 输入图像
        image_id: 图像ID
        preprocess: 是否预处理
        undistort_preset: 畸变矫正预设（针对广角摄像头）
        
    Returns:
        识别响应
    """
    detector = get_detector()
    
    # 根据畸变矫正参数创建预处理器
    if undistort_preset and undistort_preset != "none":
        from ..preprocessor import ImageProcessor
        preprocessor = ImageProcessor(camera_preset=undistort_preset, fast_mode=True)
    else:
        preprocessor = get_preprocessor()
    
    # 预处理（已优化为快速模式）
    if preprocess:
        image = preprocessor.process(image)
    
    # 检测车牌（HyperLPR3 同时返回检测和识别结果）
    detections = detector.detect(image)
    
    if not detections:
        return RecognitionResponse(
            code=ResponseCode.NO_PLATE_DETECTED,
            message=RESPONSE_MESSAGES[ResponseCode.NO_PLATE_DETECTED],
            data={
                "image_id": image_id,
                "plate_count": 0,
                "plates": []
            }
        )
    
    # 处理每个车牌
    plates = []
    recognizer = None  # 延迟加载，仅在需要时初始化
    
    for det in detections:
        # ===== 性能优化核心 =====
        # 优先使用 HyperLPR3 已返回的识别结果，避免重复识别
        if det.plate_text and len(det.plate_text) >= 7:
            # HyperLPR3 已完成识别，直接使用其结果
            plate_result = _build_result_from_hyperlpr(det)
        else:
            # HyperLPR3 未返回结果，需要调用 CharRecognizer
            if recognizer is None:
                recognizer = get_recognizer()
            
            plate_img = det.corrected_image if det.corrected_image is not None else det.cropped_image
            if plate_img is None:
                continue
            
            result = recognizer.recognize(plate_img, det.plate_type, det.bbox)
            plate_result = _build_result_from_recognizer(det, result)
        
        plates.append(plate_result)
    
    # 构建结果数据
    result_data = {
        "image_id": image_id,
        "plate_count": len(plates),
        "plates": plates
    }
    
    # 保存识别结果到本地文件 (JSON/XML)
    try:
        saved_path = save_recognition_result(result_data, image_id)
        if saved_path:
            result_data["saved_to"] = saved_path
    except Exception as e:
        logger.warning(f"保存本地文件失败: {e}")
    
    return RecognitionResponse(
        code=ResponseCode.SUCCESS,
        message=RESPONSE_MESSAGES[ResponseCode.SUCCESS],
        data=result_data
    )


def _build_result_from_hyperlpr(det) -> Dict[str, Any]:
    """
    从 HyperLPR3 检测结果构建 API 响应
    
    HyperLPR3 是端到端识别库，已包含识别结果，无需重复识别
    """
    plate_text = det.plate_text
    # 转换为 Python float，避免 numpy.float32 序列化问题
    confidence = float(det.confidence)
    
    # 构建字符级结果（基于车牌文本估算位置）
    char_results = []
    plate_image_base64 = None
    
    if det.cropped_image is not None:
        h, w = det.cropped_image.shape[:2]
        num_chars = len(plate_text)
        char_width = w // num_chars if num_chars > 0 else w
        
        for i, char in enumerate(plate_text):
            char_results.append({
                "char": char,
                "confidence": round(confidence, 4),
                "index": i,
                "bbox": {
                    "x": int(i * char_width),
                    "y": 0,
                    "width": int(char_width),
                    "height": int(h)
                }
            })
        
        # 生成矫正后的车牌图像 Base64
        plate_image_base64 = _encode_image_base64(
            det.corrected_image if det.corrected_image is not None else det.cropped_image
        )
    else:
        for i, char in enumerate(plate_text):
            char_results.append({
                "char": char,
                "confidence": round(confidence, 4),
                "index": i,
                "bbox": None
            })
    
    return {
        "plate_number": plate_text,
        "confidence": round(confidence, 4),
        "plate_type": det.plate_type,
        "bbox": {
            "x1": int(det.bbox[0]),
            "y1": int(det.bbox[1]),
            "x2": int(det.bbox[2]),
            "y2": int(det.bbox[3])
        },
        "angle": round(float(det.angle), 2),
        "char_results": char_results,
        "plate_image": plate_image_base64,
        "source": "hyperlpr3"
    }


def _build_result_from_recognizer(det, result) -> Dict[str, Any]:
    """
    从 CharRecognizer 识别结果构建 API 响应
    """
    # 生成矫正后的车牌图像 Base64
    plate_image_base64 = None
    if det.corrected_image is not None:
        plate_image_base64 = _encode_image_base64(det.corrected_image)
    elif det.cropped_image is not None:
        plate_image_base64 = _encode_image_base64(det.cropped_image)
    
    return {
        "plate_number": result.plate_number,
        "confidence": round(result.confidence, 4),
        "plate_type": det.plate_type,
        "bbox": {
            "x1": det.bbox[0],
            "y1": det.bbox[1],
            "x2": det.bbox[2],
            "y2": det.bbox[3]
        },
        "angle": round(det.angle, 2),
        "char_results": [
            {
                "char": cr.char,
                "confidence": round(cr.confidence, 4),
                "index": cr.index,
                "bbox": {
                    "x": cr.bbox[0],
                    "y": cr.bbox[1],
                    "width": cr.bbox[2],
                    "height": cr.bbox[3]
                } if cr.bbox else None
            }
            for cr in result.char_results
        ],
        "plate_image": plate_image_base64,  # 矫正后的车牌图像 (Base64)
        "source": "paddleocr"
    }


def _encode_image_base64(image: np.ndarray) -> Optional[str]:
    """
    将图像编码为 Base64 字符串
    
    Args:
        image: OpenCV 图像 (BGR)
        
    Returns:
        Base64 编码的 PNG 图像字符串
    """
    try:
        if image is None or image.size == 0:
            return None
        
        # 编码为 PNG
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        logger.warning(f"图像编码失败: {e}")
        return None


@router.get("/stats", tags=["系统"])
async def get_stats():
    """
    获取系统统计信息
    """
    import psutil
    
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # 内存使用
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # 运行时间
    uptime = time.time() - _start_time
    
    return {
        "cpu_usage_percent": cpu_percent,
        "memory_usage_percent": memory_percent,
        "memory_available_mb": memory.available / (1024 * 1024),
        "uptime_seconds": round(uptime, 2),
        "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
    }


# ==================== 视频流处理API ====================

@router.post("/recognize/video", response_model=VideoRecognitionResponse, tags=["视频识别"])
async def recognize_video(
    file: UploadFile = File(..., description="上传的视频文件"),
    target_fps: int = Query(25, ge=1, le=60, description="目标处理帧率"),
    skip_frames: int = Query(0, ge=0, description="每处理一帧后跳过的帧数"),
    max_frames: Optional[int] = Query(None, ge=1, description="最大处理帧数")
):
    """
    上传视频文件进行车牌识别
    
    支持的格式：MP4, AVI, MKV, MOV, WMV, FLV
    
    特性：
    - 实时帧率控制，支持 ≥25fps 处理速度
    - 返回每帧识别结果及统计信息
    - 自动去重，返回视频中出现的所有唯一车牌
    """
    start_time = time.time()
    video_id = str(uuid.uuid4())[:8]
    
    # 验证文件格式
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的视频格式: {suffix}，支持的格式: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # 处理视频
        result = await _process_video_file(
            tmp_path, video_id, target_fps, skip_frames, max_frames
        )
        
        # 清理临时文件
        try:
            Path(tmp_path).unlink()
        except:
            pass
        
        # 计算总处理时间
        total_time = (time.time() - start_time) * 1000
        if result.data:
            result.data["total_processing_time_ms"] = round(total_time, 2)
        
        return result
        
    except Exception as e:
        logger.error(f"处理视频失败: {e}")
        return VideoRecognitionResponse(
            code=ResponseCode.INTERNAL_ERROR,
            message=str(e),
            data={"video_id": video_id}
        )


@router.post("/recognize/video/url", response_model=VideoRecognitionResponse, tags=["视频识别"])
async def recognize_video_url(
    video_url: str = Query(..., description="视频URL"),
    video_id: Optional[str] = Query(None, description="视频ID（可选）"),
    target_fps: int = Query(25, ge=1, le=60, description="目标处理帧率"),
    skip_frames: int = Query(0, ge=0, description="每处理一帧后跳过的帧数"),
    max_frames: Optional[int] = Query(None, ge=1, description="最大处理帧数")
):
    """
    通过URL识别视频中的车牌
    
    支持HTTP/HTTPS视频链接和RTSP流
    """
    import httpx
    
    start_time = time.time()
    video_id = video_id or str(uuid.uuid4())[:8]
    
    try:
        # 判断是否为流媒体URL
        if video_url.startswith("rtsp://") or video_url.startswith("rtmp://"):
            # 直接处理流媒体
            result = await _process_video_stream(
                video_url, video_id, target_fps, max_frames
            )
        else:
            # 下载视频文件
            async with httpx.AsyncClient() as client:
                response = await client.get(video_url, timeout=60.0)
                response.raise_for_status()
            
            # 保存临时文件
            suffix = Path(video_url).suffix.lower() or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            # 处理视频
            result = await _process_video_file(
                tmp_path, video_id, target_fps, skip_frames, max_frames
            )
            
            # 清理
            try:
                Path(tmp_path).unlink()
            except:
                pass
        
        total_time = (time.time() - start_time) * 1000
        if result.data:
            result.data["total_processing_time_ms"] = round(total_time, 2)
        
        return result
        
    except httpx.HTTPError as e:
        return VideoRecognitionResponse(
            code=ResponseCode.IMAGE_LOAD_ERROR,
            message=f"下载视频失败: {e}",
            data={"video_id": video_id}
        )
    except Exception as e:
        logger.error(f"处理视频URL失败: {e}")
        return VideoRecognitionResponse(
            code=ResponseCode.INTERNAL_ERROR,
            message=str(e),
            data={"video_id": video_id}
        )


@router.websocket("/ws/video/stream")
async def websocket_video_stream(websocket: WebSocket):
    """
    WebSocket实时视频流处理
    
    客户端发送Base64编码的视频帧，服务端返回识别结果
    
    协议：
    - 客户端发送: {"frame": "base64_image_data", "frame_number": 1}
    - 服务端返回: {"frame_number": 1, "plates": [...], "processing_time_ms": 50.5}
    """
    await websocket.accept()
    logger.info("WebSocket视频流连接已建立")
    
    detector = get_detector()
    recognizer = get_recognizer()
    preprocessor = get_preprocessor()
    
    frame_count = 0
    unique_plates = set()
    
    try:
        while True:
            # 接收帧数据
            data = await websocket.receive_json()
            
            frame_start = time.time()
            frame_number = data.get("frame_number", frame_count)
            frame_data = data.get("frame")
            
            if not frame_data:
                await websocket.send_json({
                    "error": "Missing frame data",
                    "frame_number": frame_number
                })
                continue
            
            try:
                # 解码Base64帧
                image_data = base64.b64decode(frame_data)
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        "error": "Invalid frame data",
                        "frame_number": frame_number
                    })
                    continue
                
                # 预处理
                processed_frame = preprocessor.process(frame)
                
                # 检测
                detections = detector.detect(processed_frame)
                
                # 识别
                plates = []
                for det in detections:
                    plate_img = det.corrected_image if det.corrected_image is not None else det.cropped_image
                    if plate_img is None:
                        continue
                    
                    result = recognizer.recognize(plate_img, det.plate_type, det.bbox)
                    
                    plate_info = {
                        "plate_number": result.plate_number,
                        "confidence": round(result.confidence, 4),
                        "plate_type": det.plate_type,
                        "bbox": {
                            "x1": det.bbox[0],
                            "y1": det.bbox[1],
                            "x2": det.bbox[2],
                            "y2": det.bbox[3]
                        },
                        "char_results": [
                            {
                                "char": cr.char,
                                "confidence": round(cr.confidence, 4),
                                "index": cr.index,
                                "bbox": {
                                    "x": cr.bbox[0],
                                    "y": cr.bbox[1],
                                    "width": cr.bbox[2],
                                    "height": cr.bbox[3]
                                } if cr.bbox else None
                            }
                            for cr in result.char_results
                        ]
                    }
                    plates.append(plate_info)
                    unique_plates.add(result.plate_number)
                
                processing_time = (time.time() - frame_start) * 1000
                
                # 发送结果
                await websocket.send_json({
                    "frame_number": frame_number,
                    "plate_count": len(plates),
                    "plates": plates,
                    "processing_time_ms": round(processing_time, 2),
                    "unique_plates_count": len(unique_plates)
                })
                
                frame_count += 1
                
            except Exception as e:
                await websocket.send_json({
                    "error": str(e),
                    "frame_number": frame_number
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开，共处理 {frame_count} 帧")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass


async def _process_video_file(
    video_path: str,
    video_id: str,
    target_fps: int = 25,
    skip_frames: int = 0,
    max_frames: Optional[int] = None
) -> VideoRecognitionResponse:
    """
    处理视频文件（性能优化版）
    
    性能优化：
    - 优先使用 HyperLPR3 的识别结果，避免重复识别
    - 目标: ≥25fps (视频流)
    
    Args:
        video_path: 视频文件路径
        video_id: 视频ID
        target_fps: 目标帧率
        skip_frames: 跳过帧数
        max_frames: 最大处理帧数
        
    Returns:
        视频识别响应
    """
    detector = get_detector()
    recognizer = None  # 延迟加载，仅在需要时初始化
    preprocessor = get_preprocessor()
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return VideoRecognitionResponse(
            code=ResponseCode.IMAGE_LOAD_ERROR,
            message="无法打开视频文件",
            data={"video_id": video_id}
        )
    
    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"视频信息: {video_width}x{video_height}, {video_fps}fps, {total_frames}帧")
    
    # 计算帧间隔以达到目标帧率
    frame_interval = max(1, int(video_fps / target_fps))
    
    frame_results = []
    unique_plates = set()
    processed_count = 0
    frame_number = 0
    
    processing_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # 帧率控制
            if (frame_number - 1) % (frame_interval + skip_frames) != 0:
                continue
            
            # 检查最大帧数
            if max_frames and processed_count >= max_frames:
                break
            
            frame_start = time.time()
            
            # 预处理
            processed_frame = preprocessor.process(frame)
            
            # 检测
            detections = detector.detect(processed_frame)
            
            # 识别（优化：优先使用 HyperLPR3 结果）
            plates = []
            for det in detections:
                # === 性能优化：使用 HyperLPR3 的识别结果 ===
                if det.plate_text and len(det.plate_text) >= 7:
                    plate_info = _build_result_from_hyperlpr(det)
                else:
                    # 需要调用 CharRecognizer
                    if recognizer is None:
                        recognizer = get_recognizer()
                    
                    plate_img = det.corrected_image if det.corrected_image is not None else det.cropped_image
                    if plate_img is None:
                        continue
                    
                    result = recognizer.recognize(plate_img, det.plate_type, det.bbox)
                    plate_info = _build_result_from_recognizer(det, result)
                
                plates.append(plate_info)
                
                # 记录唯一车牌
                plate_number = plate_info.get("plate_number", "")
                confidence = plate_info.get("confidence", 0)
                if plate_number and confidence > 0.5:
                    unique_plates.add(plate_number)
            
            processing_time = (time.time() - frame_start) * 1000
            processing_times.append(processing_time)
            
            # 记录帧结果
            frame_result = {
                "frame_number": frame_number,
                "timestamp_ms": round((frame_number / video_fps) * 1000, 2) if video_fps > 0 else 0,
                "processing_time_ms": round(processing_time, 2),
                "plate_count": len(plates),
                "plates": plates
            }
            frame_results.append(frame_result)
            
            processed_count += 1
            
            # 为了实现异步处理，允许其他协程执行
            if processed_count % 10 == 0:
                await asyncio.sleep(0)
                
    finally:
        cap.release()
    
    # 计算统计信息
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    actual_fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
    
    return VideoRecognitionResponse(
        code=ResponseCode.SUCCESS,
        message=RESPONSE_MESSAGES[ResponseCode.SUCCESS],
        data={
            "video_id": video_id,
            "video_info": {
                "width": video_width,
                "height": video_height,
                "fps": video_fps,
                "total_frames": total_frames
            },
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "average_fps": round(actual_fps, 2),
            "unique_plates": list(unique_plates),
            "unique_plate_count": len(unique_plates),
            "frame_results": frame_results
        }
    )


async def _process_video_stream(
    stream_url: str,
    video_id: str,
    target_fps: int = 25,
    max_frames: Optional[int] = None
) -> VideoRecognitionResponse:
    """
    处理实时视频流 (RTSP/RTMP)
    
    Args:
        stream_url: 流媒体URL
        video_id: 视频ID
        target_fps: 目标帧率
        max_frames: 最大处理帧数
        
    Returns:
        视频识别响应
    """
    detector = get_detector()
    recognizer = get_recognizer()
    preprocessor = get_preprocessor()
    
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        return VideoRecognitionResponse(
            code=ResponseCode.IMAGE_LOAD_ERROR,
            message=f"无法连接视频流: {stream_url}",
            data={"video_id": video_id}
        )
    
    frame_results = []
    unique_plates = set()
    processed_count = 0
    frame_interval = 1.0 / target_fps
    last_process_time = 0
    
    processing_times = []
    
    # 设置默认最大帧数（流媒体需要限制）
    if max_frames is None:
        max_frames = target_fps * 10  # 默认处理10秒
    
    try:
        while processed_count < max_frames:
            current_time = time.time()
            
            # 帧率控制
            if current_time - last_process_time < frame_interval:
                ret, frame = cap.read()  # 读取但不处理
                continue
            
            ret, frame = cap.read()
            if not ret:
                break
            
            last_process_time = current_time
            frame_start = time.time()
            
            # 预处理
            processed_frame = preprocessor.process(frame)
            
            # 检测
            detections = detector.detect(processed_frame)
            
            # 识别
            plates = []
            for det in detections:
                plate_img = det.corrected_image if det.corrected_image is not None else det.cropped_image
                if plate_img is None:
                    continue
                
                result = recognizer.recognize(plate_img, det.plate_type, det.bbox)
                
                plate_info = {
                    "plate_number": result.plate_number,
                    "confidence": round(result.confidence, 4),
                    "plate_type": det.plate_type,
                    "bbox": {
                        "x1": det.bbox[0],
                        "y1": det.bbox[1],
                        "x2": det.bbox[2],
                        "y2": det.bbox[3]
                    }
                }
                plates.append(plate_info)
                
                if result.plate_number and result.confidence > 0.5:
                    unique_plates.add(result.plate_number)
            
            processing_time = (time.time() - frame_start) * 1000
            processing_times.append(processing_time)
            
            frame_result = {
                "frame_number": processed_count + 1,
                "timestamp_ms": round((processed_count / target_fps) * 1000, 2),
                "processing_time_ms": round(processing_time, 2),
                "plate_count": len(plates),
                "plates": plates
            }
            frame_results.append(frame_result)
            
            processed_count += 1
            
            # 允许其他协程执行
            await asyncio.sleep(0)
            
    finally:
        cap.release()
    
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    actual_fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
    
    return VideoRecognitionResponse(
        code=ResponseCode.SUCCESS,
        message=RESPONSE_MESSAGES[ResponseCode.SUCCESS],
        data={
            "video_id": video_id,
            "stream_url": stream_url,
            "processed_frames": processed_count,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "average_fps": round(actual_fps, 2),
            "unique_plates": list(unique_plates),
            "unique_plate_count": len(unique_plates),
            "frame_results": frame_results
        }
    )


@router.get("/video/formats", tags=["视频识别"])
async def get_supported_video_formats():
    """
    获取支持的视频格式列表
    """
    return {
        "supported_formats": SUPPORTED_VIDEO_FORMATS,
        "supported_codecs": ["H.264", "H.265/HEVC", "MPEG-4", "VP8", "VP9"],
        "supported_streams": ["RTSP", "RTMP", "HTTP/HTTPS"],
        "max_resolution": "4K (3840x2160)",
        "min_resolution": "480P (640x480)",
        "target_fps_range": {"min": 1, "max": 60, "default": 25}
    }

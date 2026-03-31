"""
API数据模型定义
使用Pydantic进行数据验证
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class PlateTypeEnum(str, Enum):
    """车牌类型枚举"""
    BLUE = "blue"
    YELLOW = "yellow"
    GREEN = "green"
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"


class CharBoundingBox(BaseModel):
    """字符边界框"""
    x: int = Field(..., description="字符左上角x坐标（相对于车牌图像）")
    y: int = Field(..., description="字符左上角y坐标（相对于车牌图像）")
    width: int = Field(..., description="字符宽度")
    height: int = Field(..., description="字符高度")


class CharResultSchema(BaseModel):
    """字符识别结果"""
    char: str = Field(..., description="识别的字符")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    index: int = Field(..., ge=0, description="字符位置索引")
    bbox: Optional[CharBoundingBox] = Field(default=None, description="字符边界框（相对于车牌图像）")


class BoundingBox(BaseModel):
    """边界框"""
    x1: int = Field(..., description="左上角x坐标")
    y1: int = Field(..., description="左上角y坐标")
    x2: int = Field(..., description="右下角x坐标")
    y2: int = Field(..., description="右下角y坐标")


class PlateDetectionResult(BaseModel):
    """单个车牌检测结果"""
    plate_number: str = Field(..., description="车牌号码")
    confidence: float = Field(..., ge=0, le=1, description="识别置信度")
    plate_type: PlateTypeEnum = Field(..., description="车牌类型")
    bbox: BoundingBox = Field(..., description="车牌边界框")
    angle: float = Field(default=0.0, description="倾斜角度")
    char_results: List[CharResultSchema] = Field(default=[], description="各字符识别结果")
    # 新增：矫正后的车牌图像
    plate_image: Optional[str] = Field(default=None, description="矫正后的车牌图像（Base64编码，PNG格式）")
    plate_image_url: Optional[str] = Field(default=None, description="矫正后的车牌图像URL（如果保存到本地）")


class RecognitionResponse(BaseModel):
    """识别响应"""
    code: int = Field(default=0, description="状态码，0表示成功")
    message: str = Field(default="success", description="响应消息")
    data: Optional[Dict[str, Any]] = Field(default=None, description="响应数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 0,
                "message": "识别成功",
                "data": {
                    "image_id": "img_001",
                    "processing_time_ms": 85.5,
                    "plate_count": 1,
                    "plates": [
                        {
                            "plate_number": "京A12345",
                            "confidence": 0.98,
                            "plate_type": "blue",
                            "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 260},
                            "angle": 0.0
                        }
                    ]
                }
            }
        }


class UndistortPreset(str, Enum):
    """畸变矫正预设"""
    NONE = "none"           # 不进行畸变矫正
    STANDARD = "standard"   # 标准摄像头（轻微畸变）
    WIDE_ANGLE = "wide_angle"  # 广角摄像头
    FISHEYE = "fisheye"     # 鱼眼镜头


class ImageRecognitionRequest(BaseModel):
    """图像识别请求（Base64）"""
    image_data: str = Field(..., description="Base64编码的图像数据")
    image_id: Optional[str] = Field(default=None, description="图像ID（可选）")
    undistort_preset: Optional[UndistortPreset] = Field(
        default=UndistortPreset.NONE, 
        description="畸变矫正预设（针对广角摄像头）"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_data": "/9j/4AAQSkZJRg...",
                "image_id": "img_001",
                "undistort_preset": "none"
            }
        }


class UrlRecognitionRequest(BaseModel):
    """图像识别请求（URL）"""
    image_url: str = Field(..., description="图像URL")
    image_id: Optional[str] = Field(default=None, description="图像ID（可选）")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="healthy", description="服务状态")
    version: str = Field(..., description="API版本")
    gpu_available: bool = Field(default=False, description="GPU是否可用")
    models_loaded: Dict[str, bool] = Field(default={}, description="模型加载状态")


class SystemInfoResponse(BaseModel):
    """系统信息响应"""
    cpu_usage: float = Field(..., description="CPU使用率")
    memory_usage: float = Field(..., description="内存使用率")
    gpu_info: Optional[Dict[str, Any]] = Field(default=None, description="GPU信息")
    uptime_seconds: float = Field(..., description="运行时间(秒)")


class BatchRecognitionRequest(BaseModel):
    """批量识别请求"""
    images: List[ImageRecognitionRequest] = Field(..., max_length=10, description="图像列表")


class BatchRecognitionResponse(BaseModel):
    """批量识别响应"""
    code: int = Field(default=0, description="状态码")
    message: str = Field(default="success", description="响应消息")
    total: int = Field(..., description="总数")
    success_count: int = Field(..., description="成功数量")
    results: List[RecognitionResponse] = Field(default=[], description="识别结果列表")


class VideoRecognitionRequest(BaseModel):
    """视频识别请求"""
    video_id: Optional[str] = Field(default=None, description="视频ID（可选）")
    target_fps: int = Field(default=25, ge=1, le=60, description="目标处理帧率")
    skip_frames: int = Field(default=0, ge=0, description="跳过的帧数（用于抽帧）")
    max_frames: Optional[int] = Field(default=None, description="最大处理帧数（可选）")


class FrameResult(BaseModel):
    """单帧识别结果"""
    frame_number: int = Field(..., description="帧序号")
    timestamp_ms: float = Field(..., description="帧时间戳（毫秒）")
    processing_time_ms: float = Field(..., description="处理耗时（毫秒）")
    plate_count: int = Field(default=0, description="检测到的车牌数量")
    plates: List[Dict[str, Any]] = Field(default=[], description="车牌检测结果列表")


class VideoRecognitionResponse(BaseModel):
    """视频识别响应"""
    code: int = Field(default=0, description="状态码")
    message: str = Field(default="success", description="响应消息")
    data: Optional[Dict[str, Any]] = Field(default=None, description="响应数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 0,
                "message": "视频识别完成",
                "data": {
                    "video_id": "video_001",
                    "total_frames": 150,
                    "processed_frames": 150,
                    "total_processing_time_ms": 5200.5,
                    "average_fps": 28.8,
                    "unique_plates": ["京A12345", "沪B67890"],
                    "frame_results": []
                }
            }
        }


class VideoStreamConfig(BaseModel):
    """视频流配置"""
    target_fps: int = Field(default=25, ge=1, le=60, description="目标帧率")
    buffer_size: int = Field(default=5, ge=1, le=30, description="缓冲区大小")
    skip_similar_frames: bool = Field(default=True, description="是否跳过相似帧")

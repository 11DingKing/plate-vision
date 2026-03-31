"""
图像预处理模块
包含降噪、对比度增强、白平衡调整、畸变矫正等功能
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import io

from ..utils.logger import LoggerMixin
from ..utils.constants import SUPPORTED_IMAGE_FORMATS, SUPPORTED_VIDEO_FORMATS


class ImageProcessor(LoggerMixin):
    """
    图像预处理器
    
    支持多种预处理功能：
    - 降噪、对比度增强、白平衡
    - 畸变矫正（支持预设相机参数或自定义参数）
    - 自动图像尺寸优化
    """
    
    # 预设的相机畸变参数（常见广角摄像头）
    CAMERA_PRESETS = {
        "standard": {
            # 标准摄像头（轻微畸变）
            "camera_matrix": np.array([
                [1000, 0, 640],
                [0, 1000, 360],
                [0, 0, 1]
            ], dtype=np.float64),
            "dist_coeffs": np.array([-0.1, 0.05, 0, 0, 0], dtype=np.float64)
        },
        "wide_angle": {
            # 广角摄像头（明显畸变）
            "camera_matrix": np.array([
                [800, 0, 640],
                [0, 800, 360],
                [0, 0, 1]
            ], dtype=np.float64),
            "dist_coeffs": np.array([-0.3, 0.1, 0, 0, 0], dtype=np.float64)
        },
        "fisheye": {
            # 鱼眼镜头（强畸变）
            "camera_matrix": np.array([
                [500, 0, 640],
                [0, 500, 360],
                [0, 0, 1]
            ], dtype=np.float64),
            "dist_coeffs": np.array([-0.5, 0.2, 0, 0, -0.05], dtype=np.float64)
        }
    }
    
    def __init__(
        self,
        denoise: bool = True,
        denoise_strength: int = 10,
        contrast_enhance: bool = True,
        contrast_alpha: float = 1.2,
        white_balance: bool = True,
        undistort: bool = False,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        camera_preset: Optional[str] = None,  # 新增：相机预设名称
        # CPU 性能优化选项
        max_input_size: int = 1280,
        fast_mode: bool = True
    ):
        """
        初始化图像预处理器
        
        Args:
            denoise: 是否进行降噪
            denoise_strength: 降噪强度 (1-20)
            contrast_enhance: 是否进行对比度增强
            contrast_alpha: 对比度增强系数 (1.0-2.0)
            white_balance: 是否进行白平衡调整
            undistort: 是否进行畸变矫正
            camera_matrix: 相机内参矩阵（自定义）
            dist_coeffs: 畸变系数（自定义）
            camera_preset: 相机预设名称 ("standard", "wide_angle", "fisheye")
            max_input_size: 最大输入尺寸（超过则自动缩放）
            fast_mode: 快速模式（简化预处理以提高性能）
        """
        self.denoise = denoise
        self.denoise_strength = denoise_strength
        self.contrast_enhance = contrast_enhance
        self.contrast_alpha = contrast_alpha
        self.white_balance = white_balance
        self.undistort = undistort
        self.max_input_size = max_input_size
        self.fast_mode = fast_mode
        
        # 设置相机参数
        if camera_preset and camera_preset in self.CAMERA_PRESETS:
            preset = self.CAMERA_PRESETS[camera_preset]
            self.camera_matrix = preset["camera_matrix"]
            self.dist_coeffs = preset["dist_coeffs"]
            self.undistort = True  # 使用预设时自动启用畸变矫正
            self.logger.info(f"使用相机预设: {camera_preset}")
        else:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
        
        self.logger.info(f"图像预处理器初始化完成，快速模式: {fast_mode}, 畸变矫正: {self.undistort}")
    
    def load_image(
        self,
        source: Union[str, bytes, np.ndarray, Path]
    ) -> Optional[np.ndarray]:
        """
        加载图像
        
        Args:
            source: 图像来源 (文件路径/字节数据/numpy数组)
            
        Returns:
            BGR格式的numpy数组
        """
        try:
            if isinstance(source, np.ndarray):
                return source.copy()
            
            if isinstance(source, bytes):
                # 从字节数据加载
                nparr = np.frombuffer(source, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return image
            
            if isinstance(source, (str, Path)):
                path = Path(source)
                if not path.exists():
                    self.logger.error(f"文件不存在: {path}")
                    return None
                
                suffix = path.suffix.lower()
                if suffix not in SUPPORTED_IMAGE_FORMATS:
                    self.logger.error(f"不支持的图像格式: {suffix}")
                    return None
                
                # 使用OpenCV加载图像
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    # 尝试使用PIL加载
                    pil_image = Image.open(path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                return image
            
            self.logger.error(f"不支持的输入类型: {type(source)}")
            return None
            
        except Exception as e:
            self.logger.error(f"加载图像失败: {e}")
            return None
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        执行完整的预处理流程
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            预处理后的图像
        """
        result = image.copy()
        
        # 0. 尺寸限制（CPU 性能优化）
        result = self._resize_if_needed(result)
        
        # 快速模式：只做尺寸调整，不做其他处理
        # HyperLPR3 内部已有完善的预处理，额外处理可能影响识别效果
        if self.fast_mode:
            return result
        
        # 完整模式
        # 1. 畸变矫正
        if self.undistort and self.camera_matrix is not None:
            result = self._undistort(result)
        
        # 2. 降噪
        if self.denoise:
            result = self._denoise(result)
        
        # 3. 白平衡
        if self.white_balance:
            result = self._white_balance(result)
        
        # 4. 对比度增强
        if self.contrast_enhance:
            result = self._enhance_contrast(result)
        
        return result
    
    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """
        如果图像尺寸超过限制，进行缩放
        
        这是 CPU 性能优化的关键步骤
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > self.max_input_size:
            scale = self.max_input_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.logger.debug(f"图像从 {w}x{h} 缩放到 {new_w}x{new_h}")
        
        return image
    
    def _fast_enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        快速对比度增强（比完整版本快 3-5 倍）
        
        使用简单的线性变换而非 CLAHE
        """
        # 转换到 LAB 空间只处理 L 通道
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 简单的线性拉伸
        l_min, l_max = l.min(), l.max()
        if l_max > l_min:
            l = ((l - l_min) * 255 / (l_max - l_min)).astype(np.uint8)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        图像降噪 (使用非局部均值降噪)
        """
        try:
            # 使用fastNlMeansDenoisingColored进行彩色图像降噪
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=self.denoise_strength,
                hColor=self.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
            return denoised
        except Exception as e:
            self.logger.warning(f"降噪处理失败: {e}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        对比度增强 (使用CLAHE算法)
        """
        try:
            # 转换到LAB颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 对L通道应用CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )
            l = clahe.apply(l)
            
            # 合并通道
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 应用alpha增益
            enhanced = cv2.convertScaleAbs(
                enhanced,
                alpha=self.contrast_alpha,
                beta=0
            )
            
            return enhanced
        except Exception as e:
            self.logger.warning(f"对比度增强失败: {e}")
            return image
    
    def _white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        自动白平衡 (使用灰度世界算法)
        """
        try:
            # 计算各通道均值
            b, g, r = cv2.split(image.astype(np.float32))
            b_avg = np.mean(b)
            g_avg = np.mean(g)
            r_avg = np.mean(r)
            
            # 计算灰度平均值
            avg = (b_avg + g_avg + r_avg) / 3
            
            # 计算增益
            b_gain = avg / (b_avg + 1e-6)
            g_gain = avg / (g_avg + 1e-6)
            r_gain = avg / (r_avg + 1e-6)
            
            # 应用增益
            b = np.clip(b * b_gain, 0, 255)
            g = np.clip(g * g_gain, 0, 255)
            r = np.clip(r * r_gain, 0, 255)
            
            balanced = cv2.merge([b, g, r]).astype(np.uint8)
            return balanced
        except Exception as e:
            self.logger.warning(f"白平衡调整失败: {e}")
            return image
    
    def _undistort(self, image: np.ndarray) -> np.ndarray:
        """
        畸变矫正
        """
        try:
            if self.camera_matrix is None or self.dist_coeffs is None:
                return image
            
            h, w = image.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                (w, h),
                1,
                (w, h)
            )
            
            undistorted = cv2.undistort(
                image,
                self.camera_matrix,
                self.dist_coeffs,
                None,
                new_camera_matrix
            )
            
            # 裁剪ROI区域
            x, y, w, h = roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y+h, x:x+w]
            
            return undistorted
        except Exception as e:
            self.logger.warning(f"畸变矫正失败: {e}")
            return image
    
    def resize(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        max_size: int = 1920,
        keep_ratio: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            target_size: 目标大小 (width, height)
            max_size: 最大边长
            keep_ratio: 是否保持宽高比
            
        Returns:
            (调整后的图像, 缩放比例)
        """
        h, w = image.shape[:2]
        
        if target_size:
            target_w, target_h = target_size
            if keep_ratio:
                scale = min(target_w / w, target_h / h)
            else:
                return cv2.resize(image, target_size), 1.0
        else:
            # 按最大边长缩放
            if max(w, h) <= max_size:
                return image.copy(), 1.0
            scale = max_size / max(w, h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return resized, scale
    
    def correct_rotation(
        self,
        image: np.ndarray,
        angle: float
    ) -> np.ndarray:
        """
        旋转矫正
        
        Args:
            image: 输入图像
            angle: 旋转角度 (度)
            
        Returns:
            旋转后的图像
        """
        if abs(angle) < 0.5:
            return image
        
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新图像大小
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 调整旋转矩阵
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # 执行旋转
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def crop_plate(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.1
    ) -> np.ndarray:
        """
        裁剪车牌区域
        
        Args:
            image: 原始图像
            bbox: 边界框 (x1, y1, x2, y2)
            padding: 边界扩展比例
            
        Returns:
            裁剪后的车牌图像
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 计算padding
        pw = int((x2 - x1) * padding)
        ph = int((y2 - y1) * padding)
        
        # 扩展边界
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)
        
        return image[y1:y2, x1:x2].copy()
    
    def perspective_transform(
        self,
        image: np.ndarray,
        src_points: np.ndarray,
        dst_size: Tuple[int, int] = (440, 140)
    ) -> np.ndarray:
        """
        透视变换 (将倾斜车牌矫正为正视角)
        
        Args:
            image: 输入图像
            src_points: 源四边形顶点 (4x2)
            dst_size: 目标大小 (width, height)
            
        Returns:
            透视变换后的图像
        """
        w, h = dst_size
        
        # 目标顶点
        dst_points = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)
        
        # 确保源点格式正确
        src_points = np.array(src_points, dtype=np.float32)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 执行透视变换
        warped = cv2.warpPerspective(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR
        )
        
        return warped
    
    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        """转换为灰度图"""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def to_binary(
        image: np.ndarray,
        threshold: int = 0,
        adaptive: bool = True
    ) -> np.ndarray:
        """
        转换为二值图
        
        Args:
            image: 输入图像 (灰度)
            threshold: 阈值 (仅adaptive=False时有效)
            adaptive: 是否使用自适应阈值
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if adaptive:
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
        else:
            if threshold == 0:
                _, binary = cv2.threshold(
                    gray, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            else:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        return binary

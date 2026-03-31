"""
车牌检测模块

支持多种检测方法（按优先级）：
1. HyperLPR3 - 专门针对中国车牌优化的开源库
2. 自定义 YOLOv8 车牌检测模型
3. 增强的传统 CV 方法（颜色+边缘+形状融合）

注意：不使用通用的 yolov8n.pt，因为它无法检测车牌。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time
import os

from ..utils.logger import LoggerMixin
from ..utils.constants import (
    PlateType, PLATE_COLOR_RANGES, PLATE_ASPECT_RATIOS,
    DEFAULT_CONFIDENCE_THRESHOLD, MAX_DETECTIONS_PER_FRAME
)


@dataclass
class PlateDetection:
    """车牌检测结果"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float                  # 置信度
    plate_type: str                    # 车牌类型
    angle: float                       # 倾斜角度
    corners: Optional[np.ndarray]     # 四角坐标
    cropped_image: Optional[np.ndarray] = None  # 裁剪的车牌图像
    corrected_image: Optional[np.ndarray] = None  # 矫正后的车牌图像
    plate_text: Optional[str] = None  # HyperLPR 识别的车牌号（如果有）


class PlateDetector(LoggerMixin):
    """
    车牌检测器
    
    支持多种检测方法：
    1. HyperLPR3 - 专业中国车牌识别库（推荐）
    2. 自定义 YOLO 车牌检测模型
    3. 增强的传统 CV 方法（兜底）
    """
    
    # 车牌专用模型文件名
    PLATE_MODEL_NAMES = [
        "plate_detect.pt",
        "plate_detector.pt", 
        "license_plate.pt",
        "lp_detect.pt",
        "best.pt"  # 训练输出的默认名称
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = 0.45,
        max_detections: int = MAX_DETECTIONS_PER_FRAME,
        device: str = "cpu",
        use_yolo: bool = True
    ):
        """
        初始化车牌检测器
        
        Args:
            model_path: 车牌检测模型路径
            confidence_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
            max_detections: 最大检测数量
            device: 推理设备 (cpu, cuda, mps)
            use_yolo: 是否尝试使用YOLO模型
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device
        
        # 检测方法状态
        self.hyperlpr_available = False
        self.yolo_plate_model_available = False
        self.hyperlpr_catcher = None
        self.yolo_model = None
        
        # 初始化检测方法
        self._init_hyperlpr()
        
        if use_yolo:
            self._init_yolo_plate_model()
        
        # 记录可用的检测方法
        self._log_available_methods()
    
    def _init_hyperlpr(self) -> None:
        """初始化 HyperLPR3（中国车牌识别专用库）"""
        try:
            import hyperlpr3 as lpr3
            
            # 创建识别器
            self.hyperlpr_catcher = lpr3.LicensePlateCatcher()
            self.hyperlpr_available = True
            self.logger.info("HyperLPR3 初始化成功")
            
        except ImportError:
            self.logger.warning(
                "HyperLPR3 未安装。建议安装以获得更好的检测效果: pip install hyperlpr3"
            )
            self.hyperlpr_available = False
        except Exception as e:
            self.logger.error(f"HyperLPR3 初始化失败: {e}")
            self.hyperlpr_available = False
    
    def _init_yolo_plate_model(self) -> None:
        """初始化 YOLO 车牌检测模型"""
        try:
            from ultralytics import YOLO
            
            # 查找车牌专用模型
            model_path = self._find_plate_model()
            
            if model_path:
                self.yolo_model = YOLO(model_path)
                self.yolo_model.to(self.device)
                self.yolo_plate_model_available = True
                self.logger.info(f"已加载车牌检测模型: {model_path}")
            else:
                # 如果 HyperLPR3 可用，YOLO 模型不是必须的，降低日志级别
                if self.hyperlpr_available:
                    self.logger.debug("YOLO 车牌模型未找到（HyperLPR3 已可用，无需额外模型）")
                else:
                    self.logger.warning(
                        "未找到车牌专用模型。请训练或下载车牌检测模型到 models/ 目录"
                    )
                self.yolo_plate_model_available = False
                
        except ImportError:
            self.logger.warning("ultralytics 未安装")
            self.yolo_plate_model_available = False
        except Exception as e:
            self.logger.error(f"YOLO 模型加载失败: {e}")
            self.yolo_plate_model_available = False
    
    def _find_plate_model(self) -> Optional[str]:
        """查找车牌检测模型文件"""
        # 优先使用指定的模型路径
        if self.model_path and Path(self.model_path).exists():
            return self.model_path
        
        # 模型目录
        base_dir = Path(__file__).parent.parent.parent
        models_dir = base_dir / "models"
        
        # 搜索车牌专用模型
        for model_name in self.PLATE_MODEL_NAMES:
            model_path = models_dir / model_name
            if model_path.exists():
                return str(model_path)
        
        # 检查训练输出目录
        training_dir = base_dir / "runs" / "detection"
        if training_dir.exists():
            for exp_dir in sorted(training_dir.iterdir(), reverse=True):
                weights_dir = exp_dir / "weights"
                if weights_dir.exists():
                    best_pt = weights_dir / "best.pt"
                    if best_pt.exists():
                        return str(best_pt)
        
        return None
    
    def _log_available_methods(self) -> None:
        """记录可用的检测方法"""
        methods = []
        if self.hyperlpr_available:
            methods.append("HyperLPR3")
        if self.yolo_plate_model_available:
            methods.append("YOLO车牌模型")
        methods.append("传统CV方法")  # 总是可用
        
        self.logger.info(f"车牌检测器初始化完成，可用方法: {', '.join(methods)}")
        
        # 如果 HyperLPR3 可用，显示友好提示
        if self.hyperlpr_available:
            self.logger.info("✓ 系统已就绪，使用 HyperLPR3 进行车牌识别")
    
    def detect(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> List[PlateDetection]:
        """
        检测图像中的车牌（性能优化版）
        
        按优先级尝试不同的检测方法：
        1. HyperLPR3（推荐，端到端识别，已包含矫正）
        2. YOLO 车牌模型（如果可用）
        3. 传统 CV 方法（兜底）
        
        性能优化：
        - HyperLPR3 结果无需二次矫正（内部已处理）
        - 目标: 单帧 ≤100ms (CPU), ≥25fps (视频流)
        
        Args:
            image: 输入图像 (BGR格式)
            preprocess: 是否进行预处理
            
        Returns:
            检测结果列表
        """
        start_time = time.time()
        detections = []
        use_hyperlpr = False
        
        # 方法1: HyperLPR3（最推荐，端到端检测+识别）
        if self.hyperlpr_available:
            detections = self._detect_with_hyperlpr(image)
            if detections:
                use_hyperlpr = True
                self.logger.debug(f"HyperLPR3 检测到 {len(detections)} 个车牌")
        
        # 方法2: YOLO 车牌模型
        if not detections and self.yolo_plate_model_available:
            detections = self._detect_with_yolo(image)
            if detections:
                self.logger.debug(f"YOLO 检测到 {len(detections)} 个车牌")
        
        # 方法3: 传统 CV 方法（兜底）
        if not detections:
            detections = self._detect_with_cv(image)
            if detections:
                self.logger.debug(f"CV 方法检测到 {len(detections)} 个车牌")
        
        # 后处理（仅对非 HyperLPR3 结果执行，避免冗余计算）
        for det in detections:
            # 裁剪车牌区域
            if det.cropped_image is None:
                det.cropped_image = self._crop_plate(image, det.bbox)
            
            # 检测车牌类型（HyperLPR3 已返回，跳过）
            if det.plate_type == PlateType.UNKNOWN:
                det.plate_type = self._classify_plate_type(det.cropped_image)
            
            # === 性能优化：HyperLPR3 内部已做矫正，跳过二次矫正 ===
            if use_hyperlpr:
                # HyperLPR3 返回的图像已经是矫正后的
                det.corrected_image = det.cropped_image
            else:
                # 非 HyperLPR3 检测需要手动矫正
                if det.angle == 0.0:
                    det.angle, det.corners = self._detect_angle(det.cropped_image)
                
                if det.corrected_image is None:
                    if abs(det.angle) > 1.0:
                        det.corrected_image = self._correct_perspective(
                            image, det.bbox, det.corners, det.angle
                        )
                    else:
                        det.corrected_image = det.cropped_image
        
        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"检测完成，发现 {len(detections)} 个车牌，耗时 {elapsed:.2f}ms")
        
        return detections
    
    def _detect_with_hyperlpr(self, image: np.ndarray) -> List[PlateDetection]:
        """使用 HyperLPR3 进行检测"""
        detections = []
        
        try:
            # HyperLPR3 检测
            results = self.hyperlpr_catcher(image)
            
            for result in results:
                # result 格式: (车牌号, 置信度, 类型ID, 边界框)
                plate_text = result[0]
                confidence = result[1]
                plate_type_id = result[2]
                bbox = result[3]  # [x1, y1, x2, y2]
                
                # 过滤低置信度
                if confidence < self.confidence_threshold:
                    continue
                
                # 转换车牌类型
                plate_type = self._convert_hyperlpr_type(plate_type_id)
                
                # 裁剪车牌图像
                x1, y1, x2, y2 = map(int, bbox)
                cropped = image[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None
                
                detection = PlateDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    plate_type=plate_type,
                    angle=0.0,
                    corners=None,
                    cropped_image=cropped,
                    plate_text=plate_text
                )
                detections.append(detection)
            
        except Exception as e:
            self.logger.error(f"HyperLPR3 检测失败: {e}")
        
        return detections[:self.max_detections]
    
    def _convert_hyperlpr_type(self, type_id: int) -> str:
        """转换 HyperLPR 车牌类型ID"""
        # HyperLPR3 类型映射
        type_map = {
            0: PlateType.BLUE,      # 蓝牌
            1: PlateType.YELLOW,    # 黄牌单层
            2: PlateType.YELLOW,    # 黄牌双层
            3: PlateType.GREEN,     # 新能源小车
            4: PlateType.GREEN,     # 新能源大车
            5: PlateType.WHITE,     # 白牌
            6: PlateType.BLACK,     # 黑牌
        }
        return type_map.get(type_id, PlateType.UNKNOWN)
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[PlateDetection]:
        """使用 YOLO 车牌模型进行检测"""
        detections = []
        
        try:
            results = self.yolo_model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    
                    # 验证宽高比
                    if h > 0:
                        aspect_ratio = w / h
                        if 1.5 <= aspect_ratio <= 5.0:
                            detection = PlateDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=conf,
                                plate_type=PlateType.UNKNOWN,
                                angle=0.0,
                                corners=None
                            )
                            detections.append(detection)
            
        except Exception as e:
            self.logger.error(f"YOLO 检测失败: {e}")
        
        return detections
    
    def _detect_with_cv(self, image: np.ndarray) -> List[PlateDetection]:
        """
        使用增强的传统 CV 方法检测车牌
        
        融合多种特征：
        1. 颜色特征 - 车牌特定颜色
        2. 边缘特征 - Sobel/Canny
        3. 形状特征 - 宽高比、面积
        4. 纹理特征 - 字符纹理
        """
        detections = []
        h_img, w_img = image.shape[:2]
        
        try:
            # 方法1: 基于颜色的检测
            color_candidates = self._detect_by_color(image)
            
            # 方法2: 基于边缘的检测
            edge_candidates = self._detect_by_edge(image)
            
            # 合并候选区域
            all_candidates = color_candidates + edge_candidates
            
            # 非极大值抑制
            all_candidates = self._nms_candidates(all_candidates, iou_threshold=0.3)
            
            # 验证每个候选区域
            for x, y, w, h, conf in all_candidates:
                # 提取候选区域
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_img, x + w)
                y2 = min(h_img, y + h)
                
                candidate_img = image[y1:y2, x1:x2]
                
                if candidate_img.size == 0:
                    continue
                
                # 验证是否为车牌
                is_plate, adjusted_conf = self._verify_plate_candidate(candidate_img, conf)
                
                if is_plate and adjusted_conf >= self.confidence_threshold:
                    detection = PlateDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=adjusted_conf,
                        plate_type=PlateType.UNKNOWN,
                        angle=0.0,
                        corners=None
                    )
                    detections.append(detection)
            
            # 按置信度排序
            detections.sort(key=lambda x: x.confidence, reverse=True)
            detections = detections[:self.max_detections]
            
        except Exception as e:
            self.logger.error(f"CV 检测失败: {e}")
        
        return detections
    
    def _detect_by_color(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """基于颜色特征检测车牌候选区域"""
        candidates = []
        
        # 转换到 HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_img, w_img = image.shape[:2]
        
        # 针对不同颜色车牌
        for plate_type, color_range in PLATE_COLOR_RANGES.items():
            lower = np.array(color_range["lower"])
            upper = np.array(color_range["upper"])
            
            # 创建颜色掩码
            mask = cv2.inRange(hsv, lower, upper)
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 面积过滤
                area = w * h
                if area < 1000 or area > h_img * w_img * 0.15:
                    continue
                
                # 宽高比过滤
                aspect_ratio = w / h if h > 0 else 0
                if not (2.0 <= aspect_ratio <= 5.0):
                    continue
                
                # 计算颜色覆盖度作为置信度
                roi_mask = mask[y:y+h, x:x+w]
                coverage = np.sum(roi_mask > 0) / roi_mask.size
                conf = coverage * 0.8  # 颜色方法基础置信度
                
                candidates.append((x, y, w, h, conf))
        
        return candidates
    
    def _detect_by_edge(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """基于边缘特征检测车牌候选区域"""
        candidates = []
        
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Sobel 边缘检测（水平方向）
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)
        sobel_x = np.uint8(255 * sobel_x / (sobel_x.max() + 1e-6))
        
        # 二值化
        _, binary = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作 - 连接水平线条
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 填充孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # 面积过滤
            area = w * h
            if area < 800 or area > h_img * w_img * 0.2:
                continue
            
            # 宽高比过滤
            aspect_ratio = w / h if h > 0 else 0
            if not (1.5 <= aspect_ratio <= 5.5):
                continue
            
            # 计算边缘密度
            roi = binary[y:y+h, x:x+w]
            edge_density = np.sum(roi > 0) / roi.size if roi.size > 0 else 0
            
            # 边缘密度在合理范围内（车牌字符产生的边缘）
            if 0.1 < edge_density < 0.6:
                conf = 0.3 + edge_density * 0.5  # 边缘方法基础置信度
                candidates.append((x, y, w, h, conf))
        
        return candidates
    
    def _nms_candidates(
        self,
        candidates: List[Tuple[int, int, int, int, float]],
        iou_threshold: float = 0.3
    ) -> List[Tuple[int, int, int, int, float]]:
        """非极大值抑制"""
        if not candidates:
            return []
        
        # 按置信度排序
        candidates = sorted(candidates, key=lambda x: x[4], reverse=True)
        
        kept = []
        used = [False] * len(candidates)
        
        for i, (x1, y1, w1, h1, conf1) in enumerate(candidates):
            if used[i]:
                continue
            
            kept.append(candidates[i])
            
            # 抑制重叠的候选
            for j, (x2, y2, w2, h2, conf2) in enumerate(candidates[i+1:], i+1):
                if used[j]:
                    continue
                
                # 计算 IoU
                iou = self._compute_iou(
                    (x1, y1, x1+w1, y1+h1),
                    (x2, y2, x2+w2, y2+h2)
                )
                
                if iou > iou_threshold:
                    used[j] = True
        
        return kept
    
    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """计算两个框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _verify_plate_candidate(
        self,
        candidate_img: np.ndarray,
        initial_conf: float
    ) -> Tuple[bool, float]:
        """
        验证候选区域是否为车牌
        
        通过多个特征验证：
        1. 颜色一致性
        2. 字符纹理特征
        3. 对比度特征
        """
        if candidate_img.size == 0:
            return False, 0.0
        
        h, w = candidate_img.shape[:2]
        if h < 10 or w < 30:
            return False, 0.0
        
        # 转灰度
        gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        
        # 特征1: 对比度
        contrast = gray.std()
        contrast_score = min(1.0, contrast / 50.0)
        
        # 特征2: 边缘密度（字符产生的边缘）
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        edge_score = 1.0 if 0.05 < edge_ratio < 0.4 else 0.5
        
        # 特征3: 垂直投影方差（字符会产生明显的峰谷）
        v_proj = np.sum(gray, axis=0)
        v_proj_std = np.std(v_proj) / (np.mean(v_proj) + 1e-6)
        proj_score = min(1.0, v_proj_std / 0.5)
        
        # 综合评分
        final_conf = initial_conf * 0.4 + contrast_score * 0.2 + edge_score * 0.2 + proj_score * 0.2
        
        # 阈值判断
        is_plate = final_conf >= self.confidence_threshold
        
        return is_plate, final_conf
    
    def _crop_plate(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.05
    ) -> np.ndarray:
        """裁剪车牌区域"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 添加padding
        pw = int((x2 - x1) * padding)
        ph = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)
        
        return image[y1:y2, x1:x2].copy()
    
    def _classify_plate_type(self, plate_image: np.ndarray) -> str:
        """根据颜色判断车牌类型"""
        try:
            if plate_image is None or plate_image.size == 0:
                return PlateType.UNKNOWN
            
            hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
            
            max_ratio = 0
            detected_type = PlateType.UNKNOWN
            
            for plate_type, color_range in PLATE_COLOR_RANGES.items():
                lower = np.array(color_range["lower"])
                upper = np.array(color_range["upper"])
                
                mask = cv2.inRange(hsv, lower, upper)
                ratio = np.sum(mask > 0) / mask.size
                
                if ratio > max_ratio and ratio > 0.3:
                    max_ratio = ratio
                    detected_type = plate_type
            
            return detected_type
            
        except Exception as e:
            self.logger.warning(f"车牌类型识别失败: {e}")
            return PlateType.UNKNOWN
    
    def _detect_angle(
        self,
        plate_image: np.ndarray
    ) -> Tuple[float, Optional[np.ndarray]]:
        """检测车牌倾斜角度"""
        try:
            if plate_image is None or plate_image.size == 0:
                return 0.0, None
            
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10
            )
            
            if lines is None:
                return 0.0, None
            
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                    if abs(angle) < 30:
                        angles.append(angle)
            
            if not angles:
                return 0.0, None
            
            median_angle = np.median(angles)
            corners = self._find_corners(plate_image)
            
            return median_angle, corners
            
        except Exception as e:
            self.logger.warning(f"角度检测失败: {e}")
            return 0.0, None
    
    def _find_corners(self, plate_image: np.ndarray) -> Optional[np.ndarray]:
        """查找车牌四角"""
        try:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            max_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
            
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            return box.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"四角检测失败: {e}")
            return None
    
    def _correct_perspective(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        corners: Optional[np.ndarray],
        angle: float
    ) -> np.ndarray:
        """透视矫正"""
        try:
            x1, y1, x2, y2 = bbox
            plate_img = image[y1:y2, x1:x2].copy()
            
            if corners is None or abs(angle) < 5:
                h, w = plate_img.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                corrected = cv2.warpAffine(plate_img, rotation_matrix, (w, h))
                return corrected
            
            h, w = plate_img.shape[:2]
            corners_sorted = self._sort_corners(corners)
            
            dst_w = 440
            dst_h = 140
            dst_points = np.array([
                [0, 0],
                [dst_w - 1, 0],
                [dst_w - 1, dst_h - 1],
                [0, dst_h - 1]
            ], dtype=np.float32)
            
            matrix = cv2.getPerspectiveTransform(corners_sorted, dst_points)
            corrected = cv2.warpPerspective(plate_img, matrix, (dst_w, dst_h))
            
            return corrected
            
        except Exception as e:
            self.logger.warning(f"透视矫正失败: {e}")
            return self._crop_plate(image, bbox)
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """对角点排序 (左上、右上、右下、左下)"""
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top_points = sorted_by_y[:2]
        top_points = top_points[np.argsort(top_points[:, 0])]
        bottom_points = sorted_by_y[2:]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
        
        return np.array([
            top_points[0],
            top_points[1],
            bottom_points[1],
            bottom_points[0]
        ], dtype=np.float32)


class VideoPlateDetector(PlateDetector):
    """
    视频流车牌检测器
    支持帧率控制和跟踪
    """
    
    def __init__(
        self,
        target_fps: int = 25,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = 0
        self.tracked_plates: Dict[int, PlateDetection] = {}
        self.next_track_id = 0
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float = None
    ) -> List[PlateDetection]:
        """处理视频帧"""
        current_time = timestamp or time.time()
        
        if current_time - self.last_frame_time < self.frame_interval:
            return list(self.tracked_plates.values())
        
        self.last_frame_time = current_time
        detections = self.detect(frame)
        self._update_tracking(detections)
        
        return detections
    
    def _update_tracking(self, detections: List[PlateDetection]) -> None:
        """更新跟踪状态"""
        self.tracked_plates.clear()
        for i, det in enumerate(detections):
            self.tracked_plates[i] = det

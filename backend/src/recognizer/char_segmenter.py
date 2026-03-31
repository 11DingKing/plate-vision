"""
字符分割模块
将车牌图像分割为单个字符

支持多种分割策略：
1. 垂直投影法 - 适用于清晰图像
2. 轮廓法 - 适用于一般场景
3. 连通域分析 - 适用于复杂背景
4. 均匀分割 - 兜底方案，适用于严重模糊/污损场景

自动评估图像质量，选择最佳分割策略。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import LoggerMixin
from ..utils.constants import PlateType


class SegmentMethod(Enum):
    """分割方法枚举"""
    PROJECTION = "projection"      # 垂直投影法
    CONTOUR = "contour"            # 轮廓法
    CONNECTED = "connected"        # 连通域分析
    UNIFORM = "uniform"            # 均匀分割
    HYBRID = "hybrid"              # 混合方法


@dataclass
class CharRegion:
    """字符区域"""
    image: np.ndarray              # 字符图像
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    index: int                      # 字符位置索引
    confidence: float = 1.0         # 分割置信度


@dataclass
class ImageQuality:
    """图像质量评估结果"""
    sharpness: float        # 清晰度 (0-1)
    contrast: float         # 对比度 (0-1)
    noise_level: float      # 噪声水平 (0-1, 越低越好)
    degradation: float      # 退化程度 (0-1, 越低越好)
    
    @property
    def overall_score(self) -> float:
        """综合质量分数"""
        return (self.sharpness * 0.3 + self.contrast * 0.3 + 
                (1 - self.noise_level) * 0.2 + (1 - self.degradation) * 0.2)


class CharSegmenter(LoggerMixin):
    """
    增强型字符分割器
    
    使用多种分割方法和自适应策略来提高复杂场景下的鲁棒性：
    - 自动评估图像质量
    - 根据质量选择最佳分割策略
    - 多种预处理方法适应不同场景
    - 均匀分割作为最终兜底
    """
    
    def __init__(
        self,
        target_char_height: int = 32,
        min_char_width: int = 8,
        max_char_width: int = 40,
        quality_threshold: float = 0.6
    ):
        """
        初始化字符分割器
        
        Args:
            target_char_height: 目标字符高度
            min_char_width: 最小字符宽度
            max_char_width: 最大字符宽度
            quality_threshold: 图像质量阈值，低于此值使用增强策略
        """
        self.target_char_height = target_char_height
        self.min_char_width = min_char_width
        self.max_char_width = max_char_width
        self.quality_threshold = quality_threshold
        
        self.logger.info("增强型字符分割器初始化完成")
    
    def segment(
        self,
        plate_image: np.ndarray,
        plate_type: str = PlateType.BLUE,
        method: SegmentMethod = None
    ) -> List[CharRegion]:
        """
        分割车牌字符
        
        Args:
            plate_image: 车牌图像
            plate_type: 车牌类型
            method: 指定分割方法，None 则自动选择
            
        Returns:
            字符区域列表
        """
        if plate_image is None or plate_image.size == 0:
            return []
        
        # 确定字符数量
        num_chars = 8 if plate_type == PlateType.GREEN else 7
        
        # 评估图像质量
        quality = self._assess_image_quality(plate_image)
        self.logger.debug(f"图像质量评分: {quality.overall_score:.2f}")
        
        # 根据质量选择预处理策略
        processed_images = self._adaptive_preprocess(plate_image, quality)
        
        # 选择分割方法
        if method is None:
            method = self._select_method(quality)
        
        # 尝试多种方法
        char_regions = self._try_segment_methods(
            processed_images, plate_image, num_chars, method, quality
        )
        
        # 如果所有方法都失败，使用均匀分割
        if len(char_regions) < num_chars - 2:
            self.logger.warning(f"分割方法效果不佳，使用均匀分割")
            char_regions = self._segment_uniform(plate_image, num_chars)
        
        # 过滤和排序
        char_regions = self._filter_regions(char_regions, num_chars)
        
        # 标准化字符大小
        char_regions = self._normalize_chars(char_regions)
        
        # 设置分割置信度
        self._set_region_confidence(char_regions, quality)
        
        self.logger.debug(f"分割得到 {len(char_regions)} 个字符")
        
        return char_regions
    
    def _assess_image_quality(self, image: np.ndarray) -> ImageQuality:
        """
        评估图像质量
        
        使用多个指标评估图像的清晰度、对比度、噪声等
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 清晰度评估 - 使用拉普拉斯算子的方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = min(1.0, laplacian.var() / 500.0)
        
        # 2. 对比度评估 - 使用灰度直方图的标准差
        contrast = min(1.0, gray.std() / 80.0)
        
        # 3. 噪声评估 - 使用高频分量的能量
        h, w = gray.shape
        if h > 10 and w > 10:
            # 使用小波变换或高通滤波估计噪声
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_freq = cv2.filter2D(gray.astype(float), -1, kernel)
            noise_level = min(1.0, np.abs(high_freq).mean() / 50.0)
        else:
            noise_level = 0.5
        
        # 4. 退化程度评估 - 边缘检测的响应
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        # 期望的边缘比例约为 0.1-0.3
        if edge_ratio < 0.05:
            degradation = 0.8  # 过度模糊
        elif edge_ratio > 0.5:
            degradation = 0.6  # 过度噪声
        else:
            degradation = max(0, 1.0 - edge_ratio * 3)
        
        return ImageQuality(
            sharpness=sharpness,
            contrast=contrast,
            noise_level=noise_level,
            degradation=degradation
        )
    
    def _select_method(self, quality: ImageQuality) -> SegmentMethod:
        """根据图像质量选择分割方法"""
        score = quality.overall_score
        
        if score > 0.7:
            # 高质量图像，优先使用投影法
            return SegmentMethod.PROJECTION
        elif score > 0.5:
            # 中等质量，使用混合方法
            return SegmentMethod.HYBRID
        elif score > 0.3:
            # 较低质量，使用连通域分析
            return SegmentMethod.CONNECTED
        else:
            # 严重退化，直接使用均匀分割
            return SegmentMethod.UNIFORM
    
    def _adaptive_preprocess(
        self,
        image: np.ndarray,
        quality: ImageQuality
    ) -> Dict[str, np.ndarray]:
        """
        自适应预处理
        
        根据图像质量选择不同的预处理策略
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        results = {}
        
        # 策略1: 标准预处理
        results['standard'] = self._preprocess_standard(gray)
        
        # 策略2: 增强对比度后预处理
        if quality.contrast < 0.5:
            enhanced = self._enhance_contrast(gray)
            results['enhanced'] = self._preprocess_standard(enhanced)
        
        # 策略3: 去噪后预处理
        if quality.noise_level > 0.3:
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            results['denoised'] = self._preprocess_standard(denoised)
        
        # 策略4: 锐化后预处理
        if quality.sharpness < 0.5:
            sharpened = self._sharpen_image(gray)
            results['sharpened'] = self._preprocess_standard(sharpened)
        
        # 策略5: Otsu 二值化
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        results['otsu'] = otsu
        
        return results
    
    def _preprocess_standard(self, gray: np.ndarray) -> np.ndarray:
        """标准预处理"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 自适应二值化
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # 形态学操作 - 去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """增强对比度"""
        # CLAHE (限制对比度自适应直方图均衡化)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced
    
    def _sharpen_image(self, gray: np.ndarray) -> np.ndarray:
        """锐化图像"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _try_segment_methods(
        self,
        processed_images: Dict[str, np.ndarray],
        original: np.ndarray,
        num_chars: int,
        preferred_method: SegmentMethod,
        quality: ImageQuality
    ) -> List[CharRegion]:
        """
        尝试多种分割方法，返回最佳结果
        """
        best_regions = []
        best_score = -1
        
        # 按优先级尝试不同方法
        methods_to_try = self._get_method_priority(preferred_method)
        
        for method in methods_to_try:
            for proc_name, processed in processed_images.items():
                if method == SegmentMethod.PROJECTION:
                    regions = self._segment_by_projection(processed, original)
                elif method == SegmentMethod.CONTOUR:
                    regions = self._segment_by_contour(processed, original)
                elif method == SegmentMethod.CONNECTED:
                    regions = self._segment_by_connected_components(processed, original)
                elif method == SegmentMethod.UNIFORM:
                    regions = self._segment_uniform(original, num_chars)
                else:
                    continue
                
                # 评估结果
                score = self._evaluate_segmentation(regions, num_chars)
                
                if score > best_score:
                    best_score = score
                    best_regions = regions
                
                # 如果达到很好的结果，直接返回
                if score > 0.9:
                    return best_regions
        
        return best_regions
    
    def _get_method_priority(self, preferred: SegmentMethod) -> List[SegmentMethod]:
        """获取方法优先级列表"""
        all_methods = [
            SegmentMethod.PROJECTION,
            SegmentMethod.CONTOUR,
            SegmentMethod.CONNECTED,
            SegmentMethod.UNIFORM
        ]
        
        # 将首选方法移到最前
        if preferred in all_methods:
            all_methods.remove(preferred)
            all_methods.insert(0, preferred)
        
        return all_methods
    
    def _evaluate_segmentation(
        self,
        regions: List[CharRegion],
        expected_count: int
    ) -> float:
        """
        评估分割结果的质量
        
        Returns:
            质量分数 (0-1)
        """
        if not regions:
            return 0.0
        
        # 因素1: 字符数量匹配度
        count_score = 1.0 - abs(len(regions) - expected_count) / expected_count
        count_score = max(0, count_score)
        
        # 因素2: 字符宽度一致性
        widths = [r.bbox[2] for r in regions]
        if len(widths) > 1:
            width_std = np.std(widths)
            width_mean = np.mean(widths)
            width_consistency = 1.0 - min(1.0, width_std / (width_mean + 1e-6))
        else:
            width_consistency = 0.5
        
        # 因素3: 字符间距规律性
        if len(regions) > 1:
            positions = [r.bbox[0] for r in regions]
            gaps = [positions[i+1] - positions[i] - widths[i] 
                    for i in range(len(positions)-1)]
            if len(gaps) > 1:
                gap_std = np.std(gaps)
                gap_mean = np.mean(gaps)
                gap_regularity = 1.0 - min(1.0, gap_std / (abs(gap_mean) + 1e-6))
            else:
                gap_regularity = 0.5
        else:
            gap_regularity = 0.5
        
        # 综合评分
        score = (count_score * 0.5 + width_consistency * 0.25 + gap_regularity * 0.25)
        
        return score
    
    def _segment_by_projection(
        self,
        binary: np.ndarray,
        original: np.ndarray
    ) -> List[CharRegion]:
        """使用垂直投影法分割"""
        char_regions = []
        
        h, w = binary.shape
        
        # 计算垂直投影
        projection = np.sum(binary, axis=0)
        
        # 使用自适应阈值
        threshold = max(h * 0.1, np.mean(projection) * 0.3)
        
        # 查找字符边界
        in_char = False
        start = 0
        boundaries = []
        
        for i in range(w):
            if not in_char and projection[i] > threshold:
                in_char = True
                start = i
            elif in_char and projection[i] <= threshold:
                in_char = False
                if i - start >= self.min_char_width:
                    boundaries.append((start, i))
        
        # 处理最后一个字符
        if in_char and w - start >= self.min_char_width:
            boundaries.append((start, w))
        
        # 提取字符
        for idx, (x1, x2) in enumerate(boundaries):
            # 计算有效的y边界
            char_binary = binary[:, x1:x2]
            h_proj = np.sum(char_binary, axis=1)
            y_start, y_end = self._find_vertical_bounds(h_proj, h)
            
            # 裁剪字符
            char_img = original[y_start:y_end, x1:x2].copy()
            
            if char_img.size == 0:
                continue
            
            char_region = CharRegion(
                image=char_img,
                bbox=(x1, y_start, x2 - x1, y_end - y_start),
                index=idx
            )
            char_regions.append(char_region)
        
        return char_regions
    
    def _segment_by_contour(
        self,
        binary: np.ndarray,
        original: np.ndarray
    ) -> List[CharRegion]:
        """使用轮廓法分割"""
        char_regions = []
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        h_img, w_img = binary.shape
        
        # 筛选和排序轮廓
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤条件
            if w < self.min_char_width or w > self.max_char_width:
                continue
            if h < h_img * 0.3 or h > h_img * 0.95:
                continue
            
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < 0.8 or aspect_ratio > 4.0:
                continue
            
            valid_contours.append((x, y, w, h))
        
        # 按x坐标排序
        valid_contours.sort(key=lambda c: c[0])
        
        # 合并重叠的轮廓
        valid_contours = self._merge_overlapping_contours(valid_contours)
        
        # 提取字符
        for idx, (x, y, w, h) in enumerate(valid_contours):
            char_img = original[y:y+h, x:x+w].copy()
            if char_img.size == 0:
                continue
            char_region = CharRegion(
                image=char_img,
                bbox=(x, y, w, h),
                index=idx
            )
            char_regions.append(char_region)
        
        return char_regions
    
    def _segment_by_connected_components(
        self,
        binary: np.ndarray,
        original: np.ndarray
    ) -> List[CharRegion]:
        """
        使用连通域分析进行分割
        
        更适合处理复杂背景和部分遮挡的情况
        """
        char_regions = []
        
        h_img, w_img = binary.shape
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # 筛选有效的连通域 (跳过背景 label=0)
        valid_components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # 面积过滤
            if area < 50 or area > h_img * w_img * 0.3:
                continue
            
            # 尺寸过滤
            if w < self.min_char_width or w > self.max_char_width:
                continue
            if h < h_img * 0.3 or h > h_img * 0.95:
                continue
            
            # 宽高比过滤
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 5.0:
                continue
            
            valid_components.append((x, y, w, h, area, i))
        
        # 按x坐标排序
        valid_components.sort(key=lambda c: c[0])
        
        # 提取字符
        for idx, (x, y, w, h, area, label) in enumerate(valid_components):
            # 创建掩码
            mask = (labels[y:y+h, x:x+w] == label).astype(np.uint8) * 255
            
            # 提取字符图像
            char_img = original[y:y+h, x:x+w].copy()
            
            # 应用掩码（可选，保留原始图像以保留更多信息）
            # char_img = cv2.bitwise_and(char_img, char_img, mask=mask)
            
            if char_img.size == 0:
                continue
            
            char_region = CharRegion(
                image=char_img,
                bbox=(x, y, w, h),
                index=idx
            )
            char_regions.append(char_region)
        
        return char_regions
    
    def _segment_uniform(
        self,
        original: np.ndarray,
        num_chars: int
    ) -> List[CharRegion]:
        """
        均匀分割 - 兜底方案
        
        当其他方法都失败时使用，适用于严重模糊/污损的图像
        """
        char_regions = []
        
        h, w = original.shape[:2]
        
        # 计算字符宽度
        # 考虑省份后可能有间隔点
        if num_chars == 7:
            # 普通车牌: 省 字母 · 5位
            positions = self._calculate_uniform_positions_7(w)
        else:
            # 新能源车牌: 省 字母 D/F 5位
            positions = self._calculate_uniform_positions_8(w)
        
        # 计算垂直边界
        y_margin = int(h * 0.05)
        y_start = y_margin
        y_end = h - y_margin
        
        # 提取字符
        for idx, (x1, x2) in enumerate(positions):
            x1 = max(0, x1)
            x2 = min(w, x2)
            
            char_img = original[y_start:y_end, x1:x2].copy()
            
            if char_img.size == 0:
                continue
            
            char_region = CharRegion(
                image=char_img,
                bbox=(x1, y_start, x2 - x1, y_end - y_start),
                index=idx,
                confidence=0.6  # 均匀分割置信度较低
            )
            char_regions.append(char_region)
        
        return char_regions
    
    def _calculate_uniform_positions_7(self, width: int) -> List[Tuple[int, int]]:
        """计算7位车牌的均匀分割位置"""
        # 标准7位车牌比例: 省(1) 字母(1) 间隔(0.3) 5位(5)
        # 总共约 7.3 个单位
        unit = width / 7.5
        gap = unit * 0.3
        
        positions = []
        # 省份
        positions.append((0, int(unit)))
        # 字母
        positions.append((int(unit), int(2 * unit)))
        # 5位字符 (在间隔后)
        start = int(2 * unit + gap)
        for i in range(5):
            x1 = start + int(i * unit)
            x2 = start + int((i + 1) * unit)
            positions.append((x1, x2))
        
        return positions
    
    def _calculate_uniform_positions_8(self, width: int) -> List[Tuple[int, int]]:
        """计算8位新能源车牌的均匀分割位置"""
        # 新能源8位车牌比例: 省(1) 字母(1) D/F(1) 间隔(0.2) 5位(5)
        # 总共约 8.2 个单位
        unit = width / 8.5
        gap = unit * 0.3
        
        positions = []
        # 省份
        positions.append((0, int(unit)))
        # 字母
        positions.append((int(unit), int(2 * unit)))
        # D/F
        positions.append((int(2 * unit), int(3 * unit)))
        # 5位字符 (在间隔后)
        start = int(3 * unit + gap)
        for i in range(5):
            x1 = start + int(i * unit)
            x2 = start + int((i + 1) * unit)
            positions.append((x1, x2))
        
        return positions
    
    def _merge_overlapping_contours(
        self,
        contours: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """合并重叠的轮廓"""
        if len(contours) <= 1:
            return contours
        
        merged = []
        used = [False] * len(contours)
        
        for i, (x1, y1, w1, h1) in enumerate(contours):
            if used[i]:
                continue
            
            # 检查是否与其他轮廓重叠
            merged_x1, merged_y1 = x1, y1
            merged_x2, merged_y2 = x1 + w1, y1 + h1
            
            for j, (x2, y2, w2, h2) in enumerate(contours[i+1:], i+1):
                if used[j]:
                    continue
                
                # 检查水平方向重叠
                overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
                if overlap > min(w1, w2) * 0.5:
                    # 合并
                    merged_x1 = min(merged_x1, x2)
                    merged_y1 = min(merged_y1, y2)
                    merged_x2 = max(merged_x2, x2 + w2)
                    merged_y2 = max(merged_y2, y2 + h2)
                    used[j] = True
            
            merged.append((
                merged_x1,
                merged_y1,
                merged_x2 - merged_x1,
                merged_y2 - merged_y1
            ))
            used[i] = True
        
        return merged
    
    def _find_vertical_bounds(
        self,
        projection: np.ndarray,
        height: int
    ) -> Tuple[int, int]:
        """查找垂直边界"""
        threshold = max(height * 0.05, np.mean(projection) * 0.2)
        
        y_start = 0
        y_end = height
        
        for i in range(height):
            if projection[i] > threshold:
                y_start = max(0, i - 2)
                break
        
        for i in range(height - 1, -1, -1):
            if projection[i] > threshold:
                y_end = min(height, i + 2)
                break
        
        # 确保有效范围
        if y_end <= y_start:
            y_start = 0
            y_end = height
        
        return y_start, y_end
    
    def _filter_regions(
        self,
        regions: List[CharRegion],
        expected_count: int
    ) -> List[CharRegion]:
        """过滤和调整字符区域"""
        if len(regions) == 0:
            return regions
        
        # 如果字符数量多于预期，进行过滤
        if len(regions) > expected_count:
            # 按面积排序，保留最大的几个
            regions.sort(key=lambda r: r.bbox[2] * r.bbox[3], reverse=True)
            regions = regions[:expected_count]
            # 重新按位置排序
            regions.sort(key=lambda r: r.bbox[0])
        
        # 更新索引
        for i, region in enumerate(regions):
            region.index = i
        
        return regions
    
    def _normalize_chars(
        self,
        regions: List[CharRegion]
    ) -> List[CharRegion]:
        """标准化字符大小"""
        for region in regions:
            if region.image is None or region.image.size == 0:
                continue
            
            h, w = region.image.shape[:2]
            if h == 0 or w == 0:
                continue
            
            # 计算缩放比例
            scale = self.target_char_height / h
            new_w = max(1, int(w * scale))
            new_h = self.target_char_height
            
            # 调整大小
            region.image = cv2.resize(
                region.image,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR
            )
        
        return regions
    
    def _set_region_confidence(
        self,
        regions: List[CharRegion],
        quality: ImageQuality
    ) -> None:
        """设置分割区域的置信度"""
        base_confidence = quality.overall_score
        
        for region in regions:
            # 默认置信度基于图像质量
            if region.confidence == 1.0:  # 未设置过
                region.confidence = base_confidence
            else:
                # 已有置信度，与图像质量加权
                region.confidence = region.confidence * 0.5 + base_confidence * 0.5

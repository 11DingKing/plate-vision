"""
字符识别模块
支持PaddleOCR和自定义CRNN模型
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time

from ..utils.logger import LoggerMixin
from ..utils.constants import PROVINCES, LETTERS, DIGITS, CHAR_SET, PlateType
from .char_segmenter import CharSegmenter, CharRegion


@dataclass
class CharResult:
    """单个字符识别结果"""
    char: str               # 识别的字符
    confidence: float       # 置信度
    index: int              # 位置索引
    bbox: Optional[Tuple[int, int, int, int]] = None  # 字符边界框 (x, y, w, h) 相对于车牌图像


@dataclass
class PlateResult:
    """车牌识别结果"""
    plate_number: str       # 车牌号码
    confidence: float       # 整体置信度
    char_results: List[CharResult]  # 各字符结果
    plate_type: str         # 车牌类型
    bbox: Optional[Tuple[int, int, int, int]] = None  # 边界框
    processing_time: float = 0.0  # 处理时间(ms)


class CharRecognizer(LoggerMixin):
    """
    字符识别器
    
    严格遵循 "先分割再识别" 的流程：
    1. 字符分割 - 使用 CharSegmenter 分割字符区域
    2. 字符识别 - 对每个分割出的字符单独识别
    3. 结果验证 - 整合并验证识别结果
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "ch",
        use_angle_cls: bool = False,
        force_segmentation: bool = True,  # 强制使用字符分割
        lightweight_mode: bool = True      # 轻量级模式以优化 CPU 性能
    ):
        """
        初始化字符识别器
        
        Args:
            use_gpu: 是否使用GPU
            lang: 语言
            use_angle_cls: 是否使用角度分类
            force_segmentation: 强制使用字符分割流程（确保符合规范）
            lightweight_mode: 轻量级模式优化 CPU 性能
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.force_segmentation = force_segmentation
        self.lightweight_mode = lightweight_mode
        
        self.ocr = None
        self.segmenter = CharSegmenter()
        
        self._init_ocr()
        self.logger.info(f"字符识别器初始化完成，强制分割: {force_segmentation}, 轻量模式: {lightweight_mode}")
    
    def _init_ocr(self) -> None:
        """初始化OCR引擎"""
        try:
            from paddleocr import PaddleOCR
            
            # 轻量级模式使用更少的资源以优化 CPU 性能
            if self.lightweight_mode:
                self.ocr = PaddleOCR(
                    use_angle_cls=False,  # 禁用角度分类加速
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False,
                    # CPU 优化参数
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    rec_batch_num=1,  # 减少批处理大小
                    det_limit_side_len=640,  # 限制检测尺寸
                    det_limit_type='min',
                    use_space_char=False,  # 禁用空格识别
                )
            else:
                self.ocr = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False,
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    rec_batch_num=6
                )
            
            self.logger.info(f"PaddleOCR引擎初始化成功，轻量模式: {self.lightweight_mode}")
            
        except ImportError:
            self.logger.warning("未安装PaddleOCR，将使用备用方法")
            self.ocr = None
        except Exception as e:
            self.logger.error(f"初始化OCR失败: {e}")
            self.ocr = None
    
    def recognize(
        self,
        plate_image: np.ndarray,
        plate_type: str = PlateType.BLUE,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> PlateResult:
        """
        识别车牌
        
        严格遵循 "先分割再识别" 流程：
        1. 字符分割 - 必经步骤，获取各字符位置
        2. 逐字符识别 - 对每个分割区域进行 OCR
        3. 结果验证和后处理
        
        Args:
            plate_image: 车牌图像
            plate_type: 车牌类型
            bbox: 车牌边界框
            
        Returns:
            识别结果
        """
        start_time = time.time()
        
        # ============ 核心流程：先分割再识别 ============
        # 步骤1：字符分割（必经步骤）
        char_regions = self.segmenter.segment(plate_image, plate_type)
        
        # 步骤2：逐字符识别
        if char_regions and len(char_regions) >= 5:
            # 分割成功，对每个字符单独识别
            result = self._recognize_segmented_chars(
                plate_image, char_regions, plate_type
            )
        else:
            # 分割失败/数量不足，尝试使用 OCR 整体识别后再分割
            self.logger.warning(f"字符分割结果不足: {len(char_regions) if char_regions else 0}，尝试备用方案")
            result = self._recognize_with_fallback(plate_image, plate_type)
        
        # 步骤3：后处理
        result = self._postprocess(result, plate_type)
        
        # 设置额外信息
        result.bbox = bbox
        result.plate_type = plate_type
        result.processing_time = (time.time() - start_time) * 1000
        
        self.logger.info(f"识别结果: {result.plate_number}, 置信度: {result.confidence:.2f}, 耗时: {result.processing_time:.1f}ms")
        
        return result
    
    def _recognize_segmented_chars(
        self,
        plate_image: np.ndarray,
        char_regions: List,
        plate_type: str
    ) -> PlateResult:
        """
        对分割好的字符进行识别
        
        这是标准的 "先分割再识别" 流程
        """
        char_results = []
        plate_number = ""
        
        for region in char_regions:
            # 对每个分割的字符进行识别
            if self.ocr is not None:
                char, conf = self._recognize_single_char_ocr(
                    region.image,
                    region.index,
                    plate_type
                )
            else:
                char, conf = self._recognize_single_char(
                    region.image,
                    region.index,
                    plate_type
                )
            
            # 使用分割时的置信度调整最终置信度
            final_conf = conf * region.confidence if hasattr(region, 'confidence') else conf
            
            plate_number += char
            char_results.append(CharResult(
                char=char,
                confidence=final_conf,
                index=region.index,
                bbox=region.bbox
            ))
        
        # 计算整体置信度（几何平均）
        if char_results:
            confidences = [r.confidence for r in char_results if r.confidence > 0]
            if confidences:
                avg_confidence = np.exp(np.mean(np.log(np.array(confidences) + 1e-10)))
            else:
                avg_confidence = 0.0
        else:
            avg_confidence = 0.0
        
        return PlateResult(
            plate_number=plate_number,
            confidence=avg_confidence,
            char_results=char_results,
            plate_type=plate_type
        )
    
    def _recognize_with_fallback(
        self,
        plate_image: np.ndarray,
        plate_type: str
    ) -> PlateResult:
        """
        备用识别方案：先用 OCR 获取文本，再为每个字符确定位置
        
        即使在这种情况下，也要确保返回字符级的位置信息
        """
        if self.ocr is not None:
            return self._recognize_with_paddleocr(plate_image, plate_type)
        else:
            return self._recognize_with_segmentation(plate_image, plate_type)
    
    def _recognize_with_paddleocr(
        self,
        plate_image: np.ndarray,
        plate_type: str
    ) -> PlateResult:
        """
        使用PaddleOCR识别，结合字符分割获取真实的字符级置信度和位置
        
        流程：
        1. 首先使用字符分割获取各字符的位置信息
        2. 使用PaddleOCR对整体车牌进行识别获取文本
        3. 对每个分割的字符区域单独进行OCR获取真实置信度
        4. 合并结果，返回带有准确位置和置信度的字符结果
        """
        try:
            original_h, original_w = plate_image.shape[:2]
            
            # 调整图像大小以提高识别效果
            if original_w < 200:
                scale = 200 / original_w
                resized_image = cv2.resize(
                    plate_image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )
            else:
                scale = 1.0
                resized_image = plate_image
            
            # 步骤1：使用字符分割获取字符区域
            char_regions = self.segmenter.segment(plate_image, plate_type)
            
            # 步骤2：对整体车牌进行OCR识别
            result = self.ocr.ocr(resized_image, cls=self.use_angle_cls)
            
            if result is None or len(result) == 0 or result[0] is None:
                # 如果整体OCR失败，尝试使用分割方法
                return self._recognize_with_segmentation(plate_image, plate_type)
            
            # 解析整体OCR结果获取车牌文本
            texts = []
            overall_confidences = []
            
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    conf = line[1][1]
                    texts.append(text)
                    overall_confidences.append(conf)
            
            if not texts:
                return self._recognize_with_segmentation(plate_image, plate_type)
            
            # 合并并清理文本
            plate_number = "".join(texts)
            plate_number = self._clean_plate_number(plate_number)
            
            # 步骤3：为每个字符获取真实的置信度
            char_results = []
            
            # 如果字符分割成功且数量匹配，使用分割结果获取真实置信度
            if char_regions and len(char_regions) >= len(plate_number) - 1:
                char_results = self._get_char_confidences_from_regions(
                    plate_image, 
                    plate_number, 
                    char_regions,
                    plate_type
                )
            else:
                # 分割失败时，对每个字符估算位置并单独识别
                char_results = self._estimate_char_positions_and_recognize(
                    plate_image,
                    resized_image,
                    plate_number,
                    scale,
                    plate_type
                )
            
            # 计算整体置信度（各字符置信度的加权平均）
            if char_results:
                # 使用几何平均以更好反映识别质量
                confidences = [cr.confidence for cr in char_results if cr.confidence > 0]
                if confidences:
                    avg_confidence = np.exp(np.mean(np.log(np.array(confidences) + 1e-10)))
                else:
                    avg_confidence = 0.0
            else:
                avg_confidence = sum(overall_confidences) / len(overall_confidences) if overall_confidences else 0.0
            
            return PlateResult(
                plate_number=plate_number,
                confidence=avg_confidence,
                char_results=char_results,
                plate_type=plate_type
            )
            
        except Exception as e:
            self.logger.error(f"PaddleOCR识别失败: {e}")
            return self._recognize_with_segmentation(plate_image, plate_type)
    
    def _get_char_confidences_from_regions(
        self,
        plate_image: np.ndarray,
        plate_number: str,
        char_regions: List,
        plate_type: str
    ) -> List[CharResult]:
        """
        使用分割的字符区域获取真实的字符置信度
        
        对每个分割出的字符单独进行OCR，获取真实的置信度
        """
        char_results = []
        
        # 确保字符区域数量与车牌字符数量匹配
        num_chars = len(plate_number)
        regions_to_use = char_regions[:num_chars] if len(char_regions) >= num_chars else char_regions
        
        for i, char in enumerate(plate_number):
            if i < len(regions_to_use):
                region = regions_to_use[i]
                # 对单个字符进行OCR
                char_conf = self._recognize_single_char_with_ocr(
                    region.image, 
                    char, 
                    i, 
                    plate_type
                )
                # 使用分割得到的bbox
                bbox = region.bbox  # (x, y, w, h)
            else:
                # 没有分割区域，估算位置
                char_conf = 0.5  # 默认置信度
                # 估算字符位置
                h, w = plate_image.shape[:2]
                char_width = w // num_chars
                bbox = (i * char_width, 0, char_width, h)
            
            char_results.append(CharResult(
                char=char,
                confidence=char_conf,
                index=i,
                bbox=bbox
            ))
        
        return char_results
    
    def _recognize_single_char_with_ocr(
        self,
        char_image: np.ndarray,
        expected_char: str,
        index: int,
        plate_type: str
    ) -> float:
        """
        使用OCR对单个字符进行识别，获取真实置信度
        """
        try:
            if char_image is None or char_image.size == 0:
                return 0.5
            
            # 放大字符图像以提高识别准确度
            h, w = char_image.shape[:2]
            if h < 32 or w < 20:
                scale = max(32 / h, 20 / w)
                char_image = cv2.resize(
                    char_image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )
            
            # 添加padding避免字符太靠边
            pad = 5
            char_image = cv2.copyMakeBorder(
                char_image, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255) if len(char_image.shape) == 3 else 255
            )
            
            # 对单个字符进行OCR
            result = self.ocr.ocr(char_image, cls=False)
            
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        recognized_text = line[1][0]
                        confidence = line[1][1]
                        
                        # 如果识别结果包含预期字符，返回置信度
                        if expected_char in recognized_text or recognized_text == expected_char:
                            return float(confidence)
                        
                        # 如果只识别出一个字符且与预期匹配
                        if len(recognized_text) == 1:
                            # 检查是否是常见的混淆字符
                            if self._is_similar_char(recognized_text, expected_char):
                                return float(confidence) * 0.9
                        
                        # 返回最高的置信度（可能识别错误但仍有参考价值）
                        return float(confidence) * 0.7
            
            # 如果OCR失败，根据字符类型给出估算置信度
            return self._estimate_char_confidence(expected_char, index, plate_type)
            
        except Exception as e:
            self.logger.debug(f"单字符OCR失败: {e}")
            return 0.5
    
    def _is_similar_char(self, char1: str, char2: str) -> bool:
        """检查两个字符是否是常见的混淆字符"""
        similar_pairs = [
            ('0', 'O', 'Q', 'D'),
            ('1', 'I', 'L'),
            ('2', 'Z'),
            ('5', 'S'),
            ('6', 'G'),
            ('8', 'B'),
        ]
        for group in similar_pairs:
            if char1 in group and char2 in group:
                return True
        return False
    
    def _estimate_char_confidence(
        self,
        char: str,
        index: int,
        plate_type: str
    ) -> float:
        """根据字符类型和位置估算置信度"""
        # 省份简称通常较难识别
        if index == 0 and char in PROVINCES:
            return 0.7
        # 字母在第二位
        elif index == 1 and char in LETTERS:
            return 0.8
        # 数字通常较容易识别
        elif char in DIGITS:
            return 0.85
        # 其他字母
        elif char in LETTERS:
            return 0.8
        else:
            return 0.6
    
    def _estimate_char_positions_and_recognize(
        self,
        plate_image: np.ndarray,
        resized_image: np.ndarray,
        plate_number: str,
        scale: float,
        plate_type: str
    ) -> List[CharResult]:
        """
        估算字符位置并对每个字符单独识别
        
        当字符分割失败时使用此方法
        """
        char_results = []
        h, w = plate_image.shape[:2]
        num_chars = len(plate_number)
        
        if num_chars == 0:
            return char_results
        
        # 估算字符宽度（考虑第一个字符后可能有间隔点）
        # 对于7位车牌：省+字母+·+5位
        # 对于8位新能源车牌：省+字母+D/F+5位
        
        if plate_type == PlateType.GREEN and num_chars == 8:
            # 新能源车牌的字符宽度估算
            char_width = w / 8.5  # 考虑间隔
            gap_after_second = char_width * 0.3
        else:
            # 普通车牌
            char_width = w / 7.5  # 考虑间隔
            gap_after_second = char_width * 0.4
        
        current_x = 0
        for i, char in enumerate(plate_number):
            # 计算字符位置
            x = int(current_x)
            char_w = int(char_width)
            
            # 边界检查
            if x + char_w > w:
                char_w = w - x
            if char_w <= 0:
                char_w = int(w / num_chars)
                x = min(x, w - char_w)
            
            # 提取字符图像
            char_img = plate_image[:, max(0, x):min(w, x + char_w)]
            
            if char_img.size > 0:
                # 获取真实置信度
                char_conf = self._recognize_single_char_with_ocr(
                    char_img, char, i, plate_type
                )
            else:
                char_conf = 0.5
            
            bbox = (x, 0, char_w, h)
            
            char_results.append(CharResult(
                char=char,
                confidence=char_conf,
                index=i,
                bbox=bbox
            ))
            
            # 更新x位置
            current_x += char_width
            # 第二个字符后添加间隔
            if i == 1:
                current_x += gap_after_second
        
        return char_results
    
    def _recognize_with_segmentation(
        self,
        plate_image: np.ndarray,
        plate_type: str
    ) -> PlateResult:
        """使用字符分割方法识别，返回完整的字符位置信息"""
        try:
            # 分割字符
            char_regions = self.segmenter.segment(plate_image, plate_type)
            
            if not char_regions:
                return self._empty_result()
            
            # 识别每个字符
            char_results = []
            plate_number = ""
            
            for region in char_regions:
                # 如果有 OCR 引擎，尝试用 OCR 识别单个字符
                if self.ocr is not None:
                    char, conf = self._recognize_single_char_ocr(
                        region.image,
                        region.index,
                        plate_type
                    )
                else:
                    char, conf = self._recognize_single_char(
                        region.image,
                        region.index,
                        plate_type
                    )
                
                plate_number += char
                char_results.append(CharResult(
                    char=char,
                    confidence=conf,
                    index=region.index,
                    bbox=region.bbox  # 添加字符边界框
                ))
            
            # 计算总体置信度（使用几何平均）
            if char_results:
                confidences = [r.confidence for r in char_results if r.confidence > 0]
                if confidences:
                    avg_confidence = np.exp(np.mean(np.log(np.array(confidences) + 1e-10)))
                else:
                    avg_confidence = 0.0
            else:
                avg_confidence = 0.0
            
            return PlateResult(
                plate_number=plate_number,
                confidence=avg_confidence,
                char_results=char_results,
                plate_type=plate_type
            )
            
        except Exception as e:
            self.logger.error(f"分割识别失败: {e}")
            return self._empty_result()
    
    def _recognize_single_char_ocr(
        self,
        char_image: np.ndarray,
        index: int,
        plate_type: str
    ) -> Tuple[str, float]:
        """使用OCR识别单个字符并返回置信度"""
        try:
            if char_image is None or char_image.size == 0:
                return "?", 0.0
            
            # 放大字符图像
            h, w = char_image.shape[:2]
            if h < 32 or w < 20:
                scale = max(32 / h, 20 / w)
                char_image = cv2.resize(
                    char_image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )
            
            # 添加padding
            pad = 5
            char_image = cv2.copyMakeBorder(
                char_image, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255) if len(char_image.shape) == 3 else 255
            )
            
            # OCR识别
            result = self.ocr.ocr(char_image, cls=False)
            
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        conf = line[1][1]
                        
                        # 根据位置选择合适的候选字符
                        if index == 0:
                            candidates = PROVINCES
                        elif index == 1:
                            candidates = LETTERS
                        else:
                            candidates = LETTERS + DIGITS
                        
                        # 清理识别结果
                        text = text.strip().upper()
                        if len(text) >= 1:
                            char = text[0]
                            # 验证字符
                            if char in candidates:
                                return char, float(conf)
                            # 尝试修正
                            corrected = self._correct_char(char, index)
                            if corrected in candidates:
                                return corrected, float(conf) * 0.9
                        
                        # 返回第一个字符
                        if len(text) >= 1:
                            return text[0], float(conf) * 0.7
            
            # OCR 失败，使用模板匹配
            return self._recognize_single_char(char_image, index, plate_type)
            
        except Exception as e:
            self.logger.debug(f"单字符OCR失败: {e}")
            return self._recognize_single_char(char_image, index, plate_type)
    
    def _correct_char(self, char: str, index: int) -> str:
        """根据位置修正常见的OCR错误"""
        if index >= 2:  # 数字位置
            corrections = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
        else:  # 字母位置
            corrections = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
        return corrections.get(char, char)
    
    def _recognize_single_char(
        self,
        char_image: np.ndarray,
        index: int,
        plate_type: str
    ) -> Tuple[str, float]:
        """
        使用模板匹配识别单个字符
        
        基于特征提取和模板匹配的真实实现：
        1. 预处理字符图像
        2. 提取HOG特征或使用像素匹配
        3. 与候选字符模板进行匹配
        4. 返回最佳匹配结果
        """
        # 根据位置确定候选字符
        if index == 0:
            candidates = PROVINCES
        elif index == 1:
            candidates = LETTERS
        else:
            if plate_type == PlateType.GREEN:
                candidates = LETTERS + DIGITS
            else:
                candidates = LETTERS + DIGITS
        
        if char_image is None or char_image.size == 0:
            return "?", 0.0
        
        try:
            # 预处理字符图像
            processed = self._preprocess_char_for_matching(char_image)
            
            # 使用模板匹配找最佳字符
            best_char, confidence = self._match_char_template(
                processed, candidates, index
            )
            
            return best_char, confidence
            
        except Exception as e:
            self.logger.warning(f"模板匹配失败: {e}")
            return "?", 0.0
    
    def _preprocess_char_for_matching(self, char_image: np.ndarray) -> np.ndarray:
        """
        预处理字符图像用于模板匹配
        """
        # 转灰度
        if len(char_image.shape) == 3:
            gray = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = char_image.copy()
        
        # 调整大小到标准尺寸
        target_size = (20, 40)  # 宽x高
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
        # 二值化
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 确保字符是白底黑字或黑底白字（统一为白字黑底）
        white_pixels = np.sum(binary > 127)
        total_pixels = binary.size
        if white_pixels > total_pixels / 2:
            binary = 255 - binary  # 反转
        
        return binary
    
    def _match_char_template(
        self,
        char_image: np.ndarray,
        candidates: List[str],
        index: int
    ) -> Tuple[str, float]:
        """
        使用模板匹配识别字符
        
        结合多种特征进行匹配：
        1. 像素直方图匹配
        2. 轮廓特征匹配
        3. 投影特征匹配
        """
        best_char = "?"
        best_score = 0.0
        
        # 提取输入图像特征
        input_features = self._extract_char_features(char_image)
        
        for char in candidates:
            # 生成或获取模板
            template = self._generate_char_template(char, index)
            template_features = self._extract_char_features(template)
            
            # 计算相似度
            similarity = self._compute_feature_similarity(
                input_features, template_features
            )
            
            if similarity > best_score:
                best_score = similarity
                best_char = char
        
        # 转换为置信度 (0-1范围)
        confidence = min(1.0, max(0.0, best_score))
        
        return best_char, confidence
    
    def _extract_char_features(self, char_image: np.ndarray) -> Dict[str, Any]:
        """
        提取字符特征用于匹配
        """
        features = {}
        
        h, w = char_image.shape[:2]
        
        # 1. 水平投影特征
        h_proj = np.sum(char_image, axis=1).astype(float)
        h_proj = h_proj / (np.max(h_proj) + 1e-10)
        features['h_projection'] = h_proj
        
        # 2. 垂直投影特征
        v_proj = np.sum(char_image, axis=0).astype(float)
        v_proj = v_proj / (np.max(v_proj) + 1e-10)
        features['v_projection'] = v_proj
        
        # 3. 区域密度特征 (将图像分成3x3网格)
        grid_h, grid_w = h // 3, w // 3
        densities = []
        for i in range(3):
            for j in range(3):
                region = char_image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                density = np.mean(region) / 255.0
                densities.append(density)
        features['grid_density'] = np.array(densities)
        
        # 4. 轮廓特征
        contours, _ = cv2.findContours(
            char_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            # 主轮廓的Hu矩
            main_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(main_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            # 对数变换以便比较
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            features['hu_moments'] = hu_moments
        else:
            features['hu_moments'] = np.zeros(7)
        
        # 5. 像素统计特征
        features['pixel_ratio'] = np.sum(char_image > 127) / char_image.size
        
        return features
    
    def _generate_char_template(self, char: str, index: int) -> np.ndarray:
        """
        生成字符模板图像
        
        使用OpenCV绘制标准字体字符作为模板
        """
        # 检查缓存
        cache_key = f"{char}_{index}"
        if not hasattr(self, '_template_cache'):
            self._template_cache = {}
        
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        # 创建模板图像
        template_size = (20, 40)  # 宽x高
        template = np.zeros((template_size[1], template_size[0]), dtype=np.uint8)
        
        # 选择字体
        if index == 0:
            # 中文字符使用较大的字体
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
        else:
            # 字母数字
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
        
        # 获取文字大小以居中
        if index == 0 and len(char) == 1 and ord(char) > 127:
            # 中文字符 - 使用PIL绘制
            try:
                from PIL import Image, ImageDraw, ImageFont
                
                pil_img = Image.new('L', template_size, 0)
                draw = ImageDraw.Draw(pil_img)
                
                # 尝试使用系统中文字体
                try:
                    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
                    pil_font = ImageFont.truetype(font_path, 28)
                except:
                    try:
                        font_path = "/System/Library/Fonts/PingFang.ttc"
                        pil_font = ImageFont.truetype(font_path, 28)
                    except:
                        pil_font = ImageFont.load_default()
                
                # 计算文字位置居中
                bbox = draw.textbbox((0, 0), char, font=pil_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                x = (template_size[0] - text_w) // 2
                y = (template_size[1] - text_h) // 2 - bbox[1]
                
                draw.text((x, y), char, fill=255, font=pil_font)
                template = np.array(pil_img)
                
            except ImportError:
                # PIL不可用，使用OpenCV的简单绘制
                cv2.putText(template, "?", (5, 30), font, font_scale, 255, thickness)
        else:
            # 英文字母和数字
            text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
            x = (template_size[0] - text_size[0]) // 2
            y = (template_size[1] + text_size[1]) // 2
            cv2.putText(template, char, (x, y), font, font_scale, 255, thickness)
        
        # 二值化
        _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
        
        # 缓存模板
        self._template_cache[cache_key] = template
        
        return template
    
    def _compute_feature_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """
        计算两组特征的相似度
        """
        scores = []
        weights = []
        
        # 1. 水平投影相似度 (权重: 0.2)
        h_proj_sim = 1.0 - np.mean(np.abs(
            features1['h_projection'] - features2['h_projection']
        ))
        scores.append(h_proj_sim)
        weights.append(0.2)
        
        # 2. 垂直投影相似度 (权重: 0.2)
        v_proj_sim = 1.0 - np.mean(np.abs(
            features1['v_projection'] - features2['v_projection']
        ))
        scores.append(v_proj_sim)
        weights.append(0.2)
        
        # 3. 网格密度相似度 (权重: 0.25)
        grid_sim = 1.0 - np.mean(np.abs(
            features1['grid_density'] - features2['grid_density']
        ))
        scores.append(grid_sim)
        weights.append(0.25)
        
        # 4. Hu矩相似度 (权重: 0.25)
        hu_diff = np.abs(features1['hu_moments'] - features2['hu_moments'])
        hu_sim = 1.0 / (1.0 + np.mean(hu_diff))
        scores.append(hu_sim)
        weights.append(0.25)
        
        # 5. 像素比例相似度 (权重: 0.1)
        pixel_sim = 1.0 - abs(
            features1['pixel_ratio'] - features2['pixel_ratio']
        )
        scores.append(pixel_sim)
        weights.append(0.1)
        
        # 加权平均
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_score
    
    def _clean_plate_number(self, text: str) -> str:
        """清理和规范化车牌号"""
        # 移除空格和特殊字符
        text = text.replace(" ", "").replace("·", "").replace(".", "")
        
        # 转换为大写
        text = text.upper()
        
        # 常见OCR错误修正
        corrections = {
            "0": {"O": "0", "Q": "0", "D": "0"},  # 数字位置的O->0
            "1": {"I": "1", "L": "1"},
            "2": {"Z": "2"},
            "5": {"S": "5"},
            "6": {"G": "6"},
            "8": {"B": "8"},
            "O": {"0": "O", "Q": "O"},  # 字母位置的0->O
        }
        
        # 验证和修正字符
        cleaned = list(text)
        
        for i, char in enumerate(cleaned):
            # 第一位必须是省份简称
            if i == 0 and char not in PROVINCES:
                # 尝试查找最相似的省份
                pass
            
            # 第二位必须是字母
            elif i == 1 and char not in LETTERS:
                if char == "0":
                    cleaned[i] = "O"
                elif char == "1":
                    cleaned[i] = "I"
        
        return "".join(cleaned)
    
    def _postprocess(
        self,
        result: PlateResult,
        plate_type: str
    ) -> PlateResult:
        """后处理识别结果"""
        # 验证车牌格式
        plate_number = result.plate_number
        
        # 检查长度
        expected_len = 8 if plate_type == PlateType.GREEN else 7
        
        if len(plate_number) > expected_len:
            # 截断
            plate_number = plate_number[:expected_len]
        elif len(plate_number) < expected_len:
            # 补齐（使用占位符）
            plate_number = plate_number + "?" * (expected_len - len(plate_number))
        
        result.plate_number = plate_number
        
        # 调整置信度
        if "?" in plate_number:
            result.confidence *= 0.5
        
        return result
    
    def _empty_result(self) -> PlateResult:
        """返回空结果"""
        return PlateResult(
            plate_number="",
            confidence=0.0,
            char_results=[],
            plate_type=PlateType.UNKNOWN
        )
    
    def batch_recognize(
        self,
        plate_images: List[np.ndarray],
        plate_types: Optional[List[str]] = None,
        bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[PlateResult]:
        """
        批量识别车牌
        
        Args:
            plate_images: 车牌图像列表
            plate_types: 车牌类型列表
            bboxes: 边界框列表
            
        Returns:
            识别结果列表
        """
        results = []
        
        if plate_types is None:
            plate_types = [PlateType.BLUE] * len(plate_images)
        if bboxes is None:
            bboxes = [None] * len(plate_images)
        
        for img, ptype, bbox in zip(plate_images, plate_types, bboxes):
            result = self.recognize(img, ptype, bbox)
            results.append(result)
        
        return results

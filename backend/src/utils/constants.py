"""
车牌识别系统常量定义
"""

# 中国省份简称
PROVINCES = [
    "京", "津", "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘",
    "皖", "鲁", "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋",
    "蒙", "陕", "吉", "闽", "贵", "粤", "青", "藏", "川", "宁",
    "琼", "使", "领", "警", "学", "挂"
]

# 车牌字母 (不包含I和O，避免与1和0混淆)
LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z"
]

# 数字
DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 新能源车牌特殊字符
NEW_ENERGY_CHARS = ["D", "F"]

# 特殊车牌字符
SPECIAL_CHARS = ["港", "澳", "学", "警", "挂", "领", "使"]

# 完整字符集 (用于识别)
CHAR_SET = PROVINCES + LETTERS + DIGITS + SPECIAL_CHARS

# 字符到索引的映射
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_SET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_SET)}

# 车牌类型
class PlateType:
    BLUE = "blue"           # 蓝牌 - 普通小型车
    YELLOW = "yellow"       # 黄牌 - 大型车
    GREEN = "green"         # 绿牌 - 新能源
    WHITE = "white"         # 白牌 - 军警车
    BLACK = "black"         # 黑牌 - 外资/涉外
    UNKNOWN = "unknown"     # 未知类型

# 车牌颜色范围 (HSV)
PLATE_COLOR_RANGES = {
    PlateType.BLUE: {
        "lower": [100, 50, 50],
        "upper": [130, 255, 255]
    },
    PlateType.YELLOW: {
        "lower": [15, 70, 70],
        "upper": [35, 255, 255]
    },
    PlateType.GREEN: {
        "lower": [35, 50, 50],
        "upper": [85, 255, 255]
    },
    PlateType.WHITE: {
        "lower": [0, 0, 200],
        "upper": [180, 30, 255]
    },
    PlateType.BLACK: {
        "lower": [0, 0, 0],
        "upper": [180, 255, 50]
    }
}

# 标准车牌尺寸 (mm)
PLATE_SIZES = {
    "standard": (440, 140),      # 标准单层
    "double": (440, 220),        # 双层车牌
    "new_energy": (480, 140),    # 新能源车牌
    "motorcycle": (220, 140)     # 摩托车牌
}

# 车牌宽高比
PLATE_ASPECT_RATIOS = {
    "standard": 3.14,      # 440/140
    "double": 2.0,         # 440/220
    "new_energy": 3.43,    # 480/140
}

# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]

# 支持的视频格式
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"]

# 默认置信度阈值
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# 最大检测数量
MAX_DETECTIONS_PER_FRAME = 10

# 图像预处理参数
IMAGE_PREPROCESS = {
    "target_height": 640,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225]
}

# API响应状态码
class ResponseCode:
    SUCCESS = 0
    IMAGE_LOAD_ERROR = 1001
    NO_PLATE_DETECTED = 1002
    RECOGNITION_ERROR = 1003
    INVALID_FORMAT = 1004
    PROCESSING_TIMEOUT = 1005
    INTERNAL_ERROR = 5000

# 响应消息
RESPONSE_MESSAGES = {
    ResponseCode.SUCCESS: "识别成功",
    ResponseCode.IMAGE_LOAD_ERROR: "图像加载失败",
    ResponseCode.NO_PLATE_DETECTED: "未检测到车牌",
    ResponseCode.RECOGNITION_ERROR: "识别失败",
    ResponseCode.INVALID_FORMAT: "不支持的文件格式",
    ResponseCode.PROCESSING_TIMEOUT: "处理超时",
    ResponseCode.INTERNAL_ERROR: "内部错误"
}

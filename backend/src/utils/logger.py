"""
日志工具模块
"""

import sys
import os
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    """
    配置日志系统
    
    Args:
        log_level: 日志级别
        log_dir: 日志目录
        log_format: 日志格式
        rotation: 日志轮转大小
        retention: 日志保留时间
    """
    # 移除默认处理器
    logger.remove()
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # 添加文件输出 - 常规日志
    logger.add(
        log_path / "lpr_{time:YYYY-MM-DD}.log",
        format=log_format,
        level=log_level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8"
    )
    
    # 添加文件输出 - 错误日志
    logger.add(
        log_path / "lpr_error_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="ERROR",
        rotation=rotation,
        retention=retention,
        encoding="utf-8"
    )
    
    logger.info(f"日志系统初始化完成，日志目录: {log_path.absolute()}")


def get_logger(name: str = None):
    """
    获取日志实例
    
    Args:
        name: 模块名称
        
    Returns:
        logger实例
    """
    if name:
        return logger.bind(name=name)
    return logger


class LoggerMixin:
    """
    日志混入类，为其他类提供日志功能
    """
    
    @property
    def logger(self):
        return logger.bind(name=self.__class__.__name__)

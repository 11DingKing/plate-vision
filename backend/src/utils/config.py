"""
配置管理模块

提供统一的配置读取接口
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional, Dict


class Config:
    """配置管理类"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """加载配置文件"""
        # 查找配置文件路径
        config_paths = [
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
            Path(__file__).parent.parent.parent.parent / "config" / "config.yaml",
            Path("config/config.yaml"),
            Path("config.yaml")
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
                return
        
        # 使用默认配置
        self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "output": {
                "format": "json",
                "save_local": True,
                "output_dir": "output"
            },
            "performance": {
                "cpu_optimization": {
                    "enabled": True,
                    "max_input_size": 1280,
                    "lightweight_detection": True
                }
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，如 "output.save_local"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config()
    return _config

"""
识别结果本地保存模块

支持将识别结果保存为 JSON 或 XML 格式到本地文件
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .logger import get_logger
from .config import get_config

logger = get_logger("output_saver")


class OutputSaver:
    """识别结果保存器"""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        output_format: str = "json",
        enabled: bool = True
    ):
        """
        初始化保存器
        
        Args:
            output_dir: 输出目录，默认从配置文件读取
            output_format: 输出格式，支持 "json" 或 "xml"
            enabled: 是否启用本地保存
        """
        config = get_config()
        
        self.enabled = enabled
        self.output_format = output_format.lower()
        
        # 确定输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(config.get("output.output_dir", "output"))
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录结构
        self.json_dir = self.output_dir / "json"
        self.xml_dir = self.output_dir / "xml"
        self.json_dir.mkdir(exist_ok=True)
        self.xml_dir.mkdir(exist_ok=True)
        
        logger.info(f"OutputSaver 初始化完成，输出目录: {self.output_dir}")
    
    def save(
        self,
        result: Dict[str, Any],
        image_id: str,
        output_format: Optional[str] = None
    ) -> Optional[str]:
        """
        保存识别结果到本地文件
        
        Args:
            result: 识别结果字典
            image_id: 图像ID
            output_format: 输出格式，覆盖默认设置
            
        Returns:
            保存的文件路径，失败返回 None
        """
        if not self.enabled:
            return None
        
        fmt = output_format or self.output_format
        
        try:
            if fmt == "json":
                return self._save_json(result, image_id)
            elif fmt == "xml":
                return self._save_xml(result, image_id)
            else:
                # 同时保存两种格式
                self._save_json(result, image_id)
                return self._save_xml(result, image_id)
                
        except Exception as e:
            logger.error(f"保存识别结果失败: {e}")
            return None
    
    def _save_json(self, result: Dict[str, Any], image_id: str) -> str:
        """保存为 JSON 格式"""
        # 添加元数据
        output_data = {
            "meta": {
                "image_id": image_id,
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "version": "1.0"
            },
            "result": result
        }
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_id}_{timestamp}.json"
        filepath = self.json_dir / filename
        
        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"识别结果已保存: {filepath}")
        return str(filepath)
    
    def _save_xml(self, result: Dict[str, Any], image_id: str) -> str:
        """保存为 XML 格式"""
        # 创建根元素
        root = ET.Element("recognition_result")
        
        # 添加元数据
        meta = ET.SubElement(root, "meta")
        ET.SubElement(meta, "image_id").text = image_id
        ET.SubElement(meta, "timestamp").text = datetime.now().isoformat()
        ET.SubElement(meta, "format").text = "xml"
        ET.SubElement(meta, "version").text = "1.0"
        
        # 添加结果
        result_elem = ET.SubElement(root, "result")
        self._dict_to_xml(result, result_elem)
        
        # 格式化 XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_id}_{timestamp}.xml"
        filepath = self.xml_dir / filename
        
        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(xml_str)
        
        logger.info(f"识别结果已保存: {filepath}")
        return str(filepath)
    
    def _dict_to_xml(self, data: Any, parent: ET.Element) -> None:
        """递归将字典转换为 XML 元素"""
        if isinstance(data, dict):
            for key, value in data.items():
                # XML 标签名不能以数字开头
                tag_name = f"item_{key}" if str(key)[0].isdigit() else str(key)
                child = ET.SubElement(parent, tag_name)
                self._dict_to_xml(value, child)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                child = ET.SubElement(parent, "item")
                child.set("index", str(i))
                self._dict_to_xml(item, child)
        else:
            parent.text = str(data) if data is not None else ""
    
    def get_saved_files(self, limit: int = 100) -> Dict[str, list]:
        """获取已保存的文件列表"""
        json_files = sorted(
            self.json_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:limit]
        
        xml_files = sorted(
            self.xml_dir.glob("*.xml"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:limit]
        
        return {
            "json": [str(f) for f in json_files],
            "xml": [str(f) for f in xml_files]
        }


# 全局单例
_output_saver: Optional[OutputSaver] = None


def get_output_saver() -> OutputSaver:
    """
    获取全局 OutputSaver 实例
    
    默认启用本地保存功能，同时保存 JSON 和 XML 格式
    """
    global _output_saver
    if _output_saver is None:
        config = get_config()
        _output_saver = OutputSaver(
            output_dir=config.get("output.output_dir", "output"),
            output_format=config.get("output.format", "both"),  # 默认同时保存两种格式
            enabled=config.get("output.save_local", True)  # 默认启用
        )
        logger.info(f"本地输出功能已启用，保存目录: {_output_saver.output_dir}")
    return _output_saver


def save_recognition_result(
    result: Dict[str, Any],
    image_id: str,
    output_format: Optional[str] = None
) -> Optional[str]:
    """
    便捷函数：保存识别结果
    
    Args:
        result: 识别结果字典
        image_id: 图像ID
        output_format: 输出格式
        
    Returns:
        保存的文件路径
    """
    return get_output_saver().save(result, image_id, output_format)

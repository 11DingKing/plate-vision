"""
API 接口测试
"""

import pytest
from fastapi.testclient import TestClient
import base64
import cv2
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """健康检查接口测试"""
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data


class TestRecognizeEndpoint:
    """识别接口测试"""
    
    def create_test_image(self) -> bytes:
        """创建测试图像"""
        # 创建一个简单的测试图像
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (200, 200, 200)  # 灰色背景
        
        # 绘制一个模拟车牌区域
        cv2.rectangle(img, (200, 200), (440, 260), (255, 100, 0), -1)  # 蓝色背景
        cv2.putText(img, "PLATE", (220, 245), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # 编码为 JPEG
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    
    def test_recognize_file(self):
        """测试文件上传识别"""
        image_data = self.create_test_image()
        
        response = client.post(
            "/api/v1/recognize/file",
            files={"file": ("test.jpg", image_data, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "code" in data
        assert "data" in data
    
    def test_recognize_base64(self):
        """测试 Base64 识别"""
        image_data = self.create_test_image()
        base64_data = base64.b64encode(image_data).decode()
        
        response = client.post(
            "/api/v1/recognize/base64",
            json={
                "image_data": base64_data,
                "image_id": "test_001"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "code" in data
        assert data["data"]["image_id"] == "test_001"
    
    def test_invalid_format(self):
        """测试无效格式"""
        response = client.post(
            "/api/v1/recognize/file",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        # 应该返回错误
        assert response.status_code == 400 or response.json()["code"] != 0


class TestStatsEndpoint:
    """系统状态接口测试"""
    
    def test_get_stats(self):
        """测试获取系统状态"""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "cpu_usage_percent" in data
        assert "memory_usage_percent" in data
        assert "uptime_seconds" in data


class TestRootEndpoint:
    """根路径测试"""
    
    def test_root(self):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        assert "车牌识别系统" in response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

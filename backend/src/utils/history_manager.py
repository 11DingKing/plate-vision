"""
识别历史记录管理器
使用SQLite存储识别历史记录
"""

import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger("history")

# 线程本地存储，确保每个线程有独立的数据库连接
_local = threading.local()


class HistoryManager:
    """识别历史记录管理器"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化历史记录管理器
        
        Args:
            db_path: 数据库文件路径，默认使用 data/history.db
        """
        if db_path is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "history.db")
        
        self.db_path = db_path
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接"""
        if not hasattr(_local, 'conn') or _local.conn is None:
            _local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            _local.conn.row_factory = sqlite3.Row
        return _local.conn
    
    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 创建识别历史记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                plate_number TEXT NOT NULL,
                confidence REAL NOT NULL,
                plate_type TEXT NOT NULL,
                recognize_time TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引以提高查询性能
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_recognize_time 
            ON recognition_history(recognize_time DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_plate_number 
            ON recognition_history(plate_number)
        ''')
        
        conn.commit()
        logger.info(f"历史记录数据库初始化完成: {self.db_path}")
    
    def add_record(self, image_id: str, plate_number: str, 
                   confidence: float, plate_type: str, 
                   recognize_time: Optional[datetime] = None) -> int:
        """
        添加一条识别记录
        
        Args:
            image_id: 图像ID
            plate_number: 车牌号
            confidence: 置信度
            plate_type: 车牌类型
            recognize_time: 识别时间，默认使用当前时间
            
        Returns:
            新增记录的ID
        """
        if recognize_time is None:
            recognize_time = datetime.now()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO recognition_history 
                (image_id, plate_number, confidence, plate_type, recognize_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                image_id,
                plate_number,
                confidence,
                plate_type,
                recognize_time.isoformat()
            ))
            
            conn.commit()
            record_id = cursor.lastrowid
            logger.debug(f"历史记录已保存: {plate_number}, 置信度: {confidence:.4f}")
            return record_id
            
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")
            conn.rollback()
            raise
    
    def get_records(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """
        分页获取识别记录
        
        Args:
            page: 页码，从1开始
            page_size: 每页大小
            
        Returns:
            包含分页信息和记录列表的字典
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 获取分页数据
        cursor.execute('''
            SELECT id, image_id, plate_number, confidence, plate_type, recognize_time
            FROM recognition_history
            ORDER BY recognize_time DESC
            LIMIT ? OFFSET ?
        ''', (page_size, offset))
        
        rows = cursor.fetchall()
        
        # 获取总记录数
        cursor.execute('SELECT COUNT(*) as total FROM recognition_history')
        total = cursor.fetchone()['total']
        
        # 转换为字典列表
        records = []
        for row in rows:
            records.append({
                'id': row['id'],
                'image_id': row['image_id'],
                'plate_number': row['plate_number'],
                'confidence': round(row['confidence'], 4),
                'plate_type': row['plate_type'],
                'recognize_time': row['recognize_time']
            })
        
        total_pages = (total + page_size - 1) // page_size
        
        return {
            'records': records,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        }
    
    def get_recent_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取最近的识别记录
        
        Args:
            limit: 最大记录数
            
        Returns:
            识别记录列表
        """
        result = self.get_records(page=1, page_size=limit)
        return result['records']
    
    def clear_old_records(self, days: int = 30) -> int:
        """
        清理指定天数之前的记录
        
        Args:
            days: 保留天数
            
        Returns:
            删除的记录数量
        """
        from datetime import timedelta
        
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM recognition_history
            WHERE recognize_time < ?
        ''', (cutoff_time,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"清理了 {deleted_count} 条 {days} 天前的历史记录")
        return deleted_count


# 全局历史记录管理器实例
_history_manager: Optional[HistoryManager] = None


def get_history_manager() -> HistoryManager:
    """获取历史记录管理器单例实例"""
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager()
    return _history_manager

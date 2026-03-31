"""
历史记录管理器
使用 SQLite 存储识别历史
"""
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from .logger import get_logger

logger = get_logger("history_manager")


class HistoryManager:
    """历史记录管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: Optional[str] = None):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        """初始化数据库连接"""
        if hasattr(self, '_initialized'):
            return
        
        if db_path is None:
            # 默认数据库路径
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "history.db")
        
        self.db_path = db_path
        self._local = threading.local()
        self._initialized = True
        
        self._init_database()
        logger.info(f"历史记录管理器初始化完成，数据库路径: {db_path}")
    
    def _get_connection(self):
        """获取线程本地的数据库连接"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    @contextmanager
    def _get_cursor(self):
        """获取游标上下文管理器"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            cursor.close()
    
    def _init_database(self):
        """初始化数据库表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS recognition_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recognition_time DATETIME NOT NULL,
            plate_number TEXT NOT NULL,
            confidence REAL NOT NULL,
            plate_type TEXT NOT NULL,
            image_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_recognition_time ON recognition_history(recognition_time);
        CREATE INDEX IF NOT EXISTS idx_plate_number ON recognition_history(plate_number);
        """
        
        with self._get_cursor() as cursor:
            cursor.executescript(create_table_sql)
    
    def add_record(
        self,
        plate_number: str,
        confidence: float,
        plate_type: str,
        recognition_time: Optional[datetime] = None,
        image_id: Optional[str] = None
    ) -> int:
        """
        添加一条识别记录
        
        Args:
            plate_number: 车牌号
            confidence: 置信度
            plate_type: 车牌类型
            recognition_time: 识别时间，默认为当前时间
            image_id: 图像ID
            
        Returns:
            插入的记录ID
        """
        if recognition_time is None:
            recognition_time = datetime.now()
        
        sql = """
        INSERT INTO recognition_history (
            recognition_time, plate_number, confidence, plate_type, image_id
        ) VALUES (?, ?, ?, ?, ?)
        """
        
        with self._get_cursor() as cursor:
            cursor.execute(sql, (
                recognition_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                plate_number,
                confidence,
                plate_type,
                image_id
            ))
            record_id = cursor.lastrowid
        
        logger.debug(f"添加识别记录: {plate_number}, 置信度: {confidence}")
        return record_id
    
    def add_records(
        self,
        records: List[Dict[str, Any]],
        recognition_time: Optional[datetime] = None,
        image_id: Optional[str] = None
    ) -> List[int]:
        """
        批量添加识别记录
        
        Args:
            records: 记录列表，每个记录包含 plate_number, confidence, plate_type
            recognition_time: 识别时间，默认为当前时间
            image_id: 图像ID
            
        Returns:
            插入的记录ID列表
        """
        if not records:
            return []
        
        if recognition_time is None:
            recognition_time = datetime.now()
        
        time_str = recognition_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        
        sql = """
        INSERT INTO recognition_history (
            recognition_time, plate_number, confidence, plate_type, image_id
        ) VALUES (?, ?, ?, ?, ?)
        """
        
        record_ids = []
        with self._get_cursor() as cursor:
            for record in records:
                cursor.execute(sql, (
                    time_str,
                    record["plate_number"],
                    record["confidence"],
                    record["plate_type"],
                    image_id
                ))
                record_ids.append(cursor.lastrowid)
        
        logger.debug(f"批量添加 {len(records)} 条识别记录")
        return record_ids
    
    def get_history(
        self,
        page: int = 1,
        page_size: int = 20,
        plate_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取识别历史记录
        
        Args:
            page: 页码，从1开始
            page_size: 每页数量
            plate_number: 可选，按车牌号过滤
            
        Returns:
            包含记录列表和分页信息的字典
        """
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        offset = (page - 1) * page_size
        
        # 构建查询条件
        where_clause = "1=1"
        params = []
        if plate_number:
            where_clause += " AND plate_number LIKE ?"
            params.append(f"%{plate_number}%")
        
        # 查询总数
        count_sql = f"SELECT COUNT(*) as total FROM recognition_history WHERE {where_clause}"
        with self._get_cursor() as cursor:
            cursor.execute(count_sql, params)
            total = cursor.fetchone()["total"]
        
        # 查询记录
        select_sql = f"""
        SELECT id, recognition_time, plate_number, confidence, plate_type, image_id, created_at
        FROM recognition_history
        WHERE {where_clause}
        ORDER BY recognition_time DESC
        LIMIT ? OFFSET ?
        """
        params.extend([page_size, offset])
        
        records = []
        with self._get_cursor() as cursor:
            cursor.execute(select_sql, params)
            for row in cursor.fetchall():
                records.append({
                    "id": row["id"],
                    "recognition_time": row["recognition_time"],
                    "plate_number": row["plate_number"],
                    "confidence": row["confidence"],
                    "plate_type": row["plate_type"],
                    "image_id": row["image_id"]
                })
        
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        
        return {
            "records": records,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages
            }
        }
    
    def get_recent(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取最近的识别记录
        
        Args:
            limit: 返回数量
            
        Returns:
            记录列表
        """
        limit = max(1, min(100, limit))
        
        sql = """
        SELECT id, recognition_time, plate_number, confidence, plate_type, image_id
        FROM recognition_history
        ORDER BY recognition_time DESC
        LIMIT ?
        """
        
        records = []
        with self._get_cursor() as cursor:
            cursor.execute(sql, [limit])
            for row in cursor.fetchall():
                records.append({
                    "id": row["id"],
                    "recognition_time": row["recognition_time"],
                    "plate_number": row["plate_number"],
                    "confidence": row["confidence"],
                    "plate_type": row["plate_type"],
                    "image_id": row["image_id"]
                })
        
        return records
    
    def clear_history(self, older_than_days: Optional[int] = None) -> int:
        """
        清除历史记录
        
        Args:
            older_than_days: 只清除指定天数之前的记录，None则清除全部
            
        Returns:
            删除的记录数量
        """
        if older_than_days is not None:
            sql = """
            DELETE FROM recognition_history
            WHERE recognition_time < datetime('now', ?)
            """
            params = [f"-{older_than_days} days"]
        else:
            sql = "DELETE FROM recognition_history"
            params = []
        
        with self._get_cursor() as cursor:
            cursor.execute(sql, params)
            deleted_count = cursor.rowcount
        
        logger.info(f"清除了 {deleted_count} 条历史记录")
        return deleted_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        sql_total = "SELECT COUNT(*) as total FROM recognition_history"
        sql_today = "SELECT COUNT(*) as count FROM recognition_history WHERE date(recognition_time) = date('now')"
        sql_unique_plates = "SELECT COUNT(DISTINCT plate_number) as count FROM recognition_history"
        
        stats = {}
        with self._get_cursor() as cursor:
            cursor.execute(sql_total)
            stats["total_records"] = cursor.fetchone()["total"]
            
            cursor.execute(sql_today)
            stats["today_records"] = cursor.fetchone()["count"]
            
            cursor.execute(sql_unique_plates)
            stats["unique_plates"] = cursor.fetchone()["count"]
        
        return stats


def get_history_manager(db_path: Optional[str] = None) -> HistoryManager:
    """获取历史记录管理器实例"""
    return HistoryManager(db_path)

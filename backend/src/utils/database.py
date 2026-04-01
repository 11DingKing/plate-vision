"""
数据库管理模块
使用SQLite存储识别历史记录
"""

import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from .logger import get_logger

logger = get_logger("database")

# 数据库文件路径
DB_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DB_DIR / "recognition_history.db"

# 确保数据目录存在
DB_DIR.mkdir(parents=True, exist_ok=True)

# 创建数据库引擎
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基础模型类
Base = declarative_base()


class RecognitionRecord(Base):
    """识别历史记录表"""
    __tablename__ = "recognition_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    recognition_time = Column(DateTime, index=True, comment="识别时间")
    plate_number = Column(String(32), index=True, comment="车牌号")
    confidence = Column(Float, comment="置信度")
    plate_type = Column(String(16), comment="车牌类型")
    image_id = Column(String(32), index=True, comment="图像ID")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "recognition_time": self.recognition_time.isoformat(),
            "plate_number": self.plate_number,
            "confidence": round(self.confidence, 4),
            "plate_type": self.plate_type
        }


def init_database():
    """初始化数据库，创建表"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"数据库初始化成功: {DB_PATH}")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


@contextmanager
def get_db_session():
    """获取数据库会话的上下文管理器"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"数据库操作失败: {e}")
        raise
    finally:
        session.close()


def add_recognition_record(
    plate_number: str,
    confidence: float,
    plate_type: str,
    image_id: Optional[str] = None
) -> Optional[RecognitionRecord]:
    """
    添加识别记录

    Args:
        plate_number: 车牌号
        confidence: 置信度
        plate_type: 车牌类型
        image_id: 图像ID

    Returns:
        创建的记录对象
    """
    try:
        with get_db_session() as session:
            record = RecognitionRecord(
                recognition_time=datetime.datetime.now(),
                plate_number=plate_number,
                confidence=confidence,
                plate_type=plate_type,
                image_id=image_id
            )
            session.add(record)
            session.flush()
            logger.debug(f"添加识别记录: {plate_number}, 置信度: {confidence:.4f}")
            return record
    except Exception as e:
        logger.error(f"添加识别记录失败: {e}")
        return None


def get_recognition_history(
    page: int = 1,
    page_size: int = 20
) -> Dict[str, Any]:
    """
    获取识别历史记录（分页）

    Args:
        page: 页码，从1开始
        page_size: 每页数量

    Returns:
        包含分页信息和记录列表的字典
    """
    try:
        with get_db_session() as session:
            # 计算总数
            total = session.query(RecognitionRecord).count()

            # 计算偏移量
            offset = (page - 1) * page_size

            # 查询分页数据，按识别时间倒序
            records = (
                session.query(RecognitionRecord)
                .order_by(RecognitionRecord.recognition_time.desc())
                .offset(offset)
                .limit(page_size)
                .all()
            )

            # 转换为字典列表
            record_list = [record.to_dict() for record in records]

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
                "records": record_list
            }
    except Exception as e:
        logger.error(f"获取识别历史失败: {e}")
        raise


# 初始化数据库
init_database()

import sqlite3
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import json

class DBManager:
    """数据库管理类"""
    
    def __init__(self, db_path: Path = Path("data/cache/stock_cache.db")):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        try:
            # 创建数据库目录
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 连接数据库
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 创建股票列表表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_lists (
                    list_type TEXT PRIMARY KEY,
                    stocks TEXT,
                    update_time TIMESTAMP,
                    params TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"初始化数据库失败: {str(e)}")
            raise
    
    def get_stock_list(self, list_type: str, params: dict = None) -> Optional[List[str]]:
        """获取股票列表
        
        Args:
            list_type: 列表类型
            params: 参数字典
            
        Returns:
            Optional[List[str]]: 股票列表
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 查询数据
            cursor.execute(
                "SELECT stocks, update_time, params FROM stock_lists WHERE list_type = ?",
                (list_type,)
            )
            result = cursor.fetchone()
            
            if result is None:
                return None
            
            stocks, update_time, stored_params = result
            
            # 检查是否需要更新
            if self._need_update(update_time, params, stored_params):
                return None
            
            # 解析股票列表
            return json.loads(stocks)
            
        except Exception as e:
            self.logger.error(f"获取股票列表失败: {str(e)}")
            return None
        finally:
            conn.close()
    
    def save_stock_list(self, list_type: str, stocks: List[str], params: dict = None):
        """保存股票列表
        
        Args:
            list_type: 列表类型
            stocks: 股票列表
            params: 参数字典
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 保存数据
            cursor.execute(
                """
                INSERT OR REPLACE INTO stock_lists 
                (list_type, stocks, update_time, params)
                VALUES (?, ?, ?, ?)
                """,
                (
                    list_type,
                    json.dumps(stocks),
                    datetime.now().isoformat(),
                    json.dumps(params) if params else None
                )
            )
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"保存股票列表失败: {str(e)}")
            raise
        finally:
            conn.close()
    
    def _need_update(self, update_time: str, new_params: dict, stored_params: str) -> bool:
        """检查是否需要更新数据
        
        Args:
            update_time: 更新时间
            new_params: 新参数
            stored_params: 存储的参数
            
        Returns:
            bool: 是否需要更新
        """
        try:
            # 检查更新时间（超过1天需要更新）
            last_update = datetime.fromisoformat(update_time)
            if datetime.now() - last_update > timedelta(days=1):
                return True
            
            # 如果没有新参数，不需要更新
            if not new_params:
                return False
            
            # 解析存储的参数
            stored_params_dict = json.loads(stored_params) if stored_params else {}
            
            # 比较参数
            return new_params != stored_params_dict
            
        except Exception as e:
            self.logger.error(f"检查更新状态失败: {str(e)}")
            return True 
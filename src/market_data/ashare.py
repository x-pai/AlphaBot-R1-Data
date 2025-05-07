from typing import List, Dict, Optional, Any
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from pathlib import Path
import logging
from ..core.config import BaseConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

class AShareDataFetcher:
    """A股数据获取器"""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.max_workers = config.max_workers
        self.chunk_size = config.chunk_size
    
    def get_stock_list(self, list_type: str = "all") -> List[str]:
        """获取股票列表
        
        Args:
            list_type: 列表类型，可选值：all, strong, weak, custom
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            if list_type == "all":
                return ak.stock_zh_a_spot_em()['代码'].tolist()
            elif list_type == "strong":
                # 获取强势股
                df = ak.stock_zh_a_spot_em()
                return df[df['涨跌幅'] > 0]['代码'].tolist()
            elif list_type == "weak":
                # 获取弱势股
                df = ak.stock_zh_a_spot_em()
                return df[df['涨跌幅'] < 0]['代码'].tolist()
            else:
                raise ValueError(f"Unsupported list type: {list_type}")
        except Exception as e:
            self.logger.error(f"Error getting stock list: {str(e)}")
            return []
    
    def _fetch_single_stock_data(self, 
                               stock: str,
                               start_date: str,
                               end_date: str) -> Optional[pd.DataFrame]:
        """获取单个股票的日线数据
        
        Args:
            stock: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Optional[pd.DataFrame]: 日线数据
        """
        try:
            # 这里实现具体的数据获取逻辑
            # 示例代码，需要替换为实际的数据获取实现
            data = pd.DataFrame()
            return data
        except Exception as e:
            self.logger.error(f"获取股票 {stock} 的日线数据时出错: {str(e)}")
            return None
            
    def fetch_daily_data(self,
                        stock: str,
                        start_date: str,
                        end_date: str) -> Optional[pd.DataFrame]:
        """获取日线数据
        
        Args:
            stock: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Optional[pd.DataFrame]: 日线数据
        """
        return self._fetch_single_stock_data(stock, start_date, end_date)
        
    def fetch_batch_daily_data(self,
                             stock_list: List[str],
                             start_date: str,
                             end_date: str) -> Dict[str, pd.DataFrame]:
        """批量获取日线数据
        
        Args:
            stock_list: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码到日线数据的映射
        """
        result = {}
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(
                    self._fetch_single_stock_data,
                    stock,
                    start_date,
                    end_date
                ): stock for stock in stock_list
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    data = future.result()
                    if data is not None:
                        result[stock] = data
                except Exception as e:
                    self.logger.error(f"处理股票 {stock} 的数据时出错: {str(e)}")
                    
        return result
        
    def _fetch_single_fundamental(self, stock: str) -> Optional[Dict[str, Any]]:
        """获取单个股票的基本面数据
        
        Args:
            stock: 股票代码
            
        Returns:
            Optional[Dict[str, Any]]: 基本面数据
        """
        try:
            # 这里实现具体的数据获取逻辑
            # 示例代码，需要替换为实际的数据获取实现
            data = {}
            return data
        except Exception as e:
            self.logger.error(f"获取股票 {stock} 的基本面数据时出错: {str(e)}")
            return None
            
    def fetch_fundamental_data(self, stock: str) -> Optional[Dict[str, Any]]:
        """获取基本面数据
        
        Args:
            stock: 股票代码
            
        Returns:
            Optional[Dict[str, Any]]: 基本面数据
        """
        return self._fetch_single_fundamental(stock)
        
    def fetch_batch_fundamental_data(self, stock_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量获取基本面数据
        
        Args:
            stock_list: 股票代码列表
            
        Returns:
            Dict[str, Dict[str, Any]]: 股票代码到基本面数据的映射
        """
        result = {}
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(
                    self._fetch_single_fundamental,
                    stock
                ): stock for stock in stock_list
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    data = future.result()
                    if data is not None:
                        result[stock] = data
                except Exception as e:
                    self.logger.error(f"处理股票 {stock} 的基本面数据时出错: {str(e)}")
                    
        return result
    
    def fetch_news_data(self, stock: str, lookback_days: int = 5) -> Optional[List[Dict[str, str]]]:
        """获取新闻数据
        
        Args:
            stock: 股票代码
            lookback_days: 回溯天数
            
        Returns:
            Optional[List[Dict[str, str]]]: 新闻数据列表
        """
        try:
            news_list = []
            
            # 获取公司公告
            df = ak.stock_notice_report(symbol=stock)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    news_list.append({
                        'date': row['公告日期'],
                        'title': row['公告标题'],
                        'type': 'announcement'
                    })
            
            # 获取新闻
            df = ak.stock_news_em(symbol=stock)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    news_list.append({
                        'date': row['发布时间'],
                        'title': row['标题'],
                        'type': 'news'
                    })
            
            # 按日期排序并限制数量
            news_list.sort(key=lambda x: x['date'], reverse=True)
            return news_list[:lookback_days]
            
        except Exception as e:
            self.logger.error(f"Error fetching news data for {stock}: {str(e)}")
            return None
    
    def fetch_realtime_data(self, stock: str) -> Optional[Dict[str, Any]]:
        """获取实时数据
        
        Args:
            stock: 股票代码
            
        Returns:
            Optional[Dict[str, Any]]: 实时数据
        """
        try:
            df = ak.stock_zh_a_spot_em()
            stock_data = df[df['代码'] == stock].iloc[0]
            
            return {
                'code': stock_data['代码'],
                'name': stock_data['名称'],
                'price': stock_data['最新价'],
                'change': stock_data['涨跌额'],
                'pct_change': stock_data['涨跌幅'],
                'volume': stock_data['成交量'],
                'amount': stock_data['成交额'],
                'open': stock_data['开盘'],
                'high': stock_data['最高'],
                'low': stock_data['最低'],
                'pre_close': stock_data['昨收'],
                'turnover': stock_data['换手率'],
                'pe': stock_data['市盈率'],
                'pb': stock_data['市净率']
            }
        except Exception as e:
            self.logger.error(f"Error fetching realtime data for {stock}: {str(e)}")
            return None
    
    def save_data(self, df: pd.DataFrame, stock: str, data_type: str = "daily") -> None:
        """保存数据
        
        Args:
            df: 数据框
            stock: 股票代码
            data_type: 数据类型，可选值：daily, realtime
        """
        try:
            # 使用配置中的目录
            if data_type == "daily":
                save_dir = self.config.data_config.daily_dir
            else:
                save_dir = self.config.data_config.processed_dir / data_type
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存数据
            save_path = save_dir / f"{stock}.csv"
            df.to_csv(save_path, index=False)
            self.logger.info(f"Saved {data_type} data for {stock} to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving data for {stock}: {str(e)}") 
from typing import List, Dict, Optional, Any
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from pathlib import Path
import logging
from ..core.config import BaseConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from .db_manager import DBManager

class AShareDataFetcher:
    """A股数据获取器"""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.max_workers = config.max_workers
        self.chunk_size = config.chunk_size
        self.db_manager = DBManager()
    
    def get_stock_list(self, list_type: str = "all", params: dict = None) -> List[str]:
        """获取股票列表
        
        Args:
            list_type: 列表类型，可选值：all, strong, weak, custom
            params: 参数字典，用于缓存判断
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 尝试从缓存获取
            cached_stocks = self.db_manager.get_stock_list(list_type, params)
            if cached_stocks is not None:
                self.logger.info(f"从缓存获取股票列表: {list_type}")
                return cached_stocks
            
            # 缓存未命中，从API获取
            self.logger.info(f"从API获取股票列表: {list_type}")
            if list_type == "all":
                stocks = ak.stock_zh_a_spot_em()['代码'].tolist()
            elif list_type == "strong":
                # 获取强势股
                df = ak.stock_zh_a_spot_em()
                stocks = df[df['涨跌幅'] > 0]['代码'].tolist()
            elif list_type == "weak":
                # 获取弱势股
                df = ak.stock_zh_a_spot_em()
                stocks = df[df['涨跌幅'] < 0]['代码'].tolist()
            else:
                raise ValueError(f"Unsupported list type: {list_type}")
            
            # 保存到缓存
            self.db_manager.save_stock_list(list_type, stocks, params)
            
            return stocks
            
        except Exception as e:
            self.logger.error(f"获取股票列表失败: {str(e)}")
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
            df = ak.stock_zh_a_hist(
                symbol=stock,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'pct_change',
                    '涨跌额': 'change',
                    '换手率': 'turnover'
                })
                
                # 添加股票代码列
                df['stock'] = stock
                
                # 转换日期格式
                df['date'] = pd.to_datetime(df['date'])
                
                return df
            
            return None
            
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
            # 获取单个股票的基本信息
            df = ak.stock_individual_info_em(symbol=stock)
            if df is None or df.empty:
                return None
            
            # 将DataFrame转换为字典，使用item作为key
            info_dict = dict(zip(df['item'], df['value']))
            
            # 获取公司名称
            name = info_dict.get('股票简称', '')
            
            # 获取市值（亿元）
            market_cap = float(info_dict.get('总市值', 0)) / 100000000
            
            # 获取利润增长率（使用最近一期财报数据）
            try:
                profit_df = ak.stock_financial_report_em(symbol=stock, report_type="利润表")
                if not profit_df.empty:
                    # 获取最近两期净利润
                    latest_profit = float(profit_df.iloc[0]['净利润'])
                    prev_profit = float(profit_df.iloc[1]['净利润'])
                    # 计算增长率
                    profit_growth = ((latest_profit - prev_profit) / abs(prev_profit)) * 100 if prev_profit != 0 else 0
                else:
                    profit_growth = 0
            except:
                profit_growth = 0
            
            # 获取市盈率和市净率
            try:
                pe_df = ak.stock_a_lg_indicator(symbol=stock)
                if not pe_df.empty:
                    pe = float(pe_df.iloc[-1]['pe'])
                    pb = float(pe_df.iloc[-1]['pb'])
                else:
                    pe = 0
                    pb = 0
            except:
                pe = 0
                pb = 0
            
            # 计算ROE
            roe = (pb / pe) * 100 if pe > 0 else 0
            
            return {
                'name': name,
                'market_cap': market_cap,
                'pe': pe,
                'pb': pb,
                'roe': roe,
                'profit_growth': profit_growth
            }
            
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
            try:
                df = ak.stock_notice_report(symbol=stock)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        news_list.append({
                            'date': row['公告日期'],
                            'title': row['公告标题'],
                            'type': 'announcement',
                            'source': '公司公告',
                            'url': row.get('公告链接', '')
                        })
            except Exception as e:
                self.logger.warning(f"获取股票 {stock} 的公告数据时出错: {str(e)}")
            
            # 获取新闻
            try:
                df = ak.stock_news_em(symbol=stock)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        news_list.append({
                            'date': row['发布时间'],
                            'title': row['标题'],
                            'type': 'news',
                            'source': row.get('来源', '东方财富'),
                            'url': row.get('链接', '')
                        })
            except Exception as e:
                self.logger.warning(f"获取股票 {stock} 的新闻数据时出错: {str(e)}")
            
            # 获取研报
            try:
                df = ak.stock_research_report_em(symbol=stock)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        news_list.append({
                            'date': row['发布时间'],
                            'title': row['标题'],
                            'type': 'research',
                            'source': row.get('机构', '未知机构'),
                            'url': row.get('链接', '')
                        })
            except Exception as e:
                self.logger.warning(f"获取股票 {stock} 的研报数据时出错: {str(e)}")
            
            # 获取互动易
            try:
                df = ak.stock_zh_a_alerts_cls(symbol=stock)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        news_list.append({
                            'date': row['时间'],
                            'title': row['内容'],
                            'type': 'interaction',
                            'source': '互动易',
                            'url': row.get('链接', '')
                        })
            except Exception as e:
                self.logger.warning(f"获取股票 {stock} 的互动易数据时出错: {str(e)}")
            
            # 按日期排序并限制数量
            news_list.sort(key=lambda x: x['date'], reverse=True)
            
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # 过滤日期范围内的新闻
            filtered_news = []
            for news in news_list:
                try:
                    news_date = datetime.strptime(news['date'], '%Y-%m-%d')
                    if start_date <= news_date <= end_date:
                        filtered_news.append(news)
                except ValueError:
                    continue
            
            return filtered_news
            
        except Exception as e:
            self.logger.error(f"获取股票 {stock} 的新闻数据时出错: {str(e)}")
            return None
            
    def fetch_batch_news_data(self, 
                            stock_list: List[str],
                            lookback_days: int = 5) -> Dict[str, List[Dict[str, str]]]:
        """批量获取新闻数据
        
        Args:
            stock_list: 股票代码列表
            lookback_days: 回溯天数
            
        Returns:
            Dict[str, List[Dict[str, str]]]: 股票代码到新闻数据的映射
        """
        result = {}
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(
                    self.fetch_news_data,
                    stock,
                    lookback_days
                ): stock for stock in stock_list
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    news_list = future.result()
                    if news_list is not None:
                        result[stock] = news_list
                except Exception as e:
                    self.logger.error(f"处理股票 {stock} 的新闻数据时出错: {str(e)}")
                    
        return result
        
    def analyze_news_sentiment(self, news_list: List[Dict[str, str]]) -> Dict[str, float]:
        """分析新闻情感
        
        Args:
            news_list: 新闻列表
            
        Returns:
            Dict[str, float]: 情感分析结果
        """
        try:
            if not news_list:
                return {
                    'positive': 0.0,
                    'neutral': 0.0,
                    'negative': 0.0,
                    'sentiment_score': 0.0
                }
            
            # 初始化计数器
            sentiment_counts = {
                'positive': 0,
                'neutral': 0,
                'negative': 0
            }
            
            # 情感词典
            positive_words = {'利好', '增长', '上涨', '突破', '创新高', '盈利', '增长', '扩张', '合作', '中标'}
            negative_words = {'利空', '下跌', '亏损', '风险', '警示', '违规', '处罚', '诉讼', '减持', '质押'}
            
            # 分析每条新闻
            for news in news_list:
                title = news['title'].lower()
                
                # 计算情感得分
                positive_count = sum(1 for word in positive_words if word in title)
                negative_count = sum(1 for word in negative_words if word in title)
                
                if positive_count > negative_count:
                    sentiment_counts['positive'] += 1
                elif negative_count > positive_count:
                    sentiment_counts['negative'] += 1
                else:
                    sentiment_counts['neutral'] += 1
            
            # 计算情感比例
            total = len(news_list)
            sentiment = {
                'positive': sentiment_counts['positive'] / total,
                'neutral': sentiment_counts['neutral'] / total,
                'negative': sentiment_counts['negative'] / total
            }
            
            # 计算情感得分
            sentiment['sentiment_score'] = (
                sentiment['positive'] * 1.0 +
                sentiment['neutral'] * 0.0 +
                sentiment['negative'] * -1.0
            )
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"分析新闻情感时出错: {str(e)}")
            return {
                'positive': 0.0,
                'neutral': 0.0,
                'negative': 0.0,
                'sentiment_score': 0.0
            }
    
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
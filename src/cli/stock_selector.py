import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..core.config import BaseConfig
from ..market_data.ashare import AShareDataFetcher
from ..indicators.technical import TechnicalIndicators

class StockSelector:
    """股票选择器"""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = AShareDataFetcher(config)
        self.technical = TechnicalIndicators(config.analysis.technical_config)
        self.max_workers = config.max_workers  # 最大线程数
        
    def _process_single_stock(self, 
                            stock: str,
                            start_date: str,
                            end_date: str,
                            min_price: float,
                            max_price: float,
                            min_volume: float,
                            min_market_cap: float,
                            max_pe: float,
                            min_roe: float,
                            min_profit_growth: float) -> Dict[str, Any]:
        """处理单个股票
        
        Args:
            stock: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            min_price: 最低价格
            max_price: 最高价格
            min_volume: 最小成交量
            min_market_cap: 最小市值
            max_pe: 最大市盈率
            min_roe: 最小净资产收益率
            min_profit_growth: 最小利润增长率
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 仅保留代码
            stock = stock.split(".")[0]

            # 获取日线数据
            daily_data = self.data_fetcher.fetch_daily_data(
                stock=stock,
                start_date=start_date,
                end_date=end_date
            )

            # 检查数据是否足够 
            if daily_data is None or len(daily_data) < 200: # 200天数据用于计算技术指标
                return {'stock': stock, 'status': 'no_data'}
                
            # 获取基本面数据
            # fundamental_data = self.data_fetcher.fetch_fundamental_data(stock)
            # if fundamental_data is None:
            #     return {'stock': stock, 'status': 'no_fundamental'}
                
            # 计算技术指标
            daily_data = self.technical.calculate_all(daily_data)
            
            # 获取最新数据
            latest = daily_data.iloc[-1]
            
            # 基本面筛选
            # if latest['close'] < min_price or latest['close'] > max_price:
            #     return {'stock': stock, 'status': 'price_filter'}
            # if latest['volume'] < min_volume:
            #     return {'stock': stock, 'status': 'volume_filter'}
            # if fundamental_data['market_cap'] < min_market_cap:
            #     return {'stock': stock, 'status': 'market_cap_filter'}
            # if fundamental_data['pe'] > max_pe:
            #     return {'stock': stock, 'status': 'pe_filter'}
            # if fundamental_data['roe'] < min_roe:
            #     return {'stock': stock, 'status': 'roe_filter'}
            # if fundamental_data['profit_growth'] < min_profit_growth:
            #     return {'stock': stock, 'status': 'profit_growth_filter'}
            
            # 技术面筛选
            if not self._check_technical_indicators(daily_data):
                return {'stock': stock, 'status': 'technical_filter'}
            
            # 计算得分
            technical_score = self._calculate_technical_score(daily_data)
            # fundamental_score = self._calculate_fundamental_score(fundamental_data)
            
            return {
                'stock': stock,
                'status': 'selected',
                'data': {
                    # 'name': fundamental_data['name'],
                    'price': latest['close'],
                    'change': latest['close'] / daily_data.iloc[-2]['close'] - 1,
                    'volume': latest['volume'],
                    'amount': latest['amount'],
                    # 'market_cap': fundamental_data['market_cap'],
                    # 'pe': fundamental_data['pe'],
                    # 'roe': fundamental_data['roe'],
                    # 'profit_growth': fundamental_data['profit_growth'],
                    'technical_score': technical_score,
                    # 'fundamental_score': fundamental_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"处理股票 {stock} 时出错: {str(e)}")
            return {'stock': stock, 'status': 'error'}
        
    def select_stocks(self,
                     date: str,
                     min_price: float = 5.0,
                     max_price: float = 100.0,
                     min_volume: float = 1000000,
                     min_market_cap: float = 1000000000,
                     max_pe: float = 50.0,
                     min_roe: float = 10.0,
                     min_profit_growth: float = 20.0) -> List[Dict[str, Any]]:
        """选择股票
        
        Args:
            date: 选股日期
            min_price: 最低价格
            max_price: 最高价格
            min_volume: 最小成交量
            min_market_cap: 最小市值（亿元）
            max_pe: 最大市盈率
            min_roe: 最小净资产收益率
            min_profit_growth: 最小利润增长率
            
        Returns:
            List[Dict[str, Any]]: 选中的股票列表
        """
        try:
            # 获取股票列表
            self.logger.info("开始获取股票列表...")
            stock_list = self.data_fetcher.get_stock_list(list_type="strong")
            self.logger.info(f"获取到 {len(stock_list)} 只股票")
            stock_list = stock_list[:100]
            
            # 计算开始日期（获取30天数据用于计算技术指标）
            start_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
            self.logger.info(f"获取 {start_date} 至 {date} 的数据用于计算技术指标")
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_stock = {
                    executor.submit(
                        self._process_single_stock,
                        stock=stock,
                        start_date=start_date,
                        end_date=date,
                        min_price=min_price,
                        max_price=max_price,
                        min_volume=min_volume,
                        min_market_cap=min_market_cap,
                        max_pe=max_pe,
                        min_roe=min_roe,
                        min_profit_growth=min_profit_growth
                    ): stock for stock in stock_list
                }
                
                # 初始化统计信息
                skipped_stocks = {
                    'no_data': 0,
                    'no_fundamental': 0,
                    'price_filter': 0,
                    'volume_filter': 0,
                    'market_cap_filter': 0,
                    'pe_filter': 0,
                    'roe_filter': 0,
                    'profit_growth_filter': 0,
                    'technical_filter': 0,
                    'error': 0
                }
                selected_stocks = []
                
                # 使用tqdm显示进度
                with tqdm(total=len(stock_list), desc="处理股票", unit="只") as pbar:
                    # 处理完成的任务
                    for future in as_completed(future_to_stock):
                        result = future.result()
                        status = result['status']
                        
                        if status == 'selected':
                            selected_stocks.append({
                                'stock': result['stock'],
                                **result['data']
                            })
                        else:
                            skipped_stocks[status] += 1
                            
                        pbar.update(1)
            
            # 输出统计信息
            self.logger.info("选股结果统计：")
            self.logger.info(f"总股票数: {len(stock_list)}")
            self.logger.info(f"选中股票数: {len(selected_stocks)}")
            self.logger.info("筛选原因统计：")
            for status, count in skipped_stocks.items():
                self.logger.info(f"{status}: {count}")
            
            # 按技术面得分和基本面得分的加权和排序
            selected_stocks.sort(
                key=lambda x: x['technical_score'] * 0.6 + x['fundamental_score'] * 0.4,
                reverse=True
            )
            
            return selected_stocks
            
        except Exception as e:
            self.logger.error(f"选股过程出错: {str(e)}")
            return []
    
    def _check_technical_indicators(self, df: pd.DataFrame) -> bool:
        """检查技术指标
        
        Args:
            df: 日线数据
            
        Returns:
            bool: 是否通过技术指标筛选
        """
        latest = df.iloc[-1]
        stock = latest['stock']
        
        # 检查200日均线
        if 'ma200' not in latest:
            self.logger.info(f"股票 {stock} 未通过技术指标筛选: 缺少200日均线数据")
            return False
            
        # 检查是否突破200日均线
        # 1. 当前价格在均线上方
        # 2. 最近3天至少有2天收盘价在均线上方
        # 3. 突破幅度至少1%
        price_above_ma = latest['close'] > latest['ma200']
        recent_days_above = sum(1 for i in range(-3, 0) if df.iloc[i]['close'] > df.iloc[i]['ma200'])
        break_through_pct = (latest['close'] - latest['ma200']) / latest['ma200'] * 100
        
        if not (price_above_ma and recent_days_above >= 2 and break_through_pct >= 1):
            self.logger.info(f"股票 {stock} 未通过技术指标筛选: 未有效突破200日均线 "
                           f"(当前价格:{latest['close']:.2f}, 均线:{latest['ma200']:.2f}, "
                           f"突破幅度:{break_through_pct:.2f}%, 最近3天均线上方天数:{recent_days_above})")
            return False
            
        # 检查布林带
        # 1. 价格在布林带范围内
        # 2. 价格回踩上轨
        price_in_band = latest['boll_lower'] < latest['close'] < latest['boll_upper']
        price_to_upper = (latest['boll_upper'] - latest['close']) / latest['close'] * 100
        
        if not (price_in_band and 0 < price_to_upper < 2):  # 价格回踩上轨，距离上轨不超过2%
            self.logger.info(f"股票 {stock} 未通过技术指标筛选: 布林带条件不满足 "
                           f"(价格:{latest['close']:.2f}, 上轨:{latest['boll_upper']:.2f}, "
                           f"距离上轨:{price_to_upper:.2f}%)")
            return False
            
        # 检查均线系统
        if not (latest['ma5'] > latest['ma10'] > latest['ma20']):
            self.logger.info(f"股票 {stock} 未通过技术指标筛选: 均线系统未形成多头排列")
            return False
            
        # 检查MACD
        if not (latest['macd'] > 0 and latest['macd'] > latest['macd_signal']):
            self.logger.info(f"股票 {stock} 未通过技术指标筛选: MACD未形成金叉或位于零轴下方")
            return False
            
        # 检查RSI
        if not (30 < latest['rsi'] < 70):
            self.logger.info(f"股票 {stock} 未通过技术指标筛选: RSI不在合理区间 ({latest['rsi']:.2f})")
            return False
            
        # 检查KDJ
        if not (latest['kdj_k'] > latest['kdj_d']):
            self.logger.info(f"股票 {stock} 未通过技术指标筛选: KDJ未形成金叉")
            return False
            
        self.logger.info(f"股票 {stock} 通过所有技术指标筛选")
        return True
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """计算技术面得分
        
        Args:
            df: 日线数据
            
        Returns:
            float: 技术面得分
        """
        latest = df.iloc[-1]
        score = 0.0
        
        # 200日均线突破得分
        if latest['close'] > latest['ma200'] and df.iloc[-2]['close'] <= df.iloc[-2]['ma200']:
            ma200_change = (latest['close'] - latest['ma200']) / latest['ma200'] * 100
            if ma200_change > 5:
                score += 40
            elif ma200_change > 3:
                score += 30
            elif ma200_change > 0:
                score += 20
        
        # 布林带得分
        boll_position = (latest['close'] - latest['boll_lower']) / (latest['boll_upper'] - latest['boll_lower'])
        upper_space = (latest['boll_upper'] - latest['close']) / latest['close'] * 100
        
        # 根据布林带位置和上升空间评分
        if 0.2 < boll_position < 0.4:  # 价格在布林带中下轨之间
            if upper_space > 10:
                score += 40
            elif upper_space > 7:
                score += 30
            elif upper_space > 5:
                score += 20
        elif 0.4 < boll_position < 0.6:  # 价格在布林带中轨附近
            if upper_space > 10:
                score += 30
            elif upper_space > 7:
                score += 20
            elif upper_space > 5:
                score += 10
            
        # 均线系统得分
        if latest['ma5'] > latest['ma10'] > latest['ma20']:
            score += 20
        elif latest['ma5'] > latest['ma10']:
            score += 10
            
        # MACD得分
        if latest['macd'] > 0 and latest['macd'] > latest['macd_signal']:
            score += 10
            
        # RSI得分
        if 40 < latest['rsi'] < 60:
            score += 10
            
        # KDJ得分
        if latest['kdj_k'] > latest['kdj_d']:
            score += 10
            
        # 成交量得分
        if latest['volume'] > df['volume'].mean() * 1.5:
            score += 10
            
        return score
    
    def _calculate_fundamental_score(self, data: Dict[str, float]) -> float:
        """计算基本面得分
        
        Args:
            data: 基本面数据
            
        Returns:
            float: 基本面得分
        """
        score = 0.0
        
        # PE得分
        if 0 < data['pe'] < 20:
            score += 30
        elif 0 < data['pe'] < 30:
            score += 20
        elif 0 < data['pe'] < 40:
            score += 10
            
        # ROE得分
        if data['roe'] > 20:
            score += 30
        elif data['roe'] > 15:
            score += 20
        elif data['roe'] > 10:
            score += 10
            
        # 利润增长得分
        if data['profit_growth'] > 50:
            score += 40
        elif data['profit_growth'] > 30:
            score += 30
        elif data['profit_growth'] > 20:
            score += 20
            
        return score

def setup_logging(log_dir: Path) -> logging.Logger:
    """设置日志
    
    Args:
        log_dir: 日志目录
        
    Returns:
        logging.Logger: 日志记录器
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stock_selector_{datetime.now().strftime('%Y%m%d')}.log"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_args() -> argparse.Namespace:
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='每日选股工具')
    
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y%m%d'),
                      help='选股日期，格式：YYYYMMDD')
    parser.add_argument('--min-price', type=float, default=5.0,
                      help='最低价格')
    parser.add_argument('--max-price', type=float, default=100.0,
                      help='最高价格')
    parser.add_argument('--min-volume', type=float, default=1000000,
                      help='最小成交量')
    parser.add_argument('--min-market-cap', type=float, default=1000000000,
                      help='最小市值（亿元）')
    parser.add_argument('--max-pe', type=float, default=50.0,
                      help='最大市盈率')
    parser.add_argument('--min-roe', type=float, default=10.0,
                      help='最小净资产收益率')
    parser.add_argument('--min-profit-growth', type=float, default=20.0,
                      help='最小利润增长率')
    parser.add_argument('--output-dir', type=str, default='output/stock_selection',
                      help='输出目录')
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='日志目录')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging(Path(args.log_dir))
    
    try:
        # 创建配置
        config = BaseConfig()
        
        # 创建选股器
        selector = StockSelector(config)
        
        # 选择股票
        selected_stocks = selector.select_stocks(
            date=args.date,
            min_price=args.min_price,
            max_price=args.max_price,
            min_volume=args.min_volume,
            min_market_cap=args.min_market_cap,
            max_pe=args.max_pe,
            min_roe=args.min_roe,
            min_profit_growth=args.min_profit_growth
        )
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        output_file = output_dir / f"selected_stocks_{args.date}.csv"
        df = pd.DataFrame(selected_stocks)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"Selected {len(selected_stocks)} stocks")
        logger.info(f"Results saved to {output_file}")
        
        # 打印结果
        print("\n选股结果：")
        print("=" * 100)
        print(f"{'股票代码':<10}{'股票名称':<10}{'现价':<8}{'涨跌幅':<8}{'技术得分':<8}{'基本面得分':<8}{'综合得分':<8}")
        print("-" * 100)
        
        for stock in selected_stocks:
            total_score = stock['technical_score'] + stock['fundamental_score']
            print(f"{stock['stock']:<10}{stock['name']:<10}{stock['price']:<8.2f}{stock['change']*100:<8.2f}%{stock['technical_score']:<8.1f}{stock['fundamental_score']:<8.1f}{total_score:<8.1f}")
        
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
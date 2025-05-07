import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
from ..core.config import BaseConfig, DataSourceConfig, MarketConfig, DataConfig
from ..market_data.ashare import AShareDataFetcher
from ..indicators.technical import TechnicalIndicators, IndicatorConfig
from ..models.sft_generator import SFTGenerator

def setup_logging(log_dir: Path) -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger("alphabot")
    logger.setLevel(logging.INFO)
    
    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件处理器
    log_file = log_dir / f"alphabot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AlphaBot Data Processing Tool')
    
    # 数据源参数
    parser.add_argument('--data-source', type=str, default='akshare',
                      help='数据源名称 (默认: akshare)')
    
    # 日期参数
    parser.add_argument('--start-date', type=str,
                      default=(datetime.now() - timedelta(days=365)).strftime('%Y%m%d'),
                      help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', type=str,
                      default=datetime.now().strftime('%Y%m%d'),
                      help='结束日期 (YYYYMMDD)')
    
    # 股票列表参数
    parser.add_argument('--stock-list-type', type=str,
                      choices=['all', 'strong', 'weak', 'custom'],
                      default='all',
                      help='股票列表类型')
    parser.add_argument('--custom-stocks', type=str, nargs='+',
                      help='自定义股票列表 (空格分隔)')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str,
                      default='data/processed',
                      help='输出目录')
    parser.add_argument('--log-dir', type=str,
                      default='logs',
                      help='日志目录')
    
    # 技术指标参数
    parser.add_argument('--indicators', type=str, nargs='+',
                      default=['ma', 'ema', 'macd', 'rsi', 'kdj', 'boll'],
                      help='要计算的技术指标')
    
    # SFT数据生成参数
    parser.add_argument('--generate-sft', action='store_true',
                      help='是否生成SFT数据')
    parser.add_argument('--sft-samples', type=int, default=10,
                      help='每只股票生成的SFT样本数量')
    parser.add_argument('--sft-lookback', type=int, default=5,
                      help='SFT样本的回溯天数')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置
    config = BaseConfig(
        data_source=DataSourceConfig(name=args.data_source),
        market=MarketConfig(market="CN", exchange="SSE"),
        data_config=DataConfig(
            processed_dir=Path(args.output_dir)
        ),
        log_dir=Path(args.log_dir)
    )
    
    # 设置日志
    logger = setup_logging(config.log_dir)
    logger.info("Starting AlphaBot Data Processing")
    
    try:
        # 创建数据获取器
        fetcher = AShareDataFetcher(config)
        
        # 获取股票列表
        if args.stock_list_type == 'custom' and args.custom_stocks:
            stocks = args.custom_stocks
        else:
            stocks = fetcher.get_stock_list(args.stock_list_type)
        
        logger.info(f"Processing {len(stocks)} stocks")
        
        # 创建技术指标计算器
        indicator_config = IndicatorConfig()
        indicators = TechnicalIndicators(indicator_config)
        
        # 创建SFT生成器
        sft_generator = SFTGenerator(config)
        
        # 处理每只股票
        for stock in stocks:
            try:
                # 获取日线数据
                df = fetcher.fetch_daily_data(
                    stock=stock,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
                
                if df is not None and not df.empty:
                    # 计算技术指标
                    df = indicators.calculate_all(df)
                    
                    # 保存原始数据
                    fetcher.save_data(df, stock, "daily")
                    logger.info(f"Successfully processed {stock}")
                    
                    # 生成SFT数据
                    if args.generate_sft:
                        samples = sft_generator.generate_samples(
                            df=df,
                            stock=stock,
                            num_samples=args.sft_samples,
                            lookback_days=args.sft_lookback
                        )
                        
                        if samples:
                            sft_generator.save_samples(samples)
                            logger.info(f"Generated {len(samples)} SFT samples for {stock}")
                else:
                    logger.warning(f"No data available for {stock}")
            
            except Exception as e:
                logger.error(f"Error processing {stock}: {str(e)}")
                continue
        
        logger.info("Data processing completed")
    
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
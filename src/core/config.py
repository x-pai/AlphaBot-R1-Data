from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import os

@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 5

@dataclass
class MarketConfig:
    """市场配置"""
    market: str
    exchange: str
    timezone: str = "Asia/Shanghai"
    trading_hours: Dict[str, List[str]] = field(default_factory=lambda: {
        "morning": ["09:30", "11:30"],
        "afternoon": ["13:00", "15:00"]
    })

@dataclass
class DataConfig:
    """数据配置"""
    base_dir: Path = Path(".")
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    daily_dir: Path = field(init=False)
    sft_dir: Path = field(init=False)
    train_dir: Path = field(init=False)
    val_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        # 原始数据目录
        self.raw_dir = self.base_dir / "data/raw"
        # 处理后的数据目录
        self.processed_dir = self.base_dir / "data/processed"
        # 日线数据目录
        self.daily_dir = self.base_dir / "data/processed/daily"
        # SFT数据目录
        self.sft_dir = self.base_dir / "data/processed/sft"
        # 训练集目录
        self.train_dir = self.base_dir / "data/processed/sft/train"
        # 验证集目录
        self.val_dir = self.base_dir / "data/processed/sft/val"
        # 测试集目录
        self.test_dir = self.base_dir / "data/processed/sft/test"

@dataclass
class OpenAIConfig:
    """OpenAI配置"""
    api_key: str
    base_url: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 5

@dataclass
class TechnicalConfig:
    """技术指标配置"""
    # 移动平均线周期
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 200])
    
    # MACD参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # RSI参数
    rsi_period: int = 14
    
    # KDJ参数
    kdj_k: int = 9
    kdj_d: int = 3
    kdj_j: int = 3
    
    # 布林带参数
    boll_period: int = 25
    boll_std: float = 2.0

@dataclass
class AnalysisConfig:
    """分析配置"""
    # 基本面分析
    fundamental_enabled: bool = True
    fundamental_metrics: List[str] = field(default_factory=lambda: [
        'pe', 'pb', 'roe', 'roa', 'gross_margin', 'net_margin',
        'debt_ratio', 'current_ratio', 'quick_ratio'
    ])
    
    # 技术面分析
    technical_enabled: bool = True
    technical_config: TechnicalConfig = field(default_factory=TechnicalConfig)
    technical_indicators: List[str] = field(default_factory=lambda: [
        'ma', 'ema', 'macd', 'rsi', 'kdj', 'boll',
        'volume_ma', 'obv', 'cci', 'dmi'
    ])
    
    # 消息面分析
    news_enabled: bool = True
    news_sources: List[str] = field(default_factory=lambda: [
        'announcement', 'news', 'social_media'
    ])
    news_lookback_days: int = 5
    
    # 历史数据分析
    history_enabled: bool = True
    history_days: int = 30
    history_metrics: List[str] = field(default_factory=lambda: [
        'price', 'volume', 'amount', 'turnover',
        'ma5', 'ma10', 'ma20', 'ma200',
        'vol_ma5', 'vol_ma10'
    ])

@dataclass
class BaseConfig:
    """基础配置类"""
    # 基础配置
    project_name: str = "AlphaBot-Data"
    version: str = "1.0.0"
    debug: bool = False
    
    # 路径配置
    base_dir: Path = Path(".")
    data_config: DataConfig = field(default_factory=lambda: DataConfig(base_dir=Path(".")))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    # 数据源配置
    data_source: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(name="akshare"))
    
    # OpenAI配置
    openai: Optional[OpenAIConfig] = None
    
    # 市场配置
    market: MarketConfig = field(default_factory=lambda: MarketConfig(
        market="CN",
        exchange="SSE"
    ))
    
    # 处理配置
    batch_size: int = 50  # 批处理大小
    max_workers: int = 4  # 最大线程数
    chunk_size: int = 100  # 数据分块大小
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 缓存时间（秒）
    
    # 分析配置
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        # 更新data_config的base_dir
        self.data_config.base_dir = self.base_dir
        
        # 从环境变量加载处理配置
        self.max_workers = int(os.getenv('MAX_WORKERS', self.max_workers))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', self.chunk_size))
        self.batch_size = int(os.getenv('BATCH_SIZE', self.batch_size))
        
        # 从环境变量加载分析配置
        self.analysis.fundamental_enabled = os.getenv('ANALYSIS_FUNDAMENTAL_ENABLED', 'true').lower() == 'true'
        self.analysis.technical_enabled = os.getenv('ANALYSIS_TECHNICAL_ENABLED', 'true').lower() == 'true'
        self.analysis.news_enabled = os.getenv('ANALYSIS_NEWS_ENABLED', 'true').lower() == 'true'
        self.analysis.news_lookback_days = int(os.getenv('ANALYSIS_NEWS_LOOKBACK_DAYS', '5'))
        
        # 加载技术指标配置
        ma_periods_str = os.getenv('MA_PERIODS', '5,10,20,200')
        self.analysis.technical_config.ma_periods = [int(x) for x in ma_periods_str.split(',')]
        self.analysis.technical_config.ema_periods = self.analysis.technical_config.ma_periods
        
        # 创建必要的目录
        self.data_config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.processed_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.daily_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.sft_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.train_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.val_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.test_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志文件路径
        self.log_file = self.log_dir / f"{self.project_name}.log"
        
        # 从环境变量加载OpenAI配置
        if not self.openai:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai = OpenAIConfig(
                    api_key=api_key,
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
                    timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
                    retry_count=int(os.getenv("OPENAI_RETRY_COUNT", "3")),
                    retry_delay=int(os.getenv("OPENAI_RETRY_DELAY", "5"))
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        config_dict = {
            "project_name": self.project_name,
            "version": self.version,
            "debug": self.debug,
            "base_dir": str(self.base_dir),
            "data_config": {
                "raw_dir": str(self.data_config.raw_dir),
                "processed_dir": str(self.data_config.processed_dir),
                "daily_dir": str(self.data_config.daily_dir),
                "sft_dir": str(self.data_config.sft_dir),
                "train_dir": str(self.data_config.train_dir),
                "val_dir": str(self.data_config.val_dir),
                "test_dir": str(self.data_config.test_dir)
            },
            "log_dir": str(self.log_dir),
            "output_dir": str(self.output_dir),
            "data_source": {
                "name": self.data_source.name,
                "timeout": self.data_source.timeout,
                "retry_count": self.data_source.retry_count
            },
            "market": {
                "market": self.market.market,
                "exchange": self.market.exchange,
                "timezone": self.market.timezone
            },
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "chunk_size": self.chunk_size,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "analysis": {
                "fundamental_enabled": self.analysis.fundamental_enabled,
                "fundamental_metrics": self.analysis.fundamental_metrics,
                "technical_enabled": self.analysis.technical_enabled,
                "technical_indicators": self.analysis.technical_indicators,
                "news_enabled": self.analysis.news_enabled,
                "news_sources": self.analysis.news_sources,
                "news_lookback_days": self.analysis.news_lookback_days,
                "technical_config": {
                    "ma_periods": self.analysis.technical_config.ma_periods,
                    "ema_periods": self.analysis.technical_config.ema_periods,
                    "macd_fast": self.analysis.technical_config.macd_fast,
                    "macd_slow": self.analysis.technical_config.macd_slow,
                    "macd_signal": self.analysis.technical_config.macd_signal,
                    "rsi_period": self.analysis.technical_config.rsi_period,
                    "kdj_k": self.analysis.technical_config.kdj_k,
                    "kdj_d": self.analysis.technical_config.kdj_d,
                    "kdj_j": self.analysis.technical_config.kdj_j,
                    "boll_period": self.analysis.technical_config.boll_period,
                    "boll_std": self.analysis.technical_config.boll_std
                }
            }
        }
        
        if self.openai:
            config_dict["openai"] = {
                "base_url": self.openai.base_url,
                "model": self.openai.model,
                "temperature": self.openai.temperature,
                "max_tokens": self.openai.max_tokens,
                "timeout": self.openai.timeout,
                "retry_count": self.openai.retry_count,
                "retry_delay": self.openai.retry_delay
            }
        
        return config_dict 
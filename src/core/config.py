from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

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
    # 原始数据目录
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    # 处理后的数据目录
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    # 日线数据目录
    daily_dir: Path = field(default_factory=lambda: Path("data/processed/daily"))
    # SFT数据目录
    sft_dir: Path = field(default_factory=lambda: Path("data/processed/sft"))
    # 训练集目录
    train_dir: Path = field(default_factory=lambda: Path("data/processed/sft/train"))
    # 验证集目录
    val_dir: Path = field(default_factory=lambda: Path("data/processed/sft/val"))
    # 测试集目录
    test_dir: Path = field(default_factory=lambda: Path("data/processed/sft/test"))

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
        'ma5', 'ma10', 'ma25', 'ma60',
        'vol_ma5', 'vol_ma10'
    ])
    
    # 技术指标参数
    boll_period: int = 25  # 布林带周期
    boll_std: float = 2.0  # 布林带标准差倍数
    macd_fast: int = 12   # MACD快线周期
    macd_slow: int = 26   # MACD慢线周期
    macd_signal: int = 9  # MACD信号线周期
    rsi_period: int = 14  # RSI周期
    kdj_k: int = 9       # KDJ K值周期
    kdj_d: int = 3       # KDJ D值周期
    kdj_j: int = 3       # KDJ J值周期

@dataclass
class BaseConfig:
    """基础配置类"""
    # 基础配置
    project_name: str = "AlphaBot-Data"
    version: str = "1.0.0"
    debug: bool = False
    
    # 路径配置
    base_dir: Path = Path(".")
    data_config: DataConfig = field(default_factory=DataConfig)
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
    batch_size: int = 100
    max_workers: int = 4
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 缓存时间（秒）
    
    # 分析配置
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    def __post_init__(self):
        """初始化后处理"""
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
            import os
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
        
        # 从环境变量加载分析配置
        self.analysis = AnalysisConfig(
            fundamental_enabled=os.getenv('ANALYSIS_FUNDAMENTAL_ENABLED', 'true').lower() == 'true',
            technical_enabled=os.getenv('ANALYSIS_TECHNICAL_ENABLED', 'true').lower() == 'true',
            news_enabled=os.getenv('ANALYSIS_NEWS_ENABLED', 'true').lower() == 'true',
            news_lookback_days=int(os.getenv('ANALYSIS_NEWS_LOOKBACK_DAYS', '5'))
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
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "analysis": {
                "fundamental_enabled": self.analysis.fundamental_enabled,
                "fundamental_metrics": self.analysis.fundamental_metrics,
                "technical_enabled": self.analysis.technical_enabled,
                "technical_indicators": self.analysis.technical_indicators,
                "news_enabled": self.analysis.news_enabled,
                "news_sources": self.analysis.news_sources,
                "news_lookback_days": self.analysis.news_lookback_days
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
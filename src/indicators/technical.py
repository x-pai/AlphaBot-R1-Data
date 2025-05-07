from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.core.config import AnalysisConfig

@dataclass
class IndicatorConfig:
    """技术指标配置"""
    ma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    kdj_k: int = 9
    kdj_d: int = 3
    kdj_j: int = 3
    boll_period: int = 25
    boll_std: float = 2.0
    
    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 60]
        if self.ema_periods is None:
            self.ema_periods = [5, 10, 20, 60]

class TechnicalIndicators:
    """技术指标计算类"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 添加了技术指标的DataFrame
        """
        df = self.calculate_ma(df)
        df = self.calculate_ema(df)
        df = self.calculate_macd(df)
        df = self.calculate_rsi(df)
        df = self.calculate_kdj(df)
        df = self.calculate_bollinger_bands(df)
        return df
    
    def calculate_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 添加了MA指标的DataFrame
        """
        for period in self.config.ma_periods:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指数移动平均线
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 添加了EMA指标的DataFrame
        """
        for period in self.config.ema_periods:
            df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 添加了MACD指标的DataFrame
        """
        # 计算快线和慢线的EMA
        exp1 = df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        
        # 计算MACD线
        df['macd'] = exp1 - exp2
        
        # 计算信号线
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
        
        # 计算MACD柱状图
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算RSI指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 添加了RSI指标的DataFrame
        """
        # 计算价格变化
        delta = df['close'].diff()
        
        # 分别获取上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        
        # 计算相对强度
        rs = gain / loss
        
        # 计算RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_kdj(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算KDJ指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 添加了KDJ指标的DataFrame
        """
        # 计算RSV
        low_list = df['low'].rolling(window=self.config.kdj_k, min_periods=1).min()
        high_list = df['high'].rolling(window=self.config.kdj_k, min_periods=1).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        
        # 计算K值
        df['k'] = rsv.ewm(alpha=1/self.config.kdj_d, adjust=False).mean()
        
        # 计算D值
        df['d'] = df['k'].ewm(alpha=1/self.config.kdj_j, adjust=False).mean()
        
        # 计算J值
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带
        
        Args:
            df: 日线数据
            
        Returns:
            pd.DataFrame: 添加了布林带的数据
        """
        # 使用配置中的参数
        period = self.config.analysis.boll_period
        std = self.config.analysis.boll_std
        
        # 计算中轨（25日移动平均线）
        df['boll_middle'] = df['close'].rolling(window=period).mean()
        
        # 计算标准差
        df['boll_std'] = df['close'].rolling(window=period).std()
        
        # 计算上轨和下轨
        df['boll_upper'] = df['boll_middle'] + (df['boll_std'] * std)
        df['boll_lower'] = df['boll_middle'] - (df['boll_std'] * std)
        
        return df 
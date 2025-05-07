from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from ..core.config import BaseConfig
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

class SFTGenerator:
    """SFT数据生成器"""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据集划分比例
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # 初始化OpenAI客户端
        self.openai_client = None
        if self.config.openai and self.config.openai.api_key:
            try:
                self.openai_client = openai.OpenAI(
                    api_key=self.config.openai.api_key,
                    base_url=self.config.openai.base_url,
                    timeout=self.config.openai.timeout
                )
                self.logger.info("OpenAI client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        else:
            self.logger.warning("OpenAI configuration not found or invalid. Will use template-based completion generation.")
    
    def generate_samples(self, 
                        df: pd.DataFrame,
                        stock: str,
                        num_samples: int = 10,
                        lookback_days: int = 5) -> List[Dict[str, Any]]:
        """生成训练样本
        
        Args:
            df: 股票数据
            stock: 股票代码
            num_samples: 样本数量
            lookback_days: 回溯天数
            
        Returns:
            List[Dict[str, Any]]: 训练样本列表
        """
        samples = []
        
        try:
            # 确保数据按日期排序
            df = df.sort_values('date')
            
            # 计算每日收益率
            df['return'] = df['close'].pct_change()
            
            # 生成样本
            for i in range(len(df) - lookback_days):
                # 获取历史数据窗口
                window = df.iloc[i:i+lookback_days]
                next_day = df.iloc[i+lookback_days]
                
                # 构建输入特征
                features = self._extract_features(window)
                
                # 构建标签
                label = self._generate_label(next_day)
                
                # 构建样本
                sample = {
                    'stock': stock,
                    'date': next_day['date'].strftime('%Y-%m-%d'),
                    'features': features,
                    'label': label,
                    'actual_return': next_day['return']
                }
                
                samples.append(sample)
                
                if len(samples) >= num_samples:
                    break
            
            return samples
        
        except Exception as e:
            self.logger.error(f"Error generating samples for {stock}: {str(e)}")
            return []
    
    def _extract_features(self, window: pd.DataFrame) -> Dict[str, Any]:
        """提取特征
        
        Args:
            window: 历史数据窗口
            
        Returns:
            Dict[str, Any]: 特征字典
        """
        features = {}
        
        # 价格特征
        features['price'] = {
            'open': window['open'].tolist(),
            'high': window['high'].tolist(),
            'low': window['low'].tolist(),
            'close': window['close'].tolist()
        }
        
        # 成交量特征
        features['volume'] = {
            'volume': window['volume'].tolist(),
            'amount': window['amount'].tolist()
        }
        
        # 技术指标特征
        for col in window.columns:
            if col.startswith(('ma', 'ema', 'macd', 'rsi', 'kdj', 'boll')):
                features[col] = window[col].tolist()
        
        return features
    
    def _generate_label(self, next_day: pd.Series) -> str:
        """生成标签
        
        Args:
            next_day: 下一天的数据
            
        Returns:
            str: 标签
        """
        daily_return = next_day['return']
        
        if daily_return > 0.02:  # 大涨
            return "strong_buy"
        elif daily_return > 0:   # 小涨
            return "buy"
        elif daily_return > -0.02:  # 小跌
            return "sell"
        else:  # 大跌
            return "strong_sell"
    
    def split_dataset(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """划分数据集
        
        Args:
            samples: 样本列表
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]: 训练集、验证集、测试集
        """
        # 按日期排序
        samples = sorted(samples, key=lambda x: x['date'])
        
        # 计算划分点
        n = len(samples)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        # 划分数据集
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        return train_samples, val_samples, test_samples
    
    def save_samples(self, 
                    samples: List[Dict[str, Any]],
                    output_dir: Optional[Path] = None) -> None:
        """保存样本
        
        Args:
            samples: 样本列表
            output_dir: 输出目录
        """
        try:
            # 使用配置中的目录
            train_dir = self.config.data_config.train_dir
            val_dir = self.config.data_config.val_dir
            test_dir = self.config.data_config.test_dir
            
            # 划分数据集
            train_samples, val_samples, test_samples = self.split_dataset(samples)
            
            # 生成时间戳和版本号
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = f"v{self.config.version.replace('.', '_')}"
            
            # 保存训练集
            train_file = train_dir / f"train_{version}_{timestamp}.jsonl"
            with open(train_file, 'w', encoding='utf-8') as f:
                for sample in train_samples:
                    prompt = self.generate_prompt(sample)
                    completion = self.generate_completion(sample)
                    f.write(json.dumps({
                        'input': prompt,
                        'output': completion,
                        'metadata': {
                            'stock': sample['stock'],
                            'date': sample['date'],
                            'version': version,
                            'timestamp': timestamp
                        }
                    }, ensure_ascii=False) + '\n')
            
            # 保存验证集
            val_file = val_dir / f"val_{version}_{timestamp}.jsonl"
            with open(val_file, 'w', encoding='utf-8') as f:
                for sample in val_samples:
                    prompt = self.generate_prompt(sample)
                    completion = self.generate_completion(sample)
                    f.write(json.dumps({
                        'input': prompt,
                        'output': completion,
                        'metadata': {
                            'stock': sample['stock'],
                            'date': sample['date'],
                            'version': version,
                            'timestamp': timestamp
                        }
                    }, ensure_ascii=False) + '\n')
            
            # 保存测试集
            test_file = test_dir / f"test_{version}_{timestamp}.jsonl"
            with open(test_file, 'w', encoding='utf-8') as f:
                for sample in test_samples:
                    prompt = self.generate_prompt(sample)
                    completion = self.generate_completion(sample)
                    f.write(json.dumps({
                        'input': prompt,
                        'output': completion,
                        'metadata': {
                            'stock': sample['stock'],
                            'date': sample['date'],
                            'version': version,
                            'timestamp': timestamp
                        }
                    }, ensure_ascii=False) + '\n')
            
            # 创建数据集信息文件
            info_file = self.config.data_config.sft_dir / f"dataset_info_{version}_{timestamp}.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'version': version,
                    'timestamp': timestamp,
                    'train_samples': len(train_samples),
                    'val_samples': len(val_samples),
                    'test_samples': len(test_samples),
                    'total_samples': len(samples),
                    'train_file': str(train_file),
                    'val_file': str(val_file),
                    'test_file': str(test_file),
                    'config': {
                        'train_ratio': self.train_ratio,
                        'val_ratio': self.val_ratio,
                        'test_ratio': self.test_ratio
                    }
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved {len(train_samples)} train, {len(val_samples)} validation, and {len(test_samples)} test samples")
            self.logger.info(f"Dataset info saved to {info_file}")
        
        except Exception as e:
            self.logger.error(f"Error saving samples: {str(e)}")
    
    def generate_prompt(self, sample: Dict[str, Any]) -> str:
        """生成提示文本
        
        Args:
            sample: 样本数据
            
        Returns:
            str: 提示文本
        """
        features = sample['features']
        current_price = features['price']['close'][-1]
        current_volume = features['volume']['volume'][-1]
        
        # 构建基础提示
        prompt = f"股票代码：{sample['stock']}，"
        prompt += f"当前时间：{sample['date']} 14:15:00，"
        prompt += f"收盘价：{current_price:.2f}，"
        prompt += f"成交量：{current_volume} 手，\n\n"
        
        # 添加历史数据分析
        if self.config.analysis.history_enabled:
            prompt += "**30天历史数据：**\n"
            # 计算价格变化
            price_change = (current_price - features['price']['close'][-30]) / features['price']['close'][-30] * 100
            prompt += f"- 30天价格变化：{price_change:.2f}%\n"
            
            # 计算成交量变化
            volume_change = (current_volume - features['volume']['volume'][-30]) / features['volume']['volume'][-30] * 100
            prompt += f"- 30天成交量变化：{volume_change:.2f}%\n"
            
            # 添加均线数据
            for ma in ['ma5', 'ma10', 'ma20', 'ma60']:
                if ma in features:
                    ma_value = features[ma][-1]
                    ma_change = (ma_value - features[ma][-30]) / features[ma][-30] * 100
                    prompt += f"- {ma.upper()}：{ma_value:.2f}（{ma_change:+.2f}%）\n"
            
            # 添加成交量均线
            for vol_ma in ['vol_ma5', 'vol_ma10']:
                if vol_ma in features:
                    vol_ma_value = features[vol_ma][-1]
                    vol_ma_change = (vol_ma_value - features[vol_ma][-30]) / features[vol_ma][-30] * 100
                    prompt += f"- {vol_ma.upper()}：{vol_ma_value:.2f}（{vol_ma_change:+.2f}%）\n"
            prompt += "\n"
        
        # 添加基本面分析
        if self.config.analysis.fundamental_enabled:
            prompt += "**基本面数据：**\n"
            for metric in self.config.analysis.fundamental_metrics:
                if metric in features:
                    prompt += f"- {metric.upper()}: {features[metric][-1]:.2f}\n"
            prompt += "\n"
        
        # 添加技术面分析
        if self.config.analysis.technical_enabled:
            prompt += "**技术指标：**\n"
            for indicator in self.config.analysis.technical_indicators:
                if indicator in features:
                    prompt += f"- {indicator.upper()}: {features[indicator][-1]:.2f}\n"
            prompt += "\n"
        
        # 添加消息面分析
        if self.config.analysis.news_enabled:
            prompt += "**近期重要消息：**\n"
            if 'news' in features:
                for news in features['news'][-self.config.analysis.news_lookback_days:]:
                    prompt += f"- {news['date']}: {news['title']}\n"
            prompt += "\n"
        
        # 添加问题
        prompt += "请从基本面、技术面和消息面三个维度分析该股票，重点关注30天历史数据的变化趋势，并给出具体的操作建议。"
        
        return prompt
    
    def generate_completion(self, sample: Dict[str, Any]) -> str:
        """生成分析文本
        
        Args:
            sample: 样本数据
            
        Returns:
            str: 分析文本
        """
        if not self.openai_client:
            self.logger.info("Using template-based completion generation")
            return self._generate_template_completion(sample)
            
        try:
            # 构建系统提示
            system_prompt = """你是一个专业的股票分析师，请从以下三个维度分析股票：
1. 基本面分析：关注公司的财务状况、经营状况等基本面指标
2. 技术面分析：关注价格走势、成交量、技术指标等
3. 历史数据分析：关注30天内的价格和成交量变化趋势，以及均线系统的变化
4. 消息面分析：关注公司公告、行业新闻等

请给出详细的分析和具体的操作建议。"""
            
            # 构建用户提示
            user_prompt = self.generate_prompt(sample)
            
            # 调用 OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API 调用失败: {str(e)}")
            return self._generate_template_completion(sample)
            
    def _generate_template_completion(self, sample: Dict[str, Any]) -> str:
        """生成模板化的分析文本（作为 OpenAI API 调用的备选方案）
        
        Args:
            sample: 样本数据
            
        Returns:
            str: 分析文本
        """
        features = sample['features']
        current_price = features['price']['close'][-1]
        
        # 计算30天价格变化
        price_change = (current_price - features['price']['close'][-30]) / features['price']['close'][-30] * 100
        
        # 根据标签生成结论
        label = sample['label']
        if label == 'strong_buy':
            conclusion = "强烈建议买入"
        elif label == 'buy':
            conclusion = "建议买入"
        elif label == 'hold':
            conclusion = "建议持有"
        elif label == 'sell':
            conclusion = "建议卖出"
        else:
            conclusion = "建议观望"
            
        # 构建分析文本
        analysis = f"【结论】\n{conclusion}\n\n"
        
        # 添加历史数据分析
        if self.config.analysis.history_enabled:
            analysis += "【历史数据分析】\n"
            analysis += f"1. 价格趋势：30天价格变化 {price_change:+.2f}%\n"
            
            # 分析均线系统
            ma_trends = []
            for ma in ['ma5', 'ma10', 'ma20', 'ma60']:
                if ma in features:
                    ma_value = features[ma][-1]
                    ma_change = (ma_value - features[ma][-30]) / features[ma][-30] * 100
                    ma_trends.append(f"{ma.upper()} {ma_change:+.2f}%")
            analysis += f"2. 均线系统：{', '.join(ma_trends)}\n"
            
            # 分析成交量趋势
            volume_change = (features['volume']['volume'][-1] - features['volume']['volume'][-30]) / features['volume']['volume'][-30] * 100
            analysis += f"3. 成交量趋势：30天成交量变化 {volume_change:+.2f}%\n\n"
        
        # 添加基本面分析
        if self.config.analysis.fundamental_enabled:
            analysis += "【基本面分析】\n"
            for metric in self.config.analysis.fundamental_metrics:
                if metric in features:
                    analysis += f"- {metric.upper()}: {features[metric][-1]:.2f}\n"
            analysis += "\n"
        
        # 添加技术面分析
        if self.config.analysis.technical_enabled:
            analysis += "【技术面分析】\n"
            for indicator in self.config.analysis.technical_indicators:
                if indicator in features:
                    analysis += f"- {indicator.upper()}: {features[indicator][-1]:.2f}\n"
            analysis += "\n"
        
        # 添加消息面分析
        if self.config.analysis.news_enabled:
            analysis += "【消息面分析】\n"
            if 'news' in features:
                for news in features['news'][-self.config.analysis.news_lookback_days:]:
                    analysis += f"- {news['date']}: {news['title']}\n"
            analysis += "\n"
        
        # 添加操作建议
        analysis += "【操作建议】\n"
        if price_change > 10:
            analysis += "1. 注意高位回调风险\n"
        elif price_change < -10:
            analysis += "1. 关注超跌反弹机会\n"
        else:
            analysis += "1. 关注区间震荡机会\n"
            
        if abs(volume_change) > 50:
            analysis += "2. 成交量显著变化，需密切关注\n"
        else:
            analysis += "2. 成交量相对稳定\n"
            
        analysis += "3. 建议设置止损位，控制风险\n"
        
        return analysis 
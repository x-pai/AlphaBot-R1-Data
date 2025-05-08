# AlphaBot-R1-Data

A股数据采集、处理和分析工具，支持基本面、技术面和消息面分析。

## 功能特点

- 数据采集：支持A股日线数据、基本面数据、新闻数据采集
- 数据处理：数据清洗、标准化、特征工程
- 技术分析：支持多种技术指标计算
  - 移动平均线（MA5/10/20/60）
  - 指数移动平均线（EMA5/10/20/60）
  - MACD（12/26/9）
  - RSI（14日）
  - KDJ（9/3/3）
  - 布林带（25日，2倍标准差）
- 基本面分析：支持多种财务指标分析
  - PE、PB、ROE、ROA
  - 毛利率、净利率
  - 资产负债率、流动比率、速动比率
- 消息面分析：支持公司公告、新闻、社交媒体分析
- SFT数据生成：支持生成用于模型训练的结构化数据

## 环境要求

- Python 3.8+
- 依赖包：见 requirements.txt

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/x-pai/AlphaBot-R1-Data.git
cd AlphaBot-R1-Data
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，填入必要的配置信息
```

## 配置说明

### 数据源配置
```python
data_source:
  name: "akshare"  # 数据源名称
  api_key: ""      # API密钥（如果需要）
  timeout: 30      # 请求超时时间
  retry_count: 3   # 重试次数
```

### 分析配置
```python
analysis:
  # 基本面分析
  fundamental_enabled: true
  fundamental_metrics:
    - pe
    - pb
    - roe
    - roa
    - gross_margin
    - net_margin
    - debt_ratio
    - current_ratio
    - quick_ratio

  # 技术面分析
  technical_enabled: true
  technical_indicators:
    - ma
    - ema
    - macd
    - rsi
    - kdj
    - boll
    - volume_ma
    - obv
    - cci
    - dmi

  # 技术指标参数
  boll_period: 25      # 布林带周期（已更新为25日）
  boll_std: 2.0        # 布林带标准差倍数
  macd_fast: 12        # MACD快线周期
  macd_slow: 26        # MACD慢线周期
  macd_signal: 9       # MACD信号线周期
  rsi_period: 14       # RSI周期
  kdj_k: 9            # KDJ K值周期
  kdj_d: 3            # KDJ D值周期
  kdj_j: 3            # KDJ J值周期

  # 消息面分析
  news_enabled: true
  news_sources:
    - announcement
    - news
    - social_media
  news_lookback_days: 5  # 新闻回溯天数

  # 历史数据分析
  history_enabled: true
  history_days: 30
  history_metrics:
    - price
    - volume
    - amount
    - turnover
    - ma5
    - ma10
    - ma20
    - ma60
    - vol_ma5
    - vol_ma10
```

### OpenAI配置（可选）
```python
openai:
  api_key: "your-api-key"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000
```

## 使用示例

1. 采集A股数据：
```bash
# 采集指定日期的数据
python -m src.cli.main --date 20240320

# 采集指定日期范围的数据
python -m src.cli.main --start-date 20240301 --end-date 20240320

# 采集指定类型的股票列表
python -m src.cli.main --stock-list-type strong  # 可选：all, strong, weak, custom

# 采集自定义股票列表
python -m src.cli.main --stock-list-type custom --custom-stocks 000001 600000 601318
```

2. 生成SFT数据：
```bash
# 生成指定日期的SFT数据
python -m src.cli.main --date 20240320 --generate-sft

# 生成指定日期范围的SFT数据
python -m src.cli.main --start-date 20240301 --end-date 20240320 --generate-sft

# 生成指定股票列表的SFT数据
python -m src.cli.main --stock-list-type custom --custom-stocks 000001 600000 601318 --generate-sft

# 自定义SFT生成参数
python -m src.cli.main --generate-sft --sft-samples 20 --sft-lookback 10
```

3. 选股：
```bash
# 基本选股
python -m src.cli.stock_selector --date 20240320

# 自定义选股条件
python -m src.cli.stock_selector \
    --date 20240320 \
    --min-price 5 \
    --max-price 100 \
    --min-volume 1000000 \
    --min-market-cap 1000000000 \
    --max-pe 50 \
    --min-roe 10 \
    --min-profit-growth 20

# 输出到指定文件
python -m src.cli.stock_selector \
    --date 20240320 \
    --output-file output/selected_stocks.csv
```

4. 命令行参数说明：
```bash
# 数据采集参数
--date: 指定日期（YYYYMMDD）
--start-date: 开始日期（YYYYMMDD）
--end-date: 结束日期（YYYYMMDD）
--stock-list-type: 股票列表类型（all/strong/weak/custom）
--custom-stocks: 自定义股票列表（空格分隔）
--output-dir: 输出目录
--log-dir: 日志目录
--indicators: 要计算的技术指标（ma/ema/macd/rsi/kdj/boll等）

# SFT生成参数
--generate-sft: 是否生成SFT数据
--sft-samples: 每只股票生成的SFT样本数量
--sft-lookback: SFT样本的回溯天数

# 选股参数
--min-price: 最低价格
--max-price: 最高价格
--min-volume: 最低成交量（手）
--min-market-cap: 最低市值（元）
--max-pe: 最高市盈率
--min-roe: 最低净资产收益率（%）
--min-profit-growth: 最低利润增长率（%）
--output-file: 输出文件路径
```

## 目录结构

```
.
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
│       ├── daily/        # 日线数据
│       └── sft/          # SFT数据
├── logs/                 # 日志目录
├── output/               # 输出目录
├── src/                  # 源代码
│   ├── cli/             # 命令行工具
│   ├── core/            # 核心功能
│   ├── indicators/      # 技术指标
│   ├── market_data/     # 市场数据
│   └── models/          # 模型相关
├── tests/               # 测试代码
├── .env.example         # 环境变量示例
├── requirements.txt     # 依赖包列表
└── README.md           # 项目说明
```

## 更新日志

### 2024-03-20
- 更新布林带周期从20日改为25日
- 优化布林带计算逻辑
- 添加历史数据分析配置
- 完善技术指标参数配置

## 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交 Pull Request

## 许可证

MIT License 
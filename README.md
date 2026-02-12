# Djinn - 多市场量化回测框架

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Djinn 是一个专业的开源量化回测框架，支持美股、港股和中国股票的多市场回测。框架提供完整的数据获取、策略开发、回测分析和投资组合管理功能，专注于模块化设计、类型安全和生产就绪的代码质量。

## 特性

### 🎯 多市场数据支持
- **美股数据**: 通过 Yahoo Finance 获取OHLCV、基本面数据、市场状态
- **A股/港股数据**: 通过 AKShare 获取中国和香港市场数据
- **统一数据接口**: 所有市场使用相同的 `MarketData` 结构和 `DataProvider` API
- **智能缓存**: 可配置的缓存系统减少重复数据请求
- **请求限制**: 自动限流避免被数据源限制

### 📊 专业回测引擎
- **双引擎架构**: 事件驱动回测 (精准模拟真实交易) + 向量化回测 (高性能计算)
- **完整交易模型**: 支持市价单、限价单、止损单
- **费用计算**: 佣金、滑点、印花税模型
- **绩效评估**: 30+种绩效指标，包括夏普比率、索提诺比率、最大回撤、Calmar比率等
- **详细交易记录**: 完整的交易历史和持仓跟踪

### 🧠 简化策略框架 (推荐)
- **SimpleStrategy**: 极简策略开发框架，15行代码实现完整策略
- **参数声明系统**: 使用 `param()` 声明式定义策略参数，自动验证
- **预置策略库**: RSI、MACD、布林带、均值回归等常用策略开箱即用
- **旧版兼容**: `Strategy` ABC 基类保留供高级用户使用

### 🏦 投资组合管理 (基础框架)
- **组合管理基类**: `Portfolio` 抽象基类定义核心接口
- **资产管理**: 资产跟踪、现金管理、持仓计算
- **风险管理**: 风险度量和控制框架
- **再平衡策略**: 定期和阈值再平衡接口

### ⚡ 高性能设计
- **向量化计算**: 使用 pandas/numpy 进行高效数值计算
- **多级缓存**: 内存和磁盘缓存优化数据访问
- **模块化架构**: 松耦合组件便于扩展和定制
- **类型安全**: 全面的类型注解和 mypy 严格检查

## 快速开始

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/djinn.git
cd djinn

# 2. 创建虚拟环境（推荐使用 uv）
uv venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 安装依赖和开发工具
uv pip install -e ".[dev]"

# 4. 安装预提交钩子（可选）
pre-commit install
```

### 基础使用示例

#### 使用预置策略（最简单）

```python
from datetime import datetime, timedelta
from djinn.core.strategy.impl import RSIStrategy
from djinn.core.backtest import EventDrivenBacktestEngine
from djinn.data.providers.yahoo_finance import YahooFinanceProvider

# 1. 获取数据
provider = YahooFinanceProvider(cache_enabled=True)
market_data = provider.get_ohlcv(
    symbol="AAPL",
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now(),
    interval="1d"
)

# 2. 使用预置策略
strategy = RSIStrategy(period=14, oversold=30, overbought=70)

# 3. 运行回测
engine = EventDrivenBacktestEngine(
    initial_capital=100000,
    commission_rate=0.001,
    slippage_rate=0.0005
)

result = engine.run(
    strategy=strategy,
    data={"AAPL": market_data.to_dataframe()}
)

# 4. 查看结果
print(f"总收益率: {result.total_return:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
```

#### 自定义策略（推荐方式）

```python
from djinn import SimpleStrategy, param
import pandas as pd
import numpy as np

class MyStrategy(SimpleStrategy):
    """自定义双均线策略，仅需约15行代码"""

    # 使用 param() 声明参数
    fast = param(10, min=2, max=100, description="快速均线周期")
    slow = param(30, min=5, max=200, description="慢速均线周期")

    def signals(self, data):
        """生成交易信号"""
        # 计算均线
        fast_ma = data['close'].rolling(self.params.fast).mean()
        slow_ma = data['close'].rolling(self.params.slow).mean()

        # 快线上穿慢线买入(1)，下穿卖出(-1)
        return pd.Series(np.where(fast_ma > slow_ma, 1, -1), index=data.index)

# 使用策略
strategy = MyStrategy(fast=10, slow=30)
```

### 可用预置策略

```python
from djinn.core.strategy.impl import (
    RSIStrategy,           # RSI相对强弱指标策略
    BollingerBandsStrategy, # 布林带策略
    MACDStrategy,          # MACD指标策略
    MeanReversionStrategy  # 均值回归策略
)

# RSI策略: 超卖买入，超买卖出
rsi = RSIStrategy(period=14, oversold=30, overbought=70)

# 布林带策略: 触及下轨买入，触及上轨卖出
bb = BollingerBandsStrategy(period=20, std_dev=2.0)

# MACD策略: MACD线在信号线上方买入
macd = MACDStrategy(fast=12, slow=26, signal=9)

# 均值回归: 价格偏离均线一定幅度时交易
mr = MeanReversionStrategy(period=20, threshold=0.05)
```

## 项目结构

```
djinn/
├── src/djinn/                    # 主包代码
│   ├── __init__.py               # 公开API导出
│   ├── data/                     # 数据层
│   │   ├── providers/            # 数据提供器
│   │   ├── market_data.py        # 市场数据模型
│   │   └── ...
│   ├── core/                     # 核心层
│   │   ├── strategy/             # 策略框架
│   │   │   ├── simple.py         # SimpleStrategy 基类（推荐）
│   │   │   ├── parameter.py      # 参数声明系统
│   │   │   ├── base.py           # Strategy ABC（旧版）
│   │   │   ├── impl/             # 预置策略实现
│   │   │   │   ├── rsi.py
│   │   │   │   ├── macd.py
│   │   │   │   ├── bollinger_bands.py
│   │   │   │   └── mean_reversion.py
│   │   │   └── ...
│   │   ├── backtest/             # 回测引擎
│   │   │   ├── event_driven.py   # 事件驱动引擎
│   │   │   └── vectorized.py     # 向量化引擎
│   │   └── portfolio/            # 投资组合管理
│   ├── utils/                    # 工具层
│   └── visualization/            # 可视化层
├── examples/                     # 示例代码
│   └── ma_crossover_simple_example.py  # 使用SimpleStrategy的示例
├── tests/                        # 测试文件
├── configs/                      # 配置文件
└── docs/                         # 文档
```

## 核心模块

### 简化策略框架 (`djinn` - 主包导出)

**推荐使用 `SimpleStrategy` 框架开发策略：**

```python
from djinn import SimpleStrategy, param

class MyStrategy(SimpleStrategy):
    # 声明参数
    param1 = param(default, min=..., max=..., description="...")

    def signals(self, data):
        # 实现信号逻辑
        return pd.Series(...)
```

**预置策略** (`djinn.core.strategy.impl`):
- `RSIStrategy`: RSI相对强弱指标策略
- `BollingerBandsStrategy`: 布林带策略
- `MACDStrategy`: MACD指标策略
- `MeanReversionStrategy`: 均值回归策略

### 数据模块 (`djinn.data`)
- `DataProvider`: 数据提供器抽象基类
- `YahooFinanceProvider`: 美股数据提供器
- `AKShareProvider`: A股/港股数据提供器
- `MarketData`: 统一的市场数据结构

### 回测模块 (`djinn.core.backtest`)
- `EventDrivenBacktestEngine`: 事件驱动回测引擎
- `VectorizedBacktestEngine`: 向量化回测引擎
- `BacktestResult`: 回测结果容器

## 开发指南

### 环境设置

```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate

# 安装开发依赖
uv pip install -e ".[dev]"

# 安装预提交钩子
pre-commit install
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_simple_strategy.py

# 带覆盖率
pytest --cov=src/djinn

# 并行测试
pytest -n auto
```

### 代码质量

```bash
# 格式化
black src/djinn

# 检查
ruff check src/djinn

# 类型检查
mypy src/djinn
```

## 路线图

### v0.1.0 (已完成)
- [x] 基础项目结构
- [x] 美股数据支持 (Yahoo Finance)
- [x] A股/港股数据支持 (AKShare)
- [x] 双回测引擎 (事件驱动 + 向量化)
- [x] 常用技术指标
- [x] SimpleStrategy 简化框架
- [x] 预置策略库 (RSI, MACD, Bollinger, MeanReversion)

### v0.2.0 (开发中)
- [ ] 投资组合管理完善
- [ ] 参数优化框架
- [ ] 高级可视化
- [ ] 并行计算优化

### v0.3.0 (计划中)
- [ ] 机器学习策略集成
- [ ] 实时数据支持
- [ ] Web 界面
- [ ] 数据库存储

## 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件了解详情。

---

**注意**: 本工具仅用于教育和研究目的。实际交易请谨慎，过去表现不代表未来结果。

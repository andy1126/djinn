# Djinn - 多市场量化回测框架

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Djinn 是一个专业的开源量化回测框架，支持美股、港股和中国股票的多市场回测。框架提供完整的数据获取、策略开发、回测分析和投资组合管理功能，专注于模块化设计、类型安全和生产就绪的代码质量。

## 特性

### 🎯 多市场数据支持 (已实现)
- **美股数据**: 通过 Yahoo Finance 获取OHLCV、基本面数据、市场状态
- **A股/港股数据**: 通过 AKShare 获取中国和香港市场数据
- **统一数据接口**: 所有市场使用相同的 `MarketData` 结构和 `DataProvider` API
- **智能缓存**: 可配置的缓存系统减少重复数据请求
- **请求限制**: 自动限流避免被数据源限制

### 📊 专业回测引擎 (已实现)
- **双引擎架构**: 事件驱动回测 (精准模拟真实交易) + 向量化回测 (高性能计算)
- **完整交易模型**: 支持市价单、限价单、止损单
- **费用计算**: 佣金、滑点、印花税模型
- **绩效评估**: 30+种绩效指标，包括夏普比率、索提诺比率、最大回撤、Calmar比率等
- **详细交易记录**: 完整的交易历史和持仓跟踪

### 🧠 策略框架 (已实现)
- **策略基类**: 易于扩展的 `Strategy` 抽象基类
- **技术指标库**: MA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV, Stochastic, VWAP, Ichimoku Cloud
- **预置策略**: 双均线交叉策略 (MovingAverageCrossover)
- **信号系统**: 灵活的信号生成和仓位管理
- **参数验证**: 自动参数验证和类型检查

### 🏦 投资组合管理 (基础框架)
- **组合管理基类**: `Portfolio` 抽象基类定义核心接口
- **资产管理**: 资产跟踪、现金管理、持仓计算
- **风险管理**: 风险度量和控制框架
- **再平衡策略**: 定期和阈值再平衡接口

### ⚡ 高性能设计 (部分实现)
- **向量化计算**: 使用 pandas/numpy 进行高效数值计算
- **多级缓存**: 内存和磁盘缓存优化数据访问
- **模块化架构**: 松耦合组件便于扩展和定制
- **类型安全**: 全面的类型注解和 mypy 严格检查

### 📈 可视化与报告 (计划中)
- **交互式图表**: 基于 Plotly 的权益曲线和回撤图表
- **性能报告**: HTML/PDF 格式的详细回测报告
- **Jupyter 集成**: Notebook 友好的可视化和分析工具

## 快速开始

### 安装

由于项目仍在开发中，请从源码安装：

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

或者直接安装依赖：

```bash
pip install -e .
```

### 基础使用示例

```python
from datetime import datetime, timedelta

# 导入 Djinn 模块
from djinn.data.providers.yahoo_finance import YahooFinanceProvider
from djinn.core.strategy import MovingAverageCrossover
from djinn.core.backtest import EventDrivenBacktestEngine

# 1. 获取数据
provider = YahooFinanceProvider(cache_enabled=True)
market_data = provider.get_ohlcv(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2023-12-31",
    interval="1d"
)

# 2. 创建双均线交叉策略
strategy = MovingAverageCrossover(
    fast_period=10,
    slow_period=30,
    position_sizing_method="fixed_fractional",
    position_size=0.1  # 10% 的资本
)

# 3. 运行事件驱动回测
engine = EventDrivenBacktestEngine(
    initial_capital=100000,
    commission_rate=0.001,  # 0.1% 佣金
    slippage_rate=0.0005  # 0.05% 滑点
)

# 运行回测
result = engine.run(
    strategy=strategy,
    market_data=market_data,
    verbose=True
)

# 4. 查看回测结果
print(f"初始资本: ${result.initial_capital:,.2f}")
print(f"最终资本: ${result.final_capital:,.2f}")
print(f"总收益率: {result.total_return:.2%}")
print(f"年化收益率: {result.annualized_return:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown:.2%}")
print(f"总交易次数: {result.total_trades}")
print(f"胜率: {result.win_rate:.2%}")

# 5. 查看交易详情
if result.trades:
    print("\n最近5笔交易:")
    for trade in result.trades[-5:]:
        print(f"  {trade.timestamp.date()}: {trade.direction} {trade.symbol} "
              f"{trade.quantity}股 @ ${trade.price:.2f}")
```

更完整的示例请查看 `examples/basic_backtest.py`。

### 示例代码

我们提供了完整的示例代码，涵盖：

1. **基础回测示例** (`examples/basic_backtest.py`):
   - 从 Yahoo Finance 下载真实市场数据
   - 创建双均线交叉策略
   - 运行事件驱动和向量化回测
   - 比较不同回测引擎的结果
   - 生成简单的性能报告和可视化

2. **多市场数据示例** (计划中):
   - 使用 AKShareProvider 获取A股/港股数据
   - 多市场策略回测
   - 货币转换和跨市场投资组合

3. **策略开发示例** (计划中):
   - 自定义策略实现
   - 技术指标使用
   - 参数优化和网格搜索

查看 `examples/` 目录获取最新示例。

## 项目结构

```
djinn/
├── src/djinn/              # 主包代码
│   ├── data/              # 数据层
│   │   ├── providers/     # 数据提供器 (YahooFinance, AKShare)
│   │   ├── base.py        # DataProvider 抽象基类
│   │   └── market_data.py # 市场数据模型
│   ├── core/              # 核心层
│   │   ├── strategy/      # 策略框架和技术指标
│   │   ├── backtest/      # 回测引擎 (事件驱动 + 向量化)
│   │   └── portfolio/     # 投资组合管理框架
│   ├── utils/             # 工具层 (配置、日志、验证等)
│   └── visualization/     # 可视化层 (待完善)
├── examples/              # 示例代码
│   └── basic_backtest.py  # 基础回测示例
├── configs/               # 配置文件
├── docs/                  # 文档
└── pyproject.toml         # 项目配置和依赖管理
```

## 核心模块

### 数据模块 (`djinn.data`)
- `DataProvider`: 数据提供器抽象基类，定义统一的数据获取接口
- `YahooFinanceProvider`: 美股数据提供器，支持OHLCV、基本面数据、市场状态
- `AKShareProvider`: A股/港股数据提供器，支持中国和香港市场
- `MarketData`: 统一的市场数据结构，支持OHLCV和基本面数据
- `MarketDataRequest`: 数据请求模型，提供参数验证
- 数据缓存: 可配置的多级缓存系统，支持内存和磁盘缓存
- 数据清洗: 自动数据验证、缺失值处理和异常值检测

### 策略模块 (`djinn.core.strategy`)
- `Strategy`: 策略抽象基类，定义 `initialize()`, `generate_signals()`, `calculate_indicators()` 等核心方法
- `MovingAverageCrossover`: 双均线交叉策略，支持快慢周期配置和确认机制
- 技术指标库: 包含 MA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV, Stochastic, VWAP, Ichimoku Cloud 等10+常用指标
- `Signal`: 交易信号模型，支持多种信号类型和强度
- `PositionSizing`: 仓位管理，支持固定分数、凯利公式等方法
- 参数系统: 完整的参数验证和类型检查机制

### 回测模块 (`djinn.core.backtest`)
- `BacktestEngine`: 回测引擎抽象基类，定义统一的回测接口
- `EventDrivenBacktestEngine`: 事件驱动回测引擎，模拟真实交易流程，支持精确的费用计算
- `VectorizedBacktestEngine`: 向量化回测引擎，基于 pandas/numpy 的高性能计算
- `BacktestResult`: 回测结果容器，包含30+种绩效指标和完整的交易历史
- `BacktestMode`: 回测模式配置，支持不同粒度和精度设置
- 费用模型: 佣金率、滑点率、印花税率配置
- 交易模型: 支持市价单、限价单、止损单，完整的订单生命周期管理

### 投资组合模块 (`djinn.core.portfolio`)
- `Portfolio`: 投资组合管理抽象基类，定义资产、现金、持仓管理接口
- `PortfolioStatus`: 组合状态枚举 (活跃、关闭、暂停)
- `RebalancingFrequency`: 再平衡频率 (日、周、月、季、年)
- `Asset`: 资产数据模型，包含代码、名称、类型、货币等信息
- `PortfolioRiskManager`: 组合风险管理框架 (基础实现)
- 再平衡策略: 定期再平衡和阈值再平衡接口 (待具体实现)
- 组合构建器: 等权重、市值加权、风险平价等构建方法 (待具体实现)

## 开发指南

### 环境设置

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/djinn.git
cd djinn

# 2. 创建虚拟环境（使用 uv）
uv venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 安装开发依赖
uv pip install -e ".[dev]"

# 4. 安装预提交钩子
pre-commit install
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_strategy.py

# 带覆盖率的测试
pytest --cov=src/djinn

# 并行测试
pytest -n auto
```

### 代码质量

```bash
# 代码格式化
black src/djinn

# 代码检查
ruff check src/djinn

# 类型检查
mypy src/djinn
```

## 配置说明

### 数据源配置

数据提供器可以通过代码直接配置：

```python
from djinn.data.providers.yahoo_finance import YahooFinanceProvider
from djinn.data.providers.akshare_provider import AKShareProvider

# 配置 Yahoo Finance 提供器
yahoo_provider = YahooFinanceProvider(
    cache_enabled=True,
    cache_ttl=3600,  # 缓存1小时
    max_retries=3,  # 最大重试次数
    request_delay=0.5  # 请求延迟，避免被限制
)

# 配置 AKShare 提供器 (A股/港股)
akshare_provider = AKShareProvider(
    cache_enabled=True,
    cache_ttl=3600,
    max_retries=3,
    request_delay=1.0  # 较长的延迟避免被限制
)
```

也可以通过环境变量配置：

```bash
# 缓存配置
export DIJIN_CACHE_ENABLED=true
export DIJIN_CACHE_TTL=3600

# Yahoo Finance 配置
export YAHOO_FINANCE_REQUEST_DELAY=0.5

# AKShare 配置
export AKSHARE_REQUEST_DELAY=1.0
```

### 回测配置

在 `configs/backtest_config.yaml` 中配置回测参数：

```yaml
backtest:
  initial_capital: 100000
  commission: 0.001  # 佣金率
  slippage: 0.0005   # 滑点率
  tax_rate: 0.001    # 印花税率

  risk:
    max_position_size: 0.1  # 最大单仓位比例
    stop_loss: 0.1          # 止损比例
    max_drawdown: 0.2       # 最大回撤限制
```

## 贡献指南

我们欢迎各种形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 路线图

### v0.1.0 (已完成)
- [x] 基础项目结构
- [x] 美股数据支持 (Yahoo Finance)
- [x] 双回测引擎 (事件驱动 + 向量化)
- [x] 常用技术指标 (MA, EMA, MACD, RSI, Bollinger Bands 等)
- [x] A股/港股数据支持 (AKShare)
- [x] 双均线交叉策略示例
- [x] 完整的工作流程示例

### v0.2.0 (开发中)
- [ ] 投资组合管理 (基础框架已建立)
- [ ] 参数优化框架
- [ ] 高级可视化
- [ ] 并行计算和性能优化
- [ ] 多市场投资组合支持

### v0.3.0 (计划中)
- [ ] 机器学习策略集成
- [ ] 实时数据支持
- [ ] Web 界面 (Streamlit)
- [ ] 数据库存储和持久化
- [ ] 生产环境部署和监控

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详细信息。

## 支持

- 文档: [https://djinn.readthedocs.io](https://djinn.readthedocs.io)
- 问题追踪: [GitHub Issues](https://github.com/yourusername/djinn/issues)
- 讨论: [GitHub Discussions](https://github.com/yourusername/djinn/discussions)

## 致谢

感谢以下开源项目的贡献：
- [pandas](https://pandas.pydata.org/) - 数据分析
- [yfinance](https://github.com/ranaroussi/yfinance) - 美股数据
- [AKShare](https://github.com/akfamily/akshare) - A股/港股数据
- [backtesting.py](https://github.com/kernc/backtesting.py) - 回测参考实现

---

**注意**: 本工具仅用于教育和研究目的。实际交易请谨慎，过去表现不代表未来结果。
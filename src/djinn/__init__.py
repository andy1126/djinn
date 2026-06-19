"""djinn — 多市场(A股/港股/美股)量化回测框架。

公开 API:配置、数据、策略、引擎、组合、分析、导出、CLI。
"""

from __future__ import annotations

__version__ = "0.1.0"

# 配置
# 分析 + 导出
from djinn.analytics import Report, build_report, compare_benchmark, compute_metrics

# 运行器(端到端)
from djinn.cli.runner import build_engine_config, build_strategy, run_backtest
from djinn.config import BacktestConfig, dump_config, load_config

# 数据
from djinn.data import (
    Adjust,
    Bar,
    CSVProvider,
    DataCache,
    Market,
    MarketData,
    ProviderRegistry,
    YahooProvider,
    default_registry,
    load_benchmark,
)

# 引擎
from djinn.engine import (
    BacktestResult,
    EngineConfig,
    EventDrivenEngine,
)
from djinn.io import export

# 组合
from djinn.portfolio import Account, EqualWeight, Rebalancer, RiskManager

# 策略
from djinn.strategy import (
    DCA,
    Context,
    MACrossover,
    Momentum,
    RSIReversal,
    Strategy,
    get_strategy_class,
    param,
    param_schema,
)

__all__ = [
    "__version__",
    # 配置
    "BacktestConfig",
    "load_config",
    "dump_config",
    # 数据
    "Market",
    "Adjust",
    "Bar",
    "MarketData",
    "ProviderRegistry",
    "YahooProvider",
    "CSVProvider",
    "DataCache",
    "default_registry",
    "load_benchmark",
    # 策略
    "Strategy",
    "Context",
    "param",
    "param_schema",
    "MACrossover",
    "RSIReversal",
    "Momentum",
    "DCA",
    "get_strategy_class",
    # 引擎
    "EventDrivenEngine",
    "EngineConfig",
    "BacktestResult",
    # 组合
    "Account",
    "EqualWeight",
    "Rebalancer",
    "RiskManager",
    # 分析 + 导出
    "Report",
    "build_report",
    "compute_metrics",
    "compare_benchmark",
    "export",
    # 运行器
    "run_backtest",
    "build_engine_config",
    "build_strategy",
]

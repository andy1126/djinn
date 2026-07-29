"""djinn.analytics — 绩效指标 / 交易统计 / 基准对比 / 归因 / 报告。"""

from __future__ import annotations

from djinn.analytics.attribution import (
    BrinsonResult,
    FactorAttributionResult,
    FactorExposureReport,
    brinson_attribution,
    build_exposure_report,
    factor_attribution,
)
from djinn.analytics.metrics import (
    Metrics,
    compute_max_drawdown,
    compute_metrics,
    monthly_returns,
    rolling_sharpe,
    rolling_volatility,
    yearly_returns,
)
from djinn.analytics.report import Report, build_report
from djinn.analytics.trades import (
    BenchmarkStats,
    TradeStats,
    compare_benchmark,
    compute_trade_stats,
)

__all__ = [
    "BenchmarkStats",
    "BrinsonResult",
    "FactorAttributionResult",
    "FactorExposureReport",
    "Metrics",
    "Report",
    "TradeStats",
    "brinson_attribution",
    "build_exposure_report",
    "build_report",
    "compare_benchmark",
    "compute_max_drawdown",
    "compute_metrics",
    "compute_trade_stats",
    "factor_attribution",
    "monthly_returns",
    "rolling_sharpe",
    "rolling_volatility",
    "yearly_returns",
]

"""djinn.factor.analysis — Alphalens 式因子分析:IC / 分层 / 报告。"""

from __future__ import annotations

from djinn.factor.analysis.forward_returns import compute_forward_returns
from djinn.factor.analysis.ic import compute_ic, ic_by_group, ic_decay, ic_summary
from djinn.factor.analysis.quantile import (
    long_short_curve,
    monotonicity_score,
    quantile_cumulative,
    quantile_returns,
)
from djinn.factor.analysis.report import FactorReport, analyze_factor, rank_turnover

__all__ = [
    "FactorReport",
    "analyze_factor",
    "compute_forward_returns",
    "compute_ic",
    "ic_by_group",
    "ic_decay",
    "ic_summary",
    "long_short_curve",
    "monotonicity_score",
    "quantile_cumulative",
    "quantile_returns",
    "rank_turnover",
]

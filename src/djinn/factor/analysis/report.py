"""因子分析报告:聚合 IC / 分层 / 衰减 / 换手,供 API 序列化与前端展示。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from djinn.factor.analysis.ic import compute_ic, ic_by_group, ic_decay, ic_summary
from djinn.factor.analysis.quantile import (
    long_short_curve,
    monotonicity_score,
    quantile_cumulative,
    quantile_returns,
)


def rank_turnover(factor: pd.DataFrame) -> float:
    """因子换手代理:``1 - 相邻日截面排名自相关`` 的均值(0=完全稳定,越大换手越高)。"""
    if factor.shape[0] < 2:
        return 0.0
    ranked = factor.rank(axis=1)
    ac = ranked.corrwith(ranked.shift(1), axis=1, method="pearson", drop=True)
    ac = ac.dropna()
    return float(1.0 - ac.mean()) if len(ac) else 0.0


@dataclass
class FactorReport:
    """单因子分析报告的聚合结果。"""

    factor_name: str
    ic: pd.Series
    ic_summary: dict[str, float]
    ic_decay: dict[int, pd.Series]
    quantile_returns: pd.DataFrame
    quantile_cumulative: dict[int, pd.Series]
    long_short: pd.Series
    monotonicity: float
    turnover: float
    ic_by_group: pd.Series = field(default_factory=pd.Series)

    # ── 序列化(与 BacktestReport 的 {index,values}/{index,columns,data} 约定一致)──
    @staticmethod
    def _series(s: pd.Series) -> dict[str, Any]:
        return {
            "index": [str(x) for x in s.index],
            "values": [_finite(v) for v in s.to_numpy()],
        }

    @staticmethod
    def _frame(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "index": [str(x) for x in df.index],
            "columns": [str(c) for c in df.columns],
            "data": [[_finite(v) for v in row] for row in df.to_numpy().tolist()],
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON 友好 dict(供 API 响应)。"""
        return {
            "factor_name": self.factor_name,
            "ic": self._series(self.ic),
            "ic_summary": {k: _finite(v) for k, v in self.ic_summary.items()},
            "ic_decay": {str(p): self._series(s) for p, s in self.ic_decay.items()},
            "quantile_returns": self._frame(self.quantile_returns),
            "quantile_cumulative": {
                str(q): self._series(s) for q, s in self.quantile_cumulative.items()
            },
            "long_short": self._series(self.long_short),
            "monotonicity": _finite(self.monotonicity),
            "turnover": _finite(self.turnover),
            "ic_by_group": self._series(self.ic_by_group),
        }


def analyze_factor(
    factor: pd.DataFrame,
    fwd_returns: dict[int, pd.DataFrame],
    *,
    name: str = "factor",
    ic_method: str = "spearman",
    n_quantiles: int = 5,
    industry_map: dict[str, str] | None = None,
) -> FactorReport:
    """一站式因子分析:IC + 汇总 + 衰减 + 分层 + 多空 + 单调性 + 换手。"""
    primary = min(fwd_returns) if fwd_returns else 1
    ic = compute_ic(factor, fwd_returns[primary], method=ic_method)  # type: ignore[arg-type]
    qret = quantile_returns(factor, fwd_returns[primary], n_quantiles)
    return FactorReport(
        factor_name=name,
        ic=ic,
        ic_summary=ic_summary(ic),
        ic_decay=ic_decay(factor, fwd_returns, method=ic_method),  # type: ignore[arg-type]
        quantile_returns=qret,
        quantile_cumulative=quantile_cumulative(qret),
        long_short=long_short_curve(qret),
        monotonicity=monotonicity_score(qret),
        turnover=rank_turnover(factor),
        ic_by_group=(
            ic_by_group(factor, fwd_returns[primary], industry_map, method=ic_method)  # type: ignore[arg-type]
            if industry_map
            else pd.Series(dtype=float)
        ),
    )


def _finite(v: Any) -> float | None:
    """转 finite float(NaN/Inf → None,JSON 安全)。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None

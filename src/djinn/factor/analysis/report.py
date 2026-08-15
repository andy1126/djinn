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


# 调仓频率档位(交易日口径):half_life 落在区间内 → 推荐档。
_HALF_LIFE_BUCKETS: tuple[tuple[int, str], ...] = (
    (2, "daily"),
    (8, "weekly"),
    (15, "monthly"),
)


def _recommend_freq(decay: dict[int, float]) -> str | None:
    """按 IC 衰减曲线推荐调仓频率(C11)。

    规则:找 IC 衰减到峰值 50% 的最短持有期(仅看峰值之后的周期),映射到最近
    调仓档(daily / weekly / monthly / quarterly)。无有效峰值 → None。
    """
    if not decay:
        return None
    peaks = {p: v for p, v in decay.items() if v is not None and math.isfinite(v)}
    if not peaks:
        return None
    peak_period = max(peaks, key=lambda p: abs(peaks[p]))
    peak_val = abs(peaks[peak_period])
    if peak_val == 0:
        return None
    half = peak_val * 0.5
    half_life: int | None = None
    for p in sorted(peaks):
        if p >= peak_period and abs(peaks[p]) <= half:
            half_life = p
            break
    if half_life is None:
        return "quarterly"  # 全程未衰减到一半 → 长期有效
    for bound, freq in _HALF_LIFE_BUCKETS:
        if half_life <= bound:
            return freq
    return "quarterly"


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
    recommended_rebalance: str | None = None
    data_caveats: list[str] = field(default_factory=list)

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
            "recommended_rebalance": self.recommended_rebalance,
            "data_caveats": list(self.data_caveats),
        }


def analyze_factor(
    factor: pd.DataFrame,
    fwd_returns: dict[int, pd.DataFrame],
    *,
    name: str = "factor",
    ic_method: str = "spearman",
    n_quantiles: int = 5,
    industry_map: dict[str, str] | None = None,
    caveats: list[str] | None = None,
) -> FactorReport:
    """一站式因子分析:IC + 汇总 + 衰减 + 分层 + 多空 + 单调性 + 换手。

    ``caveats`` 为数据口径告警(C3,来自 ``FactorEngine.caveats()``),随报告透出。
    """
    primary = min(fwd_returns) if fwd_returns else 1
    ic = compute_ic(factor, fwd_returns[primary], method=ic_method)  # type: ignore[arg-type]
    qret = quantile_returns(factor, fwd_returns[primary], n_quantiles)
    decay = ic_decay(factor, fwd_returns, method=ic_method)  # type: ignore[arg-type]
    decay_means = {p: float(s.mean()) for p, s in decay.items() if len(s)}
    return FactorReport(
        factor_name=name,
        ic=ic,
        ic_summary=ic_summary(ic),
        ic_decay=decay,
        quantile_returns=qret,
        quantile_cumulative=quantile_cumulative(qret),
        long_short=long_short_curve(qret),
        monotonicity=monotonicity_score(qret),
        turnover=rank_turnover(factor),
        recommended_rebalance=_recommend_freq(decay_means),
        ic_by_group=(
            ic_by_group(factor, fwd_returns[primary], industry_map, method=ic_method)  # type: ignore[arg-type]
            if industry_map
            else pd.Series(dtype=float)
        ),
        data_caveats=list(caveats or []),
    )


def _finite(v: Any) -> float | None:
    """转 finite float(NaN/Inf → None,JSON 安全)。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None

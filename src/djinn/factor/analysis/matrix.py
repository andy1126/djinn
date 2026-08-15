"""多因子诊断矩阵:因子两两截面相关 + 每因子 IC 汇总。

用途:在选股策略里加权多个因子前,先看它们是否高度相关(``ep / sp / bp`` 同源于估值,
通常 > 0.8)——避免把同一信号加权三遍。此模块只做"诊断",不替代单因子分析。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from djinn.factor.analysis.fmb import fama_macbeth
from djinn.factor.analysis.forward_returns import compute_forward_returns
from djinn.factor.analysis.ic import compute_ic, ic_summary
from djinn.factor.analysis.report import rank_turnover
from djinn.factor.preprocess import orthogonalize


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pair_corr(
    a: pd.DataFrame, b: pd.DataFrame, method: Literal["spearman", "pearson"]
) -> float:
    """两因子逐日截面相关,跨日取均值(对齐 inner)。"""
    fa, fb = a.align(b, join="inner")
    if fa.shape[0] == 0 or fa.shape[1] == 0:
        return float("nan")
    # corrwith 按 axis=1 逐日截面;两 Series 来自同行
    corrs = fa.corrwith(fb, axis=1, method=method, drop=True)
    corrs = corrs.dropna()
    return float(corrs.mean()) if len(corrs) else float("nan")


@dataclass
class FactorMatrixReport:
    """多因子诊断报告:相关矩阵 + 每因子各周期 IC 汇总 + 换手。"""

    factors: list[str]
    correlation: pd.DataFrame  # index/columns=因子名,值的逐日截面相关跨日均值
    ic_summary: dict[int, dict[str, dict[str, float]]]  # period -> factor -> summary
    turnover: dict[str, float]  # factor -> rank 换手代理
    fmb: dict[str, Any] | None = None  # Fama-MacBeth 回归结果(因子 ≥2 时)

    def to_dict(self) -> dict[str, Any]:
        """JSON 友好 dict(沿用 {index,columns,data} 约定)。"""
        return {
            "factors": list(self.factors),
            "correlation": {
                "index": [str(x) for x in self.correlation.index],
                "columns": [str(c) for c in self.correlation.columns],
                "data": [
                    [_finite(v) for v in row] for row in self.correlation.to_numpy()
                ],
            },
            "ic_summary": {
                str(p): {
                    name: {k: _finite(v) for k, v in summ.items()}
                    for name, summ in per_factor.items()
                }
                for p, per_factor in self.ic_summary.items()
            },
            "turnover": {name: _finite(v) for name, v in self.turnover.items()},
            "fmb": self.fmb,
        }


def analyze_factor_matrix(
    factors: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    periods: list[int] | tuple[int, ...] = (1, 5, 10),
    ic_method: Literal["spearman", "pearson"] = "spearman",
    *,
    orthogonalized: bool = False,
) -> FactorMatrixReport:
    """多因子诊断:相关矩阵 + 每因子各前向期 IC 汇总 + 换手。

    Args:
        factors: ``{因子名: date×symbol 面板}``(由 FactorEngine 算好)。
        prices: 收盘价宽表(用于算前向收益)。
        periods: 前向收益持有期(交易日)。
        ic_method: IC 相关方法。
        orthogonalized: 相关矩阵是否改用 Schmidt 正交化后的因子(诊断正交化效果;
            IC 汇总 / FMB 仍用原始因子,因正交化改变因子语义)。

    Note:
        ``correlation`` 是因子两两的相关(诊断冗余),不是 IC 矩阵——
        IC 本质是"因子 vs 前向收益"的相关,两因子间无对应概念;故此处只给
        每因子对每期前向收益的 ``ic_summary``。
    """
    names = list(factors)
    n = len(names)
    # 相关矩阵(可选正交化:C10 诊断"正交化后因子间相关是否归零")
    corr_factors = orthogonalize(factors, order=names) if orthogonalized else factors
    corr = pd.DataFrame(float("nan"), index=names, columns=names, dtype="float64")
    for i in range(n):
        corr.iloc[i, i] = 1.0
        for j in range(i + 1, n):
            c = _pair_corr(corr_factors[names[i]], corr_factors[names[j]], ic_method)
            corr.iloc[i, j] = c
            corr.iloc[j, i] = c

    # IC 汇总(每因子 × 每期)
    fwd = compute_forward_returns(prices, list(periods))
    ic_summ: dict[int, dict[str, dict[str, float]]] = {}
    for p, fwd_p in fwd.items():
        per_factor: dict[str, dict[str, float]] = {}
        for name, panel in factors.items():
            ic = compute_ic(panel, fwd_p, method=ic_method)
            per_factor[name] = ic_summary(ic)
        ic_summ[int(p)] = per_factor

    # 换手
    turnover = {name: rank_turnover(panel) for name, panel in factors.items()}

    # Fama-MacBeth(因子 ≥2 时):多因子风险溢价显著性
    fmb: dict[str, Any] | None = None
    if n >= 2 and periods:
        fmb = fama_macbeth(
            factors, fwd[next(iter(periods))], standardize=True
        ).to_dict()

    return FactorMatrixReport(
        factors=names,
        correlation=corr,
        ic_summary=ic_summ,
        turnover=turnover,
        fmb=fmb,
    )

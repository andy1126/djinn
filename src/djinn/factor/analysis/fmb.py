"""Fama-MacBeth 截面回归:估计多因子风险溢价(λ)的时间序列显著性。

逐日截面回归 ``r_{t+1} = a + Σ λ_k f_k + ε``,得到每个因子的 λ 时间序列,
再对其均值做 Newey-West t 检验(见 :func:`djinn.factor.analysis.ic._newey_west_t`),
从而判断因子是否获得统计显著的风险溢价(区别于单因子 IC 的显著性)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from djinn.factor.analysis.ic import _newey_west_pvalue, _newey_west_t


@dataclass
class FMBResult:
    """Fama-MacBeth 回归结果。"""

    n_days: int = 0
    # {factor: {"lambda_mean", "lambda_t", "lambda_pvalue", "pos_ratio"}}
    lambdas: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"n_days": self.n_days, "lambdas": self.lambdas}


def fama_macbeth(
    factors: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    *,
    standardize: bool = True,
) -> FMBResult:
    """Fama-MacBeth 两阶段回归(逐日截面 OLS → λ 时序 HAC 检验)。

    Args:
        factors: ``{因子名: date×symbol 因子值面板}``。
        fwd_returns: 单期前向收益(date×symbol),与因子对齐。
        standardize: 是否每日截面 z-score 化各因子(量纲可比,推荐)。

    Returns:
        :class:`FMBResult`(每个因子的 λ 均值 / NW t 值 / p 值 / 正值占比)。
    """
    if not factors:
        return FMBResult()
    # 对齐日期(所有因子与收益的并集 → 交集)
    idx = fwd_returns.index
    for df in factors.values():
        idx = idx.intersection(df.index)
    if len(idx) < 2:
        return FMBResult()

    lambdas: dict[str, list[float]] = {name: [] for name in factors}
    for ts in idx:
        y = fwd_returns.loc[ts]
        # 组装截面设计矩阵
        feats: dict[str, pd.Series] = {}
        for name, df in factors.items():
            if ts not in df.index:
                continue
            col = df.loc[ts].astype(float)
            if standardize:
                # 单日截面 zscore(忽略 NaN)
                mu = col.mean()
                sd = col.std()
                col = (col - mu) / sd if sd and sd > 0 else col - mu
            feats[name] = col
        if not feats:
            continue
        x = pd.DataFrame(feats).reindex(y.index)
        x.insert(0, "const", 1.0)
        x = x.astype(float)
        # 成对剔除 NaN
        mask = x.notna().all(axis=1) & y.notna()
        if int(mask.sum()) <= x.shape[1]:
            continue
        xv = x.to_numpy()[mask.to_numpy()]
        yv = y.to_numpy()[mask.to_numpy()]
        coef, *_ = np.linalg.lstsq(xv, yv, rcond=None)
        # coef[0]=截距,coef[1:]=各因子 λ
        for i, name in enumerate(feats):
            lambdas[name].append(float(coef[i + 1]))

    result: dict[str, dict[str, float]] = {}
    for name, series in lambdas.items():
        if len(series) < 2:
            result[name] = {
                "lambda_mean": 0.0,
                "lambda_t": 0.0,
                "lambda_pvalue": 1.0,
                "pos_ratio": 0.0,
            }
            continue
        s = pd.Series(series, dtype=float)
        mean = float(s.mean())
        t = _newey_west_t(s)
        result[name] = {
            "lambda_mean": mean,
            "lambda_t": t,
            "lambda_pvalue": _newey_west_pvalue(t),
            "pos_ratio": float((s > 0).mean()),
        }
    return FMBResult(n_days=len(idx), lambdas=result)

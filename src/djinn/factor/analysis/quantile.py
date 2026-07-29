"""分层(分位)分析:按因子值分组,看各组前向收益是否单调。

用 ``pd.qcut`` 按日截面分组;输出分层收益、累计曲线、多空曲线与单调性评分。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_returns(
    factor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """逐日按因子值分 ``n_quantiles`` 组,求各组前向收益均值。

    Returns:
        index=date、columns=分位(1=因子最低组 … n=最高组)的均值收益表。
    """
    f, r = factor.align(fwd_returns, join="inner")
    records: dict[pd.Timestamp, dict[int, float]] = {}
    for ts in f.index:
        fv = f.loc[ts].dropna()
        rv = r.loc[ts]
        if len(fv) < n_quantiles:
            continue
        try:
            q = pd.qcut(fv, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue  # 因子值全相同无法分组
        grouped = rv.loc[q.index].groupby(q).mean()
        records[ts] = {int(k) + 1: float(v) for k, v in grouped.items()}
    df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    return df


def quantile_cumulative(quant_ret: pd.DataFrame) -> dict[int, pd.Series]:
    """各分位累计净值曲线(``{quantile: Series}``,起点 1.0)。"""
    out: dict[int, pd.Series] = {}
    for q in quant_ret.columns:
        r = quant_ret[q].fillna(0.0)
        out[int(q)] = (1.0 + r).cumprod()
    return out


def long_short_curve(
    quant_ret: pd.DataFrame,
) -> pd.Series:
    """多空累计曲线 = 最高分位 − 最低分位 的逐日差累积(起点 1.0)。"""
    if quant_ret.shape[1] < 2:
        return pd.Series(dtype=float)
    top = quant_ret.columns.max()
    bottom = quant_ret.columns.min()
    diff = (quant_ret[top] - quant_ret[bottom]).fillna(0.0)
    return (1.0 + diff).cumprod()


def monotonicity_score(quant_ret: pd.DataFrame) -> float:
    """单调性评分:分位序号(1..n)与时间平均分组收益的 Spearman 相关。

    完美单调递增 → +1,完美单调递减 → -1,无单调性 → ~0。
    """
    if quant_ret.shape[1] < 2 or len(quant_ret) == 0:
        return 0.0
    mean_ret = quant_ret.mean().sort_index()
    ranks = pd.Series(
        np.arange(1, len(mean_ret) + 1, dtype=float), index=mean_ret.index
    )
    corr = ranks.corr(mean_ret, method="spearman")
    return float(corr) if pd.notna(corr) else 0.0

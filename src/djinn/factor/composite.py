"""滚动 ICIR 加权合成:把"因子诊断"(IC/ICIR)与"选股打分"缝起来。

:func:`rolling_ic_weights` 用过去滚动窗口的各因子 ICIR(信息比率)归一化得到每日
权重;:func:`composite_score` 用该权重面板对标准化因子值加权合成得分。相比手填
静态权重,滚动 ICIR 权重随因子有效性动态调整。

**防未来函数(关键)**:IC(t) = corr(factor(t), fwd_returns(t)),而 fwd_returns(t)
是 t→t+p 的未来收益 —— 直接用会引入未来函数。因此 ``rolling_ic_weights`` 必须把
IC 序列**右移 ``shift_periods``** 日:``ic_effective(t) = ic(t - shift_periods)``,
即 t 日只使用 t−p 日(其前向收益窗口 t−p→t 已完整落定)的 IC。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from djinn.factor.analysis.ic import compute_ic
from djinn.factor.preprocess import standardize


def rolling_ic_weights(
    factors: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: int = 20,
    shift_periods: int | None = None,
) -> pd.DataFrame:
    """滚动 ICIR 归一化权重面板(``date × factor``)。

    Args:
        factors: ``{因子名: date×symbol 因子值面板}``。
        fwd_returns: 单期前向收益(date×symbol),与调仓频率对齐。
        window: IC 滚动窗口(交易日)。
        min_periods: 滚动 ICIR 所需最少观测数(不足 → 权重 0)。
        shift_periods: 前向收益持有期;非 None 时把 IC 序列右移该期数,
            ``ic_effective(t) = ic(t - shift_periods)``,杜绝未来函数。
            调用方必须传与 fwd_returns 对应的持有期。

    Returns:
        ``date × factor`` 权重面板:w_k(t) = ICIR_k / Σ|ICIR|(符号保留);
        std=0 或窗口不足 → 0。
    """
    if not factors:
        return pd.DataFrame()
    ic_panel = pd.DataFrame(
        {name: compute_ic(factor, fwd_returns) for name, factor in factors.items()}
    )
    if shift_periods is not None and shift_periods > 0:
        ic_panel = ic_panel.shift(shift_periods)
    rolling_mean = ic_panel.rolling(window, min_periods=min_periods).mean()
    rolling_std = ic_panel.rolling(window, min_periods=min_periods).std()
    icir = rolling_mean.div(rolling_std.replace(0.0, np.nan)).fillna(0.0)
    denom = icir.abs().sum(axis=1).replace(0.0, np.nan)
    weights = icir.div(denom, axis=0).fillna(0.0)
    return weights


def composite_score(
    factors: dict[str, pd.DataFrame],
    weights_panel: pd.DataFrame,
) -> pd.DataFrame:
    """按权重面板对标准化因子值加权合成得分(``date × symbol``)。

    Args:
        factors: ``{因子名: date×symbol 因子值面板}``。
        weights_panel: ``date × factor`` 权重面板(:func:`rolling_ic_weights` 输出)。

    Returns:
        每日截面 ``score(t, s) = Σ_k w_k(t) · zscore(f_k(t))[s]`` 得分面板。
    """
    if not factors or weights_panel.empty:
        return pd.DataFrame()
    dates = weights_panel.index
    symbols: list[str] = []
    seen: set[str] = set()
    for df in factors.values():
        for s in df.columns:
            if s not in seen:
                seen.add(s)
                symbols.append(s)
    out = pd.DataFrame(float("nan"), index=dates, columns=symbols, dtype="float64")
    for t in dates:
        w = weights_panel.loc[t]
        score = pd.Series(0.0, index=symbols, dtype=float)
        for name, factor in factors.items():
            if t not in factor.index or w[name] == 0.0:
                continue
            # 单日截面 zscore(复用 preprocess,NaN 保留)
            z = standardize(factor.loc[[t]]).iloc[0].reindex(symbols)
            score = score + w[name] * z.fillna(0.0)
        out.loc[t] = score
    return out

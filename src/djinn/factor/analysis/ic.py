"""IC(信息系数)分析:因子值与前向收益的逐日截面相关。

全程 float64 pandas;IC 用 ``df.corrwith(fwd, axis=1, method=...)`` 逐日截面计算,
自动按 symbol 对齐并成对剔除 NaN。
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd


def _newey_west_t(ic: pd.Series, lags: int | None = None) -> float:
    """HAC(Newey-West)标准误下的 IC 均值 t 值(Bartlett 核)。

    ``lags`` 默认 ``floor(4 * (T/100)^(2/9))``(Newey-West 常用经验带宽)。
    IC 序列存在自相关时,经典 t 值会高估显著性;HAC 修正后更严谨。
    """
    n = len(ic)
    if n < 2:
        return 0.0
    x = ic.to_numpy(dtype=float)
    mean = float(np.mean(x))
    e = x - mean
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))
    # S = Σ_{j=-L}^{L} w_j * γ_j,w_j = 1 - |j|/(L+1)(Bartlett 权重);
    # γ_j = (1/n) Σ_t e_t e_{t-j}(NW 约定),Var(mean) = S / n。
    gamma = np.array([float(np.dot(e[: n - j], e[j:]) / n) for j in range(lags + 1)])
    weights = 1.0 - np.arange(lags + 1) / (lags + 1)
    S = gamma[0] + 2.0 * float(np.sum(weights[1:] * gamma[1:]))
    se = math.sqrt(max(S, 0.0) / n)
    return float(mean / se) if se > 0 else 0.0


def _newey_west_pvalue(t: float) -> float:
    """正态近似双边 p 值(避免引入 scipy 依赖)。"""
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


def compute_ic(
    factor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.Series:
    """逐日截面 IC(factor 与 fwd_returns 的相关序列,index=date)。

    - ``spearman``:Rank IC(对单调性稳健,因子分析默认);
    - ``pearson``:线性 IC。
    """
    f, r = factor.align(fwd_returns, join="inner")
    ic = f.corrwith(r, axis=1, method=method, drop=True)
    return ic.dropna()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """IC 汇总:均值、标准差、ICIR(均值/标准差)、正值占比、HAC t 值与 p 值。"""
    if len(ic) == 0:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "ic_pos_ratio": 0.0,
            "ic_t": 0.0,
            "ic_pvalue": 1.0,
        }
    mean = float(ic.mean())
    std = float(ic.std())
    icir = mean / std if std > 0 else 0.0
    pos = float((ic > 0).mean())
    t = _newey_west_t(ic)
    pvalue = _newey_west_pvalue(t)
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "ic_pos_ratio": pos,
        "ic_t": t,
        "ic_pvalue": pvalue,
    }


def ic_decay(
    factor: pd.DataFrame,
    fwd_returns: dict[int, pd.DataFrame],
    method: Literal["spearman", "pearson"] = "spearman",
) -> dict[int, pd.Series]:
    """IC 衰减:不同持有期下的 IC 序列(``{period: IC series}``)。"""
    return {p: compute_ic(factor, fwd, method) for p, fwd in fwd_returns.items()}


def ic_by_group(
    factor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    industry_map: dict[str, str],
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.Series:
    """分行业 IC(多市场用):每个行业内部独立算 IC 再按日取均值。

    Returns:
        index=行业名、value=该行业平均 IC 的 Series(按 |IC| 降序)。
    """
    groups: dict[str, list[str]] = {}
    for sym in factor.columns:
        ind = industry_map.get(sym)
        if ind is not None:
            groups.setdefault(ind, []).append(sym)
    out: dict[str, float] = {}
    for ind, syms in groups.items():
        if len(syms) < 3:
            continue  # 截面太小,IC 无意义
        ic = compute_ic(factor[syms], fwd_returns[syms], method)
        if len(ic):
            out[ind] = float(ic.mean())
    return pd.Series(out).sort_values(key=abs, ascending=False)

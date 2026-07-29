"""IC(信息系数)分析:因子值与前向收益的逐日截面相关。

全程 float64 pandas;IC 用 ``df.corrwith(fwd, axis=1, method=...)`` 逐日截面计算,
自动按 symbol 对齐并成对剔除 NaN。
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


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
    """IC 汇总:均值、标准差、ICIR(均值/标准差)、正值占比。"""
    if len(ic) == 0:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "ic_pos_ratio": 0.0}
    mean = float(ic.mean())
    std = float(ic.std())
    icir = mean / std if std > 0 else 0.0
    pos = float((ic > 0).mean())
    return {"ic_mean": mean, "ic_std": std, "icir": icir, "ic_pos_ratio": pos}


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

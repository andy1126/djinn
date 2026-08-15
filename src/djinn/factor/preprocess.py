"""因子预处理:去极值、标准化、中性化(均为逐日截面操作,行向量化)。

- :func:`winsorize`:按 MAD 或 σ 截断截面极端值,抑制离群点对 IC / 打分的干扰;
- :func:`standardize`:z-score 或 rank 标准化,使不同量纲因子可加权合成;
- :func:`neutralize`:对行业哑变量 + 对数市值做截面回归取残差,剥离行业 / 规模暴露。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def winsorize(
    df: pd.DataFrame,
    method: Literal["mad", "sigma"] = "mad",
    n: float = 3.0,
) -> pd.DataFrame:
    """逐日截面去极值(截断到 [lower, upper])。

    - ``mad``:中位数 ± n × MAD(对厚尾稳健);
    - ``sigma``:均值 ± n × 标准差。

    D12:全面板向量化(无逐行 Python 回调),与旧实现逐值相等。
    """
    too_few = df.notna().sum(axis=1) < 2  # 有效样本 <2 的行不 clip(保持原值)
    if method == "mad":
        med = df.median(axis=1)
        mad = df.sub(med, axis=0).abs().median(axis=1)
        scale = n * 1.4826
        lo = med - scale * mad
        hi = med + scale * mad
        noop = (mad == 0) | too_few  # MAD==0(全等)与样本不足 → 不截断
    else:
        mu = df.mean(axis=1)
        sd = df.std(axis=1)
        lo = mu - n * sd
        hi = mu + n * sd
        noop = (sd == 0) | too_few
    lo = lo.where(~noop, -np.inf)
    hi = hi.where(~noop, np.inf)
    return df.clip(lower=lo, upper=hi, axis=0)


def standardize(
    df: pd.DataFrame,
    method: Literal["zscore", "rank"] = "zscore",
) -> pd.DataFrame:
    """逐日截面标准化。

    - ``zscore``:减均值除标准差(截面均值≈0、标准差≈1);
    - ``rank``:截面排名归一到 [-0.5, 0.5] 区间居中。
    """
    if method == "rank":
        ranked = df.rank(axis=1, pct=True)
        return ranked.sub(0.5)
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def neutralize(
    df: pd.DataFrame,
    industry_map: dict[str, str] | None = None,
    log_mktcap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """逐日截面中性化:剔除行业与对数市值暴露,取 OLS 残差。

    Args:
        df: 因子面板(date × symbol)。
        industry_map: symbol → 行业名;None 则不加行业哑变量。
        log_mktcap: 对数市值面板(date × symbol);None 则不加市值项。

    Returns:
        残差因子面板(与输入同形状)。自变量不足的日期保持原值去均值。
    """

    out = df.copy()
    symbols = list(df.columns)
    # D12:行业哑变量跨日预计算(industry_map 恒定,get_dummies 只需一次,不再逐日重算)
    ind_dummies: pd.DataFrame | None = None
    if industry_map:
        ind = pd.Series(
            [industry_map.get(s) for s in symbols], index=symbols, dtype=object
        )
        ind_dummies = pd.get_dummies(ind, prefix="ind", dtype=float)
    for ts in df.index:
        y = df.loc[ts].astype(float)
        valid = y.notna()
        if valid.sum() < 3:
            out.loc[ts] = y - y.mean()
            continue
        syms = [s for s in symbols if bool(valid[s])]
        cols: list[pd.Series] = []
        if log_mktcap is not None:
            cap_row = log_mktcap.loc[ts].reindex(symbols).astype(float)
            cols.append(cap_row.loc[syms].rename("logcap"))
        if ind_dummies is not None:
            sub = ind_dummies.loc[syms]
            cols.extend([sub[c] for c in sub.columns])
        if not cols:
            out.loc[ts] = y - y.mean()
            continue
        x = pd.concat(cols, axis=1)
        x.insert(0, "const", 1.0)
        x = x.astype(float)
        mask = x.notna().all(axis=1).to_numpy()
        if int(mask.sum()) <= x.shape[1]:
            out.loc[ts] = y - y.mean()
            continue
        xv = x.to_numpy()[mask]
        yv = y.loc[syms].to_numpy()[mask]
        coef, *_ = np.linalg.lstsq(xv, yv, rcond=None)
        resid = yv - xv @ coef
        # 被 mask 剔除的标的(缺行业/市值自变量)置 NaN,而非保留原值 ——
        # 否则同一行混合"残差"与"原值"两种量纲,截面不可比(C5)。
        row = pd.Series(np.nan, index=y.index)
        sel = np.array(syms, dtype=object)[mask]
        row.loc[list(sel)] = resid
        out.loc[ts] = row
    return out


def orthogonalize(
    factors: dict[str, pd.DataFrame],
    order: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """逐日截面对多因子做 Schmidt 正交化(C10)。

    按 ``order``(默认 dict 顺序)依次将每个因子对**前序(已正交化)因子**回归取
    残差,剥离因子间的线性重叠;被 mask 剔除的标的置 NaN(与 :func:`neutralize` 一致)。

    Args:
        factors: ``{因子名: date×symbol 面板}``。
        order: 正交化顺序(前面的因子保留原值,后面的依次正交到前面)。

    Returns:
        同构 dict;首个因子原样,后续因子为相对前序因子的残差。
    """
    if not factors:
        return {}
    names = [n for n in (order or list(factors)) if n in factors]
    out: dict[str, pd.DataFrame] = {n: factors[n].copy() for n in names}
    if len(names) < 2:
        return out
    first = names[0]
    syms = list(out[first].columns)
    idx = out[first].index
    for ts in idx:
        for i in range(1, len(names)):
            name = names[i]
            y = out[name].loc[ts].astype(float)
            x_cols = [
                out[prev].loc[ts].astype(float).rename(prev) for prev in names[:i]
            ]
            x = pd.concat(x_cols, axis=1).reindex(syms)
            x.insert(0, "const", 1.0)
            x = x.astype(float)
            mask = x.notna().all(axis=1) & y.notna()
            if int(mask.sum()) <= x.shape[1]:
                out[name].loc[ts] = float("nan")
                continue
            xv = x.to_numpy()[mask.to_numpy()]
            yv = y.to_numpy()[mask.to_numpy()]
            coef, *_ = np.linalg.lstsq(xv, yv, rcond=None)
            resid = yv - xv @ coef
            row = pd.Series(np.nan, index=y.index)
            sel = np.array(syms, dtype=object)[mask.to_numpy()]
            row.loc[list(sel)] = resid
            out[name].loc[ts] = row
    return out

"""因子预处理测试:winsorize / standardize / neutralize(本地小样本)。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from djinn.factor.preprocess import neutralize, standardize, winsorize


def _panel(rows: int = 5, cols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([date(2024, 1, 1) + timedelta(days=i) for i in range(rows)])
    syms = [f"S{j}" for j in range(cols)]
    return pd.DataFrame(rng.normal(size=(rows, cols)), index=idx, columns=syms)


def test_standardize_zscore_cross_section() -> None:
    """z-score 后每日截面均值≈0、标准差≈1。"""
    df = _panel(rows=4, cols=20, seed=1) * 100 + 500  # 平移缩放,验证归一
    z = standardize(df, method="zscore")
    assert np.allclose(z.mean(axis=1).to_numpy(), 0.0, atol=1e-8)
    assert np.allclose(z.std(axis=1).to_numpy(), 1.0, atol=1e-8)


def test_standardize_rank_centered() -> None:
    df = _panel(rows=3, cols=10, seed=2)
    r = standardize(df, method="rank")
    assert r.min().min() >= -0.5 and r.max().max() <= 0.5
    # pct-rank 居中偏差为 1/(2n)(n=10 → 0.05),容差放宽
    assert np.allclose(r.mean(axis=1).to_numpy(), 0.0, atol=0.1)


def test_winsorize_clips_extremes() -> None:
    df = _panel(rows=2, cols=10, seed=3)
    df.iloc[0, 0] = 1e6  # 注入极端离群
    w = winsorize(df, method="mad", n=3.0)
    assert w.iloc[0, 0] < 1e6
    # 截面其余值不变(仅截断离群)
    assert w.iloc[0, 0] == pytest.approx(w.iloc[0].drop(index="S0").max(), rel=0.5)


def test_neutralize_removes_market_cap_correlation() -> None:
    """中性化后因子与 ln(市值) 截面相关性绝对值显著下降。"""
    rng = np.random.default_rng(7)
    rows, cols = 30, 30
    idx = pd.DatetimeIndex([date(2024, 1, 1) + timedelta(days=i) for i in range(rows)])
    syms = [f"S{j}" for j in range(cols)]
    logcap = pd.DataFrame(
        np.tile(rng.uniform(8, 12, cols), (rows, 1)), index=idx, columns=syms
    )
    # 因子 = 0.9*logcap + 噪声(强相关)
    noise = pd.DataFrame(rng.normal(0, 0.3, (rows, cols)), index=idx, columns=syms)
    factor = 0.9 * logcap + noise

    def _xcorr(a: pd.DataFrame, b: pd.DataFrame) -> float:
        cors = [a.loc[t].corr(b.loc[t]) for t in idx]
        return float(np.nanmean(np.abs(cors)))

    before = _xcorr(factor, logcap)
    neut = neutralize(factor, industry_map=None, log_mktcap=logcap)
    after = _xcorr(neut, logcap)
    assert before > 0.9
    assert after < 0.1


def test_neutralize_with_industry() -> None:
    """行业中性化:同行业组的残差均值应接近 0。"""
    rng = np.random.default_rng(11)
    rows, cols = 20, 12
    idx = pd.DatetimeIndex([date(2024, 1, 1) + timedelta(days=i) for i in range(rows)])
    syms = [f"S{j}" for j in range(cols)]
    industry_map = {s: ("tech" if j < 6 else "fin") for j, s in enumerate(syms)}
    # tech 组整体偏高
    base = np.array([3.0 if j < 6 else -3.0 for j in range(cols)])
    factor = pd.DataFrame(
        np.tile(base, (rows, 1)) + rng.normal(0, 0.1, (rows, cols)),
        index=idx,
        columns=syms,
    )
    neut = neutralize(factor, industry_map=industry_map, log_mktcap=None)
    row = neut.iloc[-1]
    assert abs(row[[s for s in syms if industry_map[s] == "tech"]].mean()) < 0.5
    assert abs(row[[s for s in syms if industry_map[s] == "fin"]].mean()) < 0.5


def test_neutralize_masked_nan() -> None:
    """被 mask 剔除的标的(缺市值自变量)输出 NaN,而非保留原值(C5)。"""
    rng = np.random.default_rng(5)
    rows, cols = 5, 8
    idx = pd.DatetimeIndex([date(2024, 1, 1) + timedelta(days=i) for i in range(rows)])
    syms = [f"S{j}" for j in range(cols)]
    factor = pd.DataFrame(rng.normal(size=(rows, cols)), index=idx, columns=syms)
    logcap = pd.DataFrame(rng.uniform(8, 12, (rows, cols)), index=idx, columns=syms)
    logcap["S0"] = np.nan  # S0 缺市值 → 应被 mask 剔除
    neut = neutralize(factor, industry_map=None, log_mktcap=logcap)
    # 缺自变量标的输出 NaN(而非原值),其余标的残差与市值去相关
    assert neut["S0"].isna().all()
    assert neut.drop(columns="S0").notna().all().all()


def test_score_cross_section_neutralize() -> None:
    """score_cross_section(neutralize=True) 行业分组均值 ≈ 0。"""
    from djinn.screen.scoring import FactorScore, score_cross_section

    rng = np.random.default_rng(13)
    cols = 12
    syms = [f"S{j}" for j in range(cols)]
    industry_map = {s: ("tech" if j < 6 else "fin") for j, s in enumerate(syms)}
    base = np.array([3.0 if j < 6 else -3.0 for j in range(cols)])
    vals = base + rng.normal(0, 0.1, cols)
    cross = pd.DataFrame({"f": vals}, index=syms)
    out = score_cross_section(
        cross,
        [FactorScore(factor="f", weight=1.0, direction=1)],
        neutralize=True,
        industry_map=industry_map,
    )
    assert abs(out[[s for s in syms if industry_map[s] == "tech"]].mean()) < 0.5
    assert abs(out[[s for s in syms if industry_map[s] == "fin"]].mean()) < 0.5

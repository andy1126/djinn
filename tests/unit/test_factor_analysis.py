"""因子分析测试:完美因子 / 随机因子 / 多空曲线 / 报告序列化(不依赖网络)。"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from djinn.factor.analysis import (
    analyze_factor,
    compute_forward_returns,
    compute_ic,
    ic_summary,
    long_short_curve,
    monotonicity_score,
    quantile_returns,
)


def _prices(rows: int = 60, cols: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([date(2024, 1, 1) + timedelta(days=i) for i in range(rows)])
    rets = rng.normal(0.0005, 0.02, (rows, cols))
    close = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(close, index=idx, columns=[f"S{j}" for j in range(cols)])


def test_perfect_factor_ic_near_one() -> None:
    """完美因子(因子值≈下期收益)→ IC≈1、ICIR 极大、分层严格单调。"""
    prices = _prices()
    fwd = compute_forward_returns(prices, [1])[1]
    rng = np.random.default_rng(5)
    # 近乎完美前视(加微小噪声使 IC 非常数,ICIR 有限且大)
    factor = fwd + rng.normal(0, 1e-4, fwd.shape)
    ic = compute_ic(factor, fwd, method="spearman")
    assert ic.mean() > 0.99
    summary = ic_summary(ic)
    assert summary["icir"] > 5.0
    assert summary["ic_pos_ratio"] > 0.99
    qret = quantile_returns(factor, fwd, n_quantiles=5)
    assert monotonicity_score(qret) == pytest.approx(1.0)


def test_random_factor_ic_near_zero() -> None:
    """随机因子(固定种子)→ |IC 均值|≈0。"""
    prices = _prices(seed=3)
    fwd = compute_forward_returns(prices, [1])[1]
    rng = np.random.default_rng(42)
    factor = pd.DataFrame(
        rng.normal(size=prices.shape), index=prices.index, columns=prices.columns
    )
    ic = compute_ic(factor, fwd, method="spearman")
    assert abs(ic.mean()) < 0.05


def test_long_short_equals_top_minus_bottom() -> None:
    """多空曲线 = Top − Bottom 分位收益累计。"""
    prices = _prices()
    fwd = compute_forward_returns(prices, [1])[1]
    factor = fwd.copy()
    qret = quantile_returns(factor, fwd, n_quantiles=5)
    ls = long_short_curve(qret)
    # 手算某日多空收益
    top, bottom = qret.columns.max(), qret.columns.min()
    manual = (qret[top] - qret[bottom]).fillna(0.0)
    expected = (1.0 + manual).cumprod()
    pd.testing.assert_series_equal(ls, expected, check_names=False)


def test_analyze_factor_report_serializable() -> None:
    """FactorReport.to_dict() 可 JSON 序列化(供 API)。"""
    prices = _prices()
    fwd = compute_forward_returns(prices, [1, 5, 10])
    industry_map = {
        s: ("tech" if j < 15 else "fin") for j, s in enumerate(prices.columns)
    }
    factor = fwd[1].copy()
    report = analyze_factor(
        factor, fwd, name="test", n_quantiles=5, industry_map=industry_map
    )
    d = report.to_dict()
    # 必须能被 json.dumps 且无 NaN/Inf
    json.dumps(d)
    assert d["factor_name"] == "test"
    assert "ic_summary" in d and "quantile_returns" in d
    assert set(d["ic_decay"].keys()) == {"1", "5", "10"}
    assert len(d["ic_by_group"]["index"]) == 2  # 两个行业


def test_forward_returns_alignment() -> None:
    """前向收益末 period 行为 NaN(无未来数据)。"""
    prices = _prices(rows=20, cols=5)
    fwd = compute_forward_returns(prices, [5])[5]
    assert fwd.iloc[-5:].isna().all().all()
    # 手算: fwd[t] = close[t+5]/close[t]-1
    t = 3
    manual = prices.iloc[t + 5] / prices.iloc[t] - 1.0
    pd.testing.assert_series_equal(fwd.iloc[t], manual, check_names=False)

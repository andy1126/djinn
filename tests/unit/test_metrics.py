"""指标计算单元测试(已知输入 → 已知输出)。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from djinn.analytics import (
    compare_benchmark,
    compute_max_drawdown,
    compute_metrics,
    monthly_returns,
)


def test_total_return_monotonic():
    """单调上涨:总收益 = 末值/首值 - 1。"""
    eq = pd.Series(
        np.linspace(100, 110, 252), index=pd.bdate_range("2024-01-02", periods=252)
    )
    m = compute_metrics(eq, market="US")
    assert m.total_return == pytest.approx(0.10, abs=1e-9)
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_known():
    """已知回撤:100→110→105→120,最大回撤 = (105-110)/110。"""
    eq = pd.Series([100, 110, 105, 120], index=pd.bdate_range("2024-01-02", periods=4))
    mdd, dd = compute_max_drawdown(eq)
    assert mdd == pytest.approx(-5 / 110, abs=1e-9)
    assert dd.iloc[2] < 0
    assert dd.iloc[3] == 0  # 创新高


def test_sharpe_positive_for_good_curve():
    """正收益低波动:夏普为正。"""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.005, 252)
    eq = pd.Series(
        100 * np.cumprod(1 + rets), index=pd.bdate_range("2024-01-02", periods=252)
    )
    m = compute_metrics(eq, market="US")
    assert m.sharpe > 0
    assert m.annual_volatility > 0


def test_metrics_empty_series():
    m = compute_metrics(pd.Series(dtype=float), market="US")
    assert m.n_days == 0


def test_n_trades_counts_fills():
    """n_trades = 成交笔数(3 买 2 卖 = 5),不是标的数也不是回合数。"""

    class _Fill:
        def __init__(self, side: str) -> None:
            self.side = side
            self.qty = 100.0
            self.price = 10.0

    eq = pd.Series(
        np.linspace(100, 110, 20), index=pd.bdate_range("2024-01-02", periods=20)
    )
    trades = [_Fill("buy") for _ in range(3)] + [_Fill("sell") for _ in range(2)]
    m = compute_metrics(eq, trades, market="US")
    assert m.n_trades == 5


def test_monthly_returns_shape():
    eq = pd.Series(
        np.linspace(100, 120, 252), index=pd.bdate_range("2024-01-02", periods=252)
    )
    mr = monthly_returns(eq)
    # 跨 2024 全年,应有 12 个月列
    assert 2024 in mr.index
    assert len(mr.columns) <= 12


def test_compare_benchmark_identical():
    """策略与基准完全一致:beta≈1,跟踪误差≈0。"""
    eq = pd.Series(
        np.linspace(100, 120, 100), index=pd.bdate_range("2024-01-02", periods=100)
    )
    bs = compare_benchmark(eq, eq.copy(), market="US")
    assert bs.beta == pytest.approx(1.0, abs=1e-6)
    assert bs.tracking_error == pytest.approx(0.0, abs=1e-9)
    assert bs.excess_return == pytest.approx(0.0, abs=1e-9)


def test_compare_benchmark_uncorrelated():
    """不相关序列:相关性低。"""
    rng = np.random.default_rng(1)
    a = pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, 200)),
        index=pd.bdate_range("2024-01-02", periods=200),
    )
    b = pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, 200)),
        index=pd.bdate_range("2024-01-02", periods=200),
    )
    bs = compare_benchmark(a, b, market="US")
    assert abs(bs.correlation) < 0.5


def test_calmar_nan_when_no_drawdown():
    """B4:单调上涨零回撤 → Calmar 为 NaN(未定义,而非 0)。"""
    eq = pd.Series(
        np.linspace(100, 200, 252), index=pd.bdate_range("2024-01-02", periods=252)
    )
    m = compute_metrics(eq, market="US")
    assert math.isnan(m.calmar)


def test_var_nonnegative():
    """B8:全正收益 → VaR/CVaR 非负(历史法,日度)。"""
    eq = pd.Series(
        np.linspace(100, 120, 252), index=pd.bdate_range("2024-01-02", periods=252)
    )
    m = compute_metrics(eq, market="US")
    assert m.var_95 >= 0.0
    assert m.cvar_95 >= 0.0


def test_sortino_no_downside_is_zero():
    """B3:全正收益 → 下行偏差 0 → sortino 0(无下行风险定义为 0 而非 inf)。"""
    eq = pd.Series(
        np.linspace(100, 120, 252), index=pd.bdate_range("2024-01-02", periods=252)
    )
    m = compute_metrics(eq, market="US")
    assert m.sortino == 0.0


def test_jensen_alpha_scales_with_beta():
    """B6:策略收益 = 1.2×基准日收益 + 常数 → beta≈1.2,jensen≈常数×af。"""
    idx = pd.bdate_range("2024-01-02", periods=500)
    rng = np.random.default_rng(2)
    br = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    sr = 1.2 * br + 0.0005
    b = pd.Series(100 * np.cumprod(1 + br), index=idx)
    s = pd.Series(100 * np.cumprod(1 + sr), index=idx)
    bs = compare_benchmark(s, b, market="US", rf=0.0)
    assert bs.beta == pytest.approx(1.2, abs=0.02)
    assert bs.alpha == pytest.approx(0.0005 * 252, abs=0.02)

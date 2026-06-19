"""指标计算单元测试(已知输入 → 已知输出)。"""

from __future__ import annotations

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

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


def test_sortino_standard():
    """B3:索提诺标准口径(MAR=rf/af,全样本下行偏差)——与手算公式逐位一致。"""
    rets = pd.Series(
        [0.01, -0.02, 0.03, -0.01, 0.02], index=pd.bdate_range("2024-01-02", periods=5)
    )
    eq = 100 * (1 + rets).cumprod()
    m = compute_metrics(eq, market="US", rf=0.0)
    af = 252.0
    # compute_metrics 的收益序列 = equity.pct_change() 去首行(NaN)
    excess = eq.pct_change().dropna().to_numpy()  # rf=0 → mar=0
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt((downside**2).mean()) * np.sqrt(af)
    expected = float(excess.mean() * af / downside_dev)
    assert m.sortino == pytest.approx(expected, rel=1e-6)


def test_monthly_first_month_present():
    """B5:首月收益(期初→首月末)不再被 dropna 丢掉。"""
    idx = pd.bdate_range("2020-01-01", periods=23)  # 覆盖 2020-01 整月(1/1~1/31)
    vals = np.linspace(100000, 105000, 23)
    eq = pd.Series(vals, index=idx)
    m = monthly_returns(eq)
    assert 2020 in m.index, "首月行不应被 dropna 丢掉"
    first = m.loc[2020].dropna().iloc[0]
    assert first == pytest.approx(0.05)  # 105000/100000 − 1


def test_upside_capture():
    """B6:策略与基准几乎同步 → upside_capture ≈1。"""
    idx = pd.bdate_range("2024-01-02", periods=400)
    rng = np.random.default_rng(1)
    br = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    sr = br + rng.normal(0, 0.001, 400)
    b = pd.Series(100 * np.cumprod(1 + br), index=idx)
    s = pd.Series(100 * np.cumprod(1 + sr), index=idx)
    bs = compare_benchmark(s, b, market="US", rf=0.0)
    assert bs.upside_capture == pytest.approx(1.0, rel=0.1)


def test_turnover_annual():
    """B8:turnover_annual 字段存在,无成交时为 0,有成交时为正。"""
    idx = pd.bdate_range("2024-01-02", periods=60)
    eq = pd.Series(100 * np.exp(np.cumsum(np.full(60, 0.001))), index=idx)
    m0 = compute_metrics(eq, market="US")
    assert m0.turnover_annual == 0.0
    # 构造两笔成交(买卖各一,单边成交额=市值一半)
    from datetime import date

    from djinn.engine.events import Fill

    fills = [
        Fill(1, date(2024, 2, 1), "S", "buy", 50.0, 100.0, 0.0),
        Fill(2, date(2024, 2, 2), "S", "sell", 50.0, 100.0, 0.0),
    ]
    m1 = compute_metrics(eq, market="US", trades=fills)
    assert m1.turnover_annual > 0.0
    # 双边换手 = 2×(50×100)/平均净值;turnover_annual = turnover×af/n/2
    avg_eq = float(eq.mean())
    assert m1.turnover == pytest.approx(2 * 5000.0 / avg_eq, rel=1e-6)
    assert m1.turnover_annual == pytest.approx(
        m1.turnover * 252.0 / len(eq) / 2.0, rel=1e-6
    )


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

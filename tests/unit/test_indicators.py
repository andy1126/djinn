"""指标库单测:sma/ema/rsi/macd/bb/交叉/变化率等对标已知数值。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from djinn import indicators


def test_sma():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = indicators.sma(s, 3)
    assert np.isnan(out.iloc[0])
    assert out.iloc[2] == 2.0  # (1+2+3)/3
    assert out.iloc[4] == 4.0  # (3+4+5)/3


def test_ema_rma_have_values():
    s = pd.Series(range(1, 31), dtype=float)
    for f in (indicators.ema, indicators.rma):
        out = f(s, 5)
        assert len(out) == len(s)
        assert not out.dropna().empty


def test_cross_over_under():
    a = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    b = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    assert list(indicators.cross_over(a, b)) == [False, False, True, False, False]
    assert list(indicators.cross_under(a, b)) == [False, False, False, False, True]


def test_change_roc_momentum():
    s = pd.Series([10.0, 11.0, 13.0])
    assert indicators.change(s, 1).iloc[1] == 1.0
    assert indicators.momentum(s, 1).iloc[2] == 2.0
    assert abs(indicators.roc(s, 1).iloc[1] - 10.0) < 1e-9  # (11/10-1)*100


def test_rsi_bounds():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(0, 1, 200))
    r = indicators.rsi(s, 14)
    assert r.notna().all()
    assert ((r >= 0) & (r <= 100)).all()


def test_dataframe_indicators_columns():
    idx = pd.date_range("2024-01-01", periods=60)
    close = pd.Series(np.linspace(10, 20, 60), index=idx)
    high = close + 1
    low = close - 1
    assert set(indicators.macd(close).columns) == {"macd", "signal", "hist"}
    assert set(indicators.bb(close, 20).columns) == {"upper", "mid", "lower"}
    assert set(indicators.donchian(high, low, 20).columns) == {"upper", "lower"}
    assert set(indicators.stoch(high, low, close).columns) == {"k", "d"}


def test_highest_lowest_atr():
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0])
    high = close + 1
    low = close - 1
    assert indicators.highest(close, 3).iloc[4] == 13.0
    assert indicators.lowest(close, 3).iloc[4] == 11.0
    atr = indicators.atr(high, low, close, 3)
    assert len(atr) == len(close)


def test_barssince():
    cond = pd.Series([False, True, False, False, True])
    out = indicators.barssince(cond)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 0
    assert out.iloc[3] == 2
    assert out.iloc[4] == 0


def test_indicator_specs():
    specs = indicators.indicator_specs()
    assert {s["name"] for s in specs} == set(indicators.__all__)
    for s in specs:
        assert s["category"]
        assert s["signature"]
        assert s["description"]
        assert s["source"]
    rsi = next(s for s in specs if s["name"] == "rsi")
    assert rsi["signature"].startswith("rsi(close")

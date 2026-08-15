"""指标库单测:sma/ema/rsi/macd/bb/交叉/变化率等对标已知数值。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_rsi_all_up():
    """连续上涨窗口:avg_loss==0 → RSI=100(而非 fillna 的 50)。"""
    s = pd.Series(np.linspace(10.0, 30.0, 30))  # 单调上涨
    r = indicators.rsi(s, 14)
    assert r.iloc[-1] == 100.0


def test_rsi_all_down():
    """连续下跌窗口:avg_gain==0 → RSI=0。"""
    s = pd.Series(np.linspace(30.0, 10.0, 30))
    r = indicators.rsi(s, 14)
    assert r.iloc[-1] == 0.0


def test_rsi_flat():
    """常数序列(平盘):双 0 → RSI=50。"""
    s = pd.Series([10.0] * 30)
    r = indicators.rsi(s, 14)
    assert r.iloc[-1] == 50.0


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


def test_tr():
    high = pd.Series([11.0, 13.0, 12.0, 14.0])
    low = pd.Series([9.0, 10.0, 11.0, 12.0])
    close = pd.Series([10.0, 12.0, 11.0, 13.0])
    t = indicators.tr(high, low, close)
    assert t.iloc[0] == 2.0  # 首根无 prev_close,取 high-low
    assert t.iloc[1] == 3.0  # max(3, |13-10|=3, |10-10|=0)
    assert t.iloc[2] == 1.0  # max(1, |12-12|=0, |11-12|=1)
    assert t.iloc[3] == 3.0  # max(2, |14-11|=3, |12-11|=1)


def test_vwma_equal_volume_is_mean():
    close = pd.Series([10.0, 11.0, 12.0])
    volume = pd.Series([100.0, 100.0, 100.0])
    out = indicators.vwma(close, volume, 3)
    assert abs(out.iloc[2] - 11.0) < 1e-9


def test_hma_finite():
    s = pd.Series(range(1, 41), dtype=float)
    out = indicators.hma(s, 16)
    assert len(out) == len(s)
    assert not out.dropna().empty


def test_mfi_wpr_bounds():
    rng = np.random.default_rng(2)
    close = pd.Series(rng.normal(100, 5, 300)).cumsum()
    high = close + 1
    low = close - 1
    volume = pd.Series(rng.uniform(1000, 5000, 300))
    mfi = indicators.mfi(high, low, close, volume, 14)
    assert mfi.dropna().between(0, 100).all()
    wpr = indicators.wpr(high, low, close, 14)
    assert wpr.dropna().between(-100, 0).all()


def test_aroon_duplicate_extreme_uses_last():
    """Aroon 重复极值取最近一次(旧 argmax 取首次导致数值偏低)。"""
    idx = pd.bdate_range("2024-01-01", periods=5)
    # 末日窗口 [10,5,10,8]:最高 10 出现在 idx0(首次)与 idx2(最近)
    high = pd.Series([1, 10, 5, 10, 8], index=idx)
    out = indicators.aroon(high, high, 4)
    # 距最近一次极值 = 1 期 → aroon_up = 100×(4-1)/4 = 75
    assert out["aroon_up"].iloc[-1] == pytest.approx(75.0)
    # 无重复窗口 [5,10,8,7] → 距 10 = 2 期 → 50
    high2 = pd.Series([1, 5, 10, 8, 7], index=idx)
    assert indicators.aroon(high2, high2, 4)["aroon_up"].iloc[-1] == pytest.approx(50.0)


def test_mfi_all_inflow_is_100():
    """MFI 全流入 → 100(旧实现 neg_sum=0 时返回 NaN),全流出 → 0。"""
    idx = pd.bdate_range("2024-01-01", periods=8)
    v = pd.Series([100.0] * 8, index=idx)
    up = pd.Series([10, 11, 12, 13, 14, 15, 16, 17], index=idx)
    assert indicators.mfi(up, up - 0.1, up, v, 4).iloc[-1] == pytest.approx(100.0)
    down = pd.Series([17, 16, 15, 14, 13, 12, 11, 10], index=idx)
    assert indicators.mfi(down, down - 0.1, down, v, 4).iloc[-1] == pytest.approx(0.0)
    # 价格横盘(无流向)→ 中性 50(非 NaN)
    flat = pd.Series([10.0] * 8, index=idx)
    assert indicators.mfi(flat, flat - 0.1, flat, v, 4).iloc[-1] == pytest.approx(50.0)


def test_dmi_aroon_kc_columns():
    idx = pd.date_range("2024-01-01", periods=80)
    close = pd.Series(np.linspace(10, 20, 80), index=idx)
    high = close + 1
    low = close - 1
    assert set(indicators.dmi(high, low, close).columns) == {
        "plus_di",
        "minus_di",
        "adx",
    }
    aroon = indicators.aroon(high, low)
    assert set(aroon.columns) == {"aroon_up", "aroon_down"}
    assert ((aroon.dropna() >= 0) & (aroon.dropna() <= 100)).all().all()
    assert set(indicators.kc(high, low, close).columns) == {"mid", "upper", "lower"}


def test_rising_falling():
    s = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    assert list(indicators.rising(s, 1)) == [False, True, True, False, False]
    assert list(indicators.falling(s, 1)) == [False, False, False, True, True]


def test_supertrend_uptrend():
    # atr_period=1 → ATR=TR=2(逐根 high-low),factor=1 → 上下轨 ±2,hl2 整数。
    high = pd.Series([12.0, 13.0, 14.0, 15.0])
    low = pd.Series([10.0, 11.0, 12.0, 13.0])
    close = pd.Series([11.0, 12.0, 13.0, 14.0])
    out = indicators.supertrend(high, low, close, factor=1.0, atr_period=1)
    assert list(out["supertrend"]) == [9.0, 10.0, 11.0, 12.0]
    assert list(out["direction"]) == [1, 1, 1, 1]


def test_supertrend_downtrend_flips():
    idx = pd.date_range("2024-01-01", periods=80)
    close = pd.Series(np.linspace(50.0, 10.0, 80), index=idx)
    high = close + 1
    low = close - 1
    out = indicators.supertrend(high, low, close, factor=3.0, atr_period=5)
    assert out["direction"].iloc[-1] == -1
    assert out["supertrend"].iloc[-1] > close.iloc[-1]


def test_psar_uptrend_anchor():
    high = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    low = pd.Series([9.0, 10.0, 11.0, 12.0, 13.0])
    sar = indicators.psar(high, low, 0.02, 0.02, 0.2)
    assert sar.iloc[0] == 9.0
    assert abs(sar.iloc[1] - 9.02) < 1e-9
    assert sar.iloc[2] == 9.0  # 被钳制到前低
    assert abs(sar.iloc[3] - 9.18) < 1e-9
    assert abs(sar.iloc[4] - 9.4856) < 1e-9


def test_psar_below_price_in_uptrend():
    idx = pd.date_range("2024-01-01", periods=80)
    high = pd.Series(np.linspace(10.0, 40.0, 80), index=idx)
    low = high - 2
    sar = indicators.psar(high, low)
    assert sar.notna().all()
    assert (sar < high).all()  # 强多头无翻转,SAR 始终在最高价之下


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

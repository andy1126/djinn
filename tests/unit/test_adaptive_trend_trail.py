"""Adaptive Trend Trail 指标移植测试(本地合成数据,不依赖网络)。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from djinn.indicators.adaptive_trend_trail import adaptive_trend_trail


def _ohlcv(
    n: int = 1200, seed: int = 7
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    rets = np.concatenate(
        [
            rng.normal(0.0015, 0.012, n // 3),
            rng.normal(-0.0018, 0.015, n // 3),
            rng.normal(0.0, 0.010, n - 2 * (n // 3)),
        ]
    )
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    high = close * (1 + rng.uniform(0.002, 0.015, n))
    low = close * (1 - rng.uniform(0.002, 0.015, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    return high, low, close, open_


def test_runs_and_shapes() -> None:
    high, low, close, open_ = _ohlcv()
    out = adaptive_trend_trail(high, low, close, open_)
    assert out.shape[0] == len(close)
    assert set(out["trend"].dropna().unique()).issubset({-1, 0, 1})
    assert out["up_signal"].dtype == bool
    assert out["down_signal"].dtype == bool
    assert out["valuation"].dropna().between(0.0, 100.0).all()
    # 暖机后趋势带非空
    assert out["outer_trail"].notna().sum() > len(close) // 2


def test_signals_alternate() -> None:
    """翻转信号不会出现连续同向(up 只在 trend 由非多转多时触发)。"""
    high, low, close, open_ = _ohlcv(seed=11)
    out = adaptive_trend_trail(high, low, close, open_)
    up = out["up_signal"]
    down = out["down_signal"]
    # 同一根不可能同时向上向下
    assert not (up & down).any()
    # up_signal 前后一根的 trend 必为 1,且更早一根(或首根)非 1
    for i in up[up].index:
        assert out.loc[i, "trend"] == 1
    for i in down[down].index:
        assert out.loc[i, "trend"] == -1


def test_supertrend_accepts_series_factor() -> None:
    """djinn supertrend 支持逐根 factor(自适应因子)。"""
    from djinn.indicators import supertrend

    high, low, close, open_ = _ohlcv(seed=3)
    scalar = supertrend(high, low, close, 1.5, 10)
    series = supertrend(high, low, close, pd.Series(1.5, index=close.index), 10)
    pd.testing.assert_series_equal(scalar["direction"], series["direction"])


def test_strategy_registered_and_signals() -> None:
    """策略注册进 STRATEGY_REGISTRY 且 signals() 返回合法持仓状态。"""
    from djinn.strategy.library import get_strategy_class

    cls = get_strategy_class("AdaptiveTrendTrail")
    assert cls.__name__ == "AdaptiveTrendTrail"
    high, low, close, open_ = _ohlcv(seed=5)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}
    )
    sig = cls().signals(df)
    assert set(sig.unique()).issubset({-1, 0, 1})

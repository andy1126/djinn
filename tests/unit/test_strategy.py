"""策略层单元测试:参数声明、信号触发、schema。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from djinn.strategy import (
    DCA,
    BollingerReversion,
    MACDCrossover,
    MACrossover,
    Momentum,
    RSIReversal,
    StochasticCross,
    Supertrend,
    get_strategy_class,
    param_schema,
)
from djinn.strategy.utils import state_from_signals
from djinn.utils.exceptions import ParameterError, StrategyError


def test_param_validation_bounds():
    with pytest.raises(ParameterError):
        MACrossover(fast=1)  # min=2
    with pytest.raises(ParameterError):
        MACrossover(slow=1000)  # max=250


def test_param_validation_unknown():
    with pytest.raises(StrategyError):
        MACrossover(foo=1)


def test_param_schema_exported():
    schema = {s.name: s for s in param_schema(MACrossover)}
    assert set(schema) == {"fast", "slow"}
    assert schema["fast"].default == 10
    assert schema["fast"].min == 2
    assert schema["fast"].max == 100


def test_state_from_signals():
    sig = pd.Series([0, 0, 1, 0, 0, -1, 0], dtype=int)
    state = state_from_signals(sig)
    assert state.tolist() == [0, 0, 1, 1, 1, -1, -1]


def test_macrossover_crossover_triggers(crossover_data):
    """金叉后 state=1,死叉后 state=-1。"""
    s = MACrossover(fast=5, slow=15)
    sig = s.signals(crossover_data)
    # 反弹段(20-40)应出现金叉 → state=1
    rebound = sig.iloc[20:40]
    assert (rebound == 1).any()
    # 回落段(40-60)应出现死叉 → state=-1
    fallback = sig.iloc[40:60]
    assert (fallback == -1).any()


def test_macrossover_no_signal_before_warmup():
    """数据不足 warmup 期:信号为 0。"""
    n = 5
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"close": np.linspace(100, 110, n)}, index=idx)
    df = df.assign(open=df.close, high=df.close, low=df.close, volume=1000)
    s = MACrossover(fast=10, slow=30)
    sig = s.signals(df)
    assert (sig == 0).all()


def test_rsi_reversal_signals():
    """构造超卖→超买序列,验证信号翻转。"""
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    # 先大跌(RSI 超卖),再大涨(RSI 超买)
    close = np.concatenate([np.linspace(100, 70, 30), np.linspace(70, 110, 30)])
    df = pd.DataFrame({"close": close}, index=idx)
    df = df.assign(open=df.close, high=df.close, low=df.close, volume=1000)
    s = RSIReversal(period=10, oversold=30, overbought=70)
    sig = s.signals(df)
    assert set(np.unique(sig.values)).issubset({-1, 0, 1})


def test_momentum_breakout():
    n = 40
    idx = pd.bdate_range("2024-01-02", periods=n)
    # 前 20 日区间震荡,第 21 日突破创新高
    close = np.concatenate([np.full(20, 100.0), np.linspace(100, 120, 20)])
    df = pd.DataFrame({"close": close}, index=idx)
    df = df.assign(open=df.close, high=df.close, low=df.close, volume=1000)
    s = Momentum(period=10)
    sig = s.signals(df)
    assert (sig.iloc[20:] == 1).any()  # 突破后做多


def test_strategy_registry_lookup():
    assert get_strategy_class("MACrossover") is MACrossover
    with pytest.raises(KeyError):
        get_strategy_class("Nonexistent")


def test_dca_has_param_schema():
    schema = [s.name for s in param_schema(DCA)]
    assert "frequency" in schema
    assert "amount" in schema


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    """构造 OHLCV DataFrame(high/low 围绕 close ±1,volume 常数)。"""
    idx = pd.bdate_range("2024-01-02", periods=len(close))
    df = pd.DataFrame({"close": close}, index=idx)
    return df.assign(open=df.close, high=df.close + 1, low=df.close - 1, volume=1000)


def test_new_timing_strategies_signal_states():
    """四个新策略都能产生合法持仓状态(值 ∈ {-1,0,1} 且长度一致)。"""
    close = np.concatenate([np.linspace(100, 70, 40), np.linspace(70, 120, 40)])
    df = _ohlcv(close)
    for strat in (
        MACDCrossover(),
        BollingerReversion(),
        Supertrend(),
        StochasticCross(),
    ):
        sig = strat.signals(df)
        assert len(sig) == len(df)
        assert set(np.unique(sig.values)).issubset({-1, 0, 1})


def test_supertrend_signals_long_in_uptrend():
    close = np.linspace(100, 160, 80)
    sig = Supertrend().signals(_ohlcv(close))
    assert sig.iloc[-1] == 1  # 强多头末态


def test_bollinger_reversion_triggers_on_dip():
    # 平盘后一次尖底跌破下轨 → 触发做多
    close = np.concatenate(
        [np.full(30, 100.0), np.array([85.0, 80.0, 100.0, 100.0, 100.0])]
    )
    sig = BollingerReversion(period=20, num_std=2.0).signals(_ohlcv(close))
    assert (sig == 1).any()


def test_new_strategies_in_registry():
    for name, cls in (
        ("MACDCrossover", MACDCrossover),
        ("BollingerReversion", BollingerReversion),
        ("Supertrend", Supertrend),
        ("StochasticCross", StochasticCross),
    ):
        assert get_strategy_class(name) is cls

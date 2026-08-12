"""Pine Script 转译单测:受支持子集 → djinn Python → 编译 → 运行。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from djinn.strategy.parameter import param_schema
from djinn.strategy.pine import pine_to_python
from djinn.strategy.user import compile_user_strategy
from djinn.utils.exceptions import StrategyError


def _ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1e6),
        }
    )


SMA_CROSS_PINE = """
//@version=5
strategy("My SMA Cross", overlay=true)
fastLen = input.int(10, "Fast", minval=2, maxval=100)
slowLen = input.int(30, "Slow")
fast = ta.sma(close, fastLen)
slow = ta.sma(close, slowLen)
longCondition = ta.crossover(fast, slow)
shortCondition = ta.crossunder(fast, slow)
if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.close("Long")
"""


def test_sma_cross_roundtrip():
    py = pine_to_python(SMA_CROSS_PINE)
    assert "fastLen = param(10, min=2, max=100, description='Fast')" in py
    assert "def signals(self, data):" in py
    assert "cross_over(fast, slow)" in py

    cls = compile_user_strategy("MySMACross", py, "python")
    assert [p.name for p in param_schema(cls)] == ["fastLen", "slowLen"]
    sig = cls().signals(_ohlcv())
    assert set(sig.unique()).issubset({-1, 0, 1})


def test_rsi_mapping_and_float_input():
    pine = """
//@version=5
strategy("RSI")
len = input.int(14, "Len")
thr = input.float(70.0, "Overbought")
r = ta.rsi(close, len)
if (ta.crossunder(r, thr))
    strategy.close("Long")
if (ta.crossunder(r, 100 - thr))
    strategy.entry("Long", strategy.long)
"""
    py = pine_to_python(pine)
    assert "rsi(close, self.len)" in py
    assert "thr = param(70.0, description='Overbought')" in py
    cls = compile_user_strategy("RSI", py, "python")
    cls().signals(_ohlcv())


def test_nz_and_math_and_history_ref():
    pine = """
//@version=5
strategy("X")
src = ta.sma(close, 20)
prev = src[1]
chg = math.abs(close - nz(prev, close))
if (chg > 5)
    strategy.entry("L", strategy.long)
"""
    py = pine_to_python(pine)
    assert "src.shift(1)" in py
    assert ".fillna(" in py
    assert "abs(" in py
    compile_user_strategy("X", py, "python")


def test_unsupported_constructs_raise():
    cases = [
        # strategy.exit
        """
//@version=5
strategy("X")
if (ta.crossover(ta.sma(close, 10), ta.sma(close, 20)))
    strategy.entry("L", strategy.long)
strategy.exit("L", stop=close * 0.9)
""",
        # tuple unpacking
        """
//@version=5
strategy("X")
[macdLine, signalLine, _] = ta.macd(close, 12, 26, 9)
if (macdLine > signalLine)
    strategy.entry("L", strategy.long)
""",
        # short
        """
//@version=5
strategy("X")
if (ta.crossunder(ta.sma(close, 10), ta.sma(close, 20)))
    strategy.entry("S", strategy.short)
""",
    ]
    for pine in cases:
        try:
            pine_to_python(pine)
            raise AssertionError("should reject unsupported Pine")
        except StrategyError:
            pass


def test_no_signals_raises():
    try:
        pine_to_python('//@version=5\nstrategy("X")\nma = ta.sma(close, 20)\n')
        raise AssertionError("should require strategy.entry/close")
    except StrategyError:
        pass

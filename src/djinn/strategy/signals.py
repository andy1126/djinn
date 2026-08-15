"""信号指标注册表:把「OHLCV → 稀疏信号」的指标统一接入通用策略。

约定:每个信号函数 ``fn(data, **params) -> pd.Series``,``data`` 为单标的 OHLCV
DataFrame(列 ``open/high/low/close/volume``),返回稀疏 ``{-1, 0, 1}``:
- ``+1`` = 做多事件;
- ``-1`` = 平仓事件;
- ``0``   = 无事件。

通用 :class:`~djinn.strategy.library.signal_strategy.SignalStrategy` 会经
:func:`~djinn.strategy.utils.state_from_signals` 把稀疏信号补成持仓状态
(1 持多 / -1 空仓 / 0 中性)。新增信号指标只需 ``@register_signal_indicator``
注册即可,无需再写策略类。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from djinn.indicators import sma, supertrend
from djinn.indicators.adaptive_trend_trail import adaptive_trend_trail
from djinn.utils.exceptions import StrategyError

# 信号指标名 → 函数。
SIGNAL_INDICATORS: dict[str, Callable[..., pd.Series]] = {}


def register_signal_indicator(
    name: str,
) -> Callable[[Callable[..., pd.Series]], Callable[..., pd.Series]]:
    """装饰器:把 ``fn(data, **params) -> Series`` 注册为信号指标。"""

    def deco(fn: Callable[..., pd.Series]) -> Callable[..., pd.Series]:
        SIGNAL_INDICATORS[name] = fn
        return fn

    return deco


def get_signal_indicator(name: str) -> Callable[..., pd.Series]:
    """按名取信号函数;未知抛 StrategyError(附可用列表)。"""
    if name not in SIGNAL_INDICATORS:
        raise StrategyError(f"未知信号指标 {name!r},可用: {sorted(SIGNAL_INDICATORS)}")
    return SIGNAL_INDICATORS[name]


@register_signal_indicator("supertrend")
def supertrend_signal(
    data: pd.DataFrame, *, factor: float = 3.0, atr_period: int = 10
) -> pd.Series:
    """Supertrend 方向:翻多 +1、翻空 -1。"""
    d = supertrend(
        data["high"], data["low"], data["close"], float(factor), int(atr_period)
    )["direction"]
    sig = pd.Series(0, index=data.index, dtype=int)
    sig[d == 1] = 1
    sig[d == -1] = -1
    return sig


@register_signal_indicator("adaptive_trend_trail")
def adaptive_trend_trail_signal(data: pd.DataFrame, **params: object) -> pd.Series:
    """Adaptive Trend Trail 趋势翻转:up_signal +1、down_signal -1。

    参数透传给 :func:`adaptive_trend_trail`(trend_length/momentum_length/
    sensitivity/st_*_length/st_*_factor)。
    """
    out = adaptive_trend_trail(
        data["high"], data["low"], data["close"], data["open"], **params  # type: ignore[arg-type]
    )
    sig = pd.Series(0, index=data.index, dtype=int)
    sig[out["up_signal"]] = 1
    sig[out["down_signal"]] = -1
    return sig


@register_signal_indicator("ma_cross")
def ma_cross_signal(data: pd.DataFrame, *, fast: int = 10, slow: int = 30) -> pd.Series:
    """双均线交叉:快线上穿慢线 +1、下穿 -1。"""
    ma_fast = sma(data["close"], int(fast))
    ma_slow = sma(data["close"], int(slow))
    up = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    down = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    sig = pd.Series(0, index=data.index, dtype=int)
    sig[up] = 1
    sig[down] = -1
    return sig

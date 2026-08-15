"""Adaptive Trend Trail 趋势跟踪策略(信号由同名指标生成)。"""

from __future__ import annotations

import pandas as pd

from djinn.indicators.adaptive_trend_trail import adaptive_trend_trail
from djinn.strategy.base import Strategy, param
from djinn.strategy.utils import state_from_signals


class AdaptiveTrendTrail(Strategy):
    """Adaptive Trend Trail 趋势跟踪(仅做多)。

    - ``up_signal``(趋势翻多)→ 做多
    - ``down_signal``(趋势翻空)→ 平仓

    底层为自适应 Supertrend 矩阵 + 复合 regime 状态机(见
    :func:`djinn.indicators.adaptive_trend_trail.adaptive_trend_trail`)。
    """

    trend_length = param(34, min=10, max=200, description="趋势基准 EMA 周期")
    momentum_length = param(12, min=3, max=100, description="动量回看周期")
    sensitivity = param(
        0.35, min=0.10, max=1.25, description="信号选择性(越高信号越少)"
    )
    st_fast_length = param(9, min=2, max=100, description="快 Supertrend ATR 周期")
    st_fast_factor = param(
        1.45, min=0.25, max=10.0, description="快 Supertrend ATR 倍数"
    )
    st_mid_length = param(14, min=2, max=150, description="中 Supertrend ATR 周期")
    st_mid_factor = param(
        1.95, min=0.25, max=10.0, description="中 Supertrend ATR 倍数"
    )
    st_slow_length = param(21, min=2, max=200, description="慢 Supertrend ATR 周期")
    st_slow_factor = param(
        2.55, min=0.25, max=10.0, description="慢 Supertrend ATR 倍数"
    )

    def signals(self, data: pd.DataFrame) -> pd.Series:
        out = adaptive_trend_trail(
            data["high"],
            data["low"],
            data["close"],
            data["open"],
            trend_length=int(self.trend_length),
            momentum_length=int(self.momentum_length),
            sensitivity=float(self.sensitivity),
            st_fast_length=int(self.st_fast_length),
            st_fast_factor=float(self.st_fast_factor),
            st_mid_length=int(self.st_mid_length),
            st_mid_factor=float(self.st_mid_factor),
            st_slow_length=int(self.st_slow_length),
            st_slow_factor=float(self.st_slow_factor),
        )
        sig = pd.Series(0, index=data.index, dtype=int)
        sig[out["up_signal"]] = 1
        sig[out["down_signal"]] = -1
        return state_from_signals(sig)

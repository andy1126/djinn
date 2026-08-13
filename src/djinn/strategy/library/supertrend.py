"""Supertrend 趋势跟踪策略:方向翻多持有,翻空平仓。"""

from __future__ import annotations

import pandas as pd

from djinn.indicators import supertrend
from djinn.strategy.base import Strategy, param
from djinn.strategy.utils import state_from_signals


class Supertrend(Strategy):
    """Supertrend 趋势跟踪。

    - direction 翻 +1(多头)→ 做多(+1)
    - direction 翻 -1(空头)→ 平仓(-1)
    """

    factor = param(3.0, min=1.0, max=10.0, description="ATR 倍数")
    atr_period = param(10, min=2, max=100, description="ATR 周期")

    def signals(self, data: pd.DataFrame) -> pd.Series:
        st = supertrend(
            data["high"],
            data["low"],
            data["close"],
            float(self.factor),
            int(self.atr_period),
        )
        direction = st["direction"]
        sig = pd.Series(0, index=data.index, dtype=int)
        sig[direction == 1] = 1
        sig[direction == -1] = -1
        return state_from_signals(sig)

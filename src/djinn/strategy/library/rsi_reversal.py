"""RSI 超买超卖反转策略。"""

from __future__ import annotations

import pandas as pd

from djinn.indicators import rsi
from djinn.strategy.base import Strategy, param
from djinn.strategy.utils import state_from_signals


class RSIReversal(Strategy):
    """RSI 超买超卖反转。

    - RSI < oversold → 超卖,做多(+1)
    - RSI > overbought → 超买,平仓(-1)
    """

    period = param(14, min=2, max=100, description="RSI 周期")
    oversold = param(30, min=5, max=50, description="超卖阈值")
    overbought = param(70, min=50, max=95, description="超买阈值")

    def signals(self, data: pd.DataFrame) -> pd.Series:
        r = rsi(data["close"], int(self.period))
        sig = pd.Series(0, index=data.index, dtype=int)
        long_state = r < self.oversold
        exit_state = r > self.overbought
        sig[long_state] = 1
        sig[exit_state] = -1
        return state_from_signals(sig)

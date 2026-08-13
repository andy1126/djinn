"""MACD 金叉死叉策略:MACD 线上穿信号线做多,下穿平仓。"""

from __future__ import annotations

import pandas as pd

from djinn.indicators import macd
from djinn.strategy.base import Strategy, param
from djinn.strategy.utils import state_from_signals


class MACDCrossover(Strategy):
    """MACD 交叉。

    - MACD 线上穿信号线(金叉)→ 做多(+1)
    - MACD 线下穿信号线(死叉)→ 平仓(-1)
    """

    fast = param(12, min=2, max=100, description="快线 EMA 周期")
    slow = param(26, min=5, max=250, description="慢线 EMA 周期")
    signal = param(9, min=2, max=100, description="信号线周期")

    def signals(self, data: pd.DataFrame) -> pd.Series:
        m = macd(data["close"], int(self.fast), int(self.slow), int(self.signal))
        above = m["macd"] > m["signal"]
        crossed_up = above & ~above.shift(1, fill_value=False)
        crossed_down = ~above & above.shift(1, fill_value=False)
        sig = pd.Series(0, index=data.index, dtype=int)
        sig[crossed_up] = 1
        sig[crossed_down] = -1
        return state_from_signals(sig)

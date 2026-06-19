"""均线交叉策略:fast 上穿 slow 做多,下穿平仓。"""

from __future__ import annotations

import pandas as pd

from djinn.strategy.base import Strategy, param
from djinn.strategy.utils import state_from_signals


class MACrossover(Strategy):
    """双均线交叉(fast / slow)。

    - fast 上穿 slow → 做多(信号 +1)
    - fast 下穿 slow → 平仓(信号 -1)
    """

    fast = param(10, min=2, max=100, description="快速均线周期")
    slow = param(30, min=5, max=250, description="慢速均线周期")

    def signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ma_fast = close.rolling(int(self.fast)).mean()
        ma_slow = close.rolling(int(self.slow)).mean()
        above = ma_fast > ma_slow
        crossed_up = above & ~above.shift(1, fill_value=False)
        crossed_down = ~above & above.shift(1, fill_value=False)
        sig = pd.Series(0, index=close.index, dtype=int)
        sig[crossed_up] = 1
        sig[crossed_down] = -1
        return state_from_signals(sig)

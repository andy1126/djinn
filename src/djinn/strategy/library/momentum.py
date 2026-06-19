"""动量突破策略:价格突破 N 日高点做多,跌破 N 日低点平仓。"""

from __future__ import annotations

import pandas as pd

from djinn.strategy.base import Strategy, param
from djinn.strategy.utils import state_from_signals


class Momentum(Strategy):
    """N 日通道突破(Donchian 风格)。

    - 收盘 > 过去 ``period`` 日最高(不含当日) → 做多(+1)
    - 收盘 < 过去 ``period`` 日最低(不含当日) → 平仓(-1)
    """

    period = param(20, min=2, max=250, description="通道周期")

    def signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        # 用 shift(1) 排除当日,避免未来函数
        hh = close.rolling(int(self.period)).max().shift(1)
        ll = close.rolling(int(self.period)).min().shift(1)
        sig = pd.Series(0, index=close.index, dtype=int)
        sig[close > hh] = 1
        sig[close < ll] = -1
        return state_from_signals(sig)

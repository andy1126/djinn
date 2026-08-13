"""布林带均值回归策略:收盘跌破下轨做多,突破上轨平仓。"""

from __future__ import annotations

import pandas as pd

from djinn.indicators import bb
from djinn.strategy.base import Strategy, param
from djinn.strategy.utils import state_from_signals


class BollingerReversion(Strategy):
    """布林带均值回归。

    - 收盘跌破下轨(超卖)→ 做多(+1)
    - 收盘突破上轨(超买)→ 平仓(-1)
    """

    period = param(20, min=2, max=250, description="布林带周期")
    num_std = param(2.0, min=0.5, max=5.0, description="标准差倍数")

    def signals(self, data: pd.DataFrame) -> pd.Series:
        b = bb(data["close"], int(self.period), float(self.num_std))
        sig = pd.Series(0, index=data.index, dtype=int)
        sig[data["close"] < b["lower"]] = 1
        sig[data["close"] > b["upper"]] = -1
        return state_from_signals(sig)

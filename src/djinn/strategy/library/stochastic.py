"""随机指标金叉死叉策略:K 线上穿 D 线做多,下穿平仓。"""

from __future__ import annotations

import pandas as pd

from djinn.indicators import stoch
from djinn.strategy.base import Strategy, param
from djinn.strategy.utils import state_from_signals


class StochasticCross(Strategy):
    """随机指标(KD)交叉。

    - K 线上穿 D 线(金叉)→ 做多(+1)
    - K 线下穿 D 线(死叉)→ 平仓(-1)
    """

    k_period = param(14, min=2, max=100, description="%K 周期")
    d_period = param(3, min=1, max=50, description="%D 平滑周期")
    smooth = param(3, min=1, max=50, description="%K 平滑周期")

    def signals(self, data: pd.DataFrame) -> pd.Series:
        s = stoch(
            data["high"],
            data["low"],
            data["close"],
            int(self.k_period),
            int(self.d_period),
            int(self.smooth),
        )
        above = s["k"] > s["d"]
        crossed_up = above & ~above.shift(1, fill_value=False)
        crossed_down = ~above & above.shift(1, fill_value=False)
        sig = pd.Series(0, index=data.index, dtype=int)
        sig[crossed_up] = 1
        sig[crossed_down] = -1
        return state_from_signals(sig)

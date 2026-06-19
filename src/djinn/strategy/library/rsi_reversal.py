"""RSI 超买超卖反转策略。"""

from __future__ import annotations

import pandas as pd

from djinn.strategy.base import Strategy, param
from djinn.strategy.utils import state_from_signals


def rsi(close: pd.Series, period: int) -> pd.Series:
    """计算 RSI(Wilder 平滑)。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder 平滑:首个 period 用简单平均,后续用指数加权递推
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


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

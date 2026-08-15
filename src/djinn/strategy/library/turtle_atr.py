"""海龟/ATR 策略:Donchian 突破入场 + ATR 风险定仓。"""

from __future__ import annotations

import math

from djinn.indicators import atr as atr_ind
from djinn.strategy.base import Context, Strategy, param


class TurtleATR(Strategy):
    """海龟法则(Donchian 突破 + ATR 定仓)。

    - 收盘突破过去 ``entry`` 日高点 → 做多;
    - 收盘跌破过去 ``exit_`` 日低点 → 平仓;
    - 仓位按 ATR 定:权重 = ``risk_per_unit`` × price / ATR,
      波动越大仓位越小,上限 100%(经典海龟:单位 = 1% 净值 ÷ N,``N`` = ATR 值,
      ``atr_period`` 只是算 ATR 的窗口,不再乘进分母)。
    """

    entry = param(20, min=2, max=250, description="突破周期(入场)")
    exit_ = param(10, min=2, max=250, description="退出周期")
    atr_period = param(20, min=2, max=100, description="ATR 周期")
    risk_per_unit = param(0.01, min=0.001, max=0.05, description="每单位风险(净值占比)")

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self._long: dict[str, bool] = {}

    def on_bar(self, ctx: Context) -> None:
        for s in ctx.data.symbols:
            df = ctx.data[s]
            if len(df) < int(self.entry) + 1:
                continue
            close = df["close"]
            # 通道不含当日(shift 1),防未来函数
            hh = float(close.rolling(int(self.entry)).max().shift(1).iloc[-1])
            ll = float(close.rolling(int(self.exit_)).min().shift(1).iloc[-1])
            a = float(
                atr_ind(df["high"], df["low"], close, int(self.atr_period)).iloc[-1]
            )
            if math.isnan(hh) or math.isnan(ll) or math.isnan(a):
                continue
            price = float(close.iloc[-1])
            is_long = self._long.get(s, False)
            if not is_long and price > hh:
                self._long[s] = True
                # 经典海龟:weight = risk_per_unit × price / N(N = ATR 值)。
                # 旧实现多除一个 atr_period(ATR 已按该窗口算得),仓位被缩小 atr_period 倍。
                w = (
                    min(1.0, float(self.risk_per_unit) * price / a)
                    if a > 0
                    else 1.0
                )
                ctx.order_target_percent(s, w)
            elif is_long and price < ll:
                self._long[s] = False
                ctx.order_target_percent(s, 0.0)

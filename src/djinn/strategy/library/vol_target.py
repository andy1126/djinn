"""波动率目标仓位策略:按组合已实现波动率动态缩放等权底仓。"""

from __future__ import annotations

import math

import pandas as pd

from djinn.strategy.base import SCOPE_PORTFOLIO, Context, Strategy, param


class VolTarget(Strategy):
    """波动率目标(等权底仓 + 按已实现波动率缩放)。

    每个调仓日计算组合自身已实现波动率(年化,美股口径 252),把等权底仓按
    ``min(1, target_vol / realized_vol)`` 缩放(长期不杠杆),剩余资金留在现金。
    波动高于目标时降低仓位、低于目标时加回,是常用的风控手法。
    """

    scope = SCOPE_PORTFOLIO

    target_vol = param(0.10, min=0.01, max=0.5, description="年化目标波动率")
    lookback = param(20, min=5, max=120, description="已实现波动率回看窗口(交易日)")
    rebalance_freq = param(5, min=1, max=60, description="调仓间隔(交易日)")

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self._bars_seen = 0
        self._equity_hist: list[float] = []

    def on_bar(self, ctx: Context) -> None:
        n = self._bars_seen
        self._bars_seen += 1
        self._equity_hist.append(ctx.portfolio.equity)
        if n % int(self.rebalance_freq) != 0:
            return
        symbols = [s for s in ctx.data.symbols if len(ctx.data[s]) > 0]
        if not symbols:
            return
        hist = pd.Series(self._equity_hist, dtype="float64")
        scale = 1.0
        if len(hist) >= 2:
            daily_vol = float(
                hist.pct_change().dropna().tail(int(self.lookback)).std(ddof=0)
            )
            ann_vol = daily_vol * math.sqrt(252)
            if ann_vol > 0:
                scale = min(1.0, float(self.target_vol) / ann_vol)
        w = scale / len(symbols)
        for s in symbols:
            ctx.order_target_percent(s, w)

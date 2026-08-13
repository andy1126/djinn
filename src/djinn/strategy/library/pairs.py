"""配对交易(做多便宜侧):两标的价差 z-score 均值回归。"""

from __future__ import annotations

import math

import pandas as pd

from djinn.strategy.base import SCOPE_PORTFOLIO, Context, Strategy, param


class PairsSpread(Strategy):
    """配对交易(长仓近似)。

    ``spread = close_a / close_b`` 的滚动 z-score:z 过低(A 相对便宜)→ 全仓 A;
    z 过高 → 全仓 B;回归到 ``exit_z`` 以内 → 平仓。引擎仅支持做多,故以
    「做多便宜侧」近似配对的多空对冲。
    """

    scope = SCOPE_PORTFOLIO

    symbol_a = param(None, description="标的 A(代码)")
    symbol_b = param(None, description="标的 B(代码)")
    lookback = param(60, min=10, max=500, description="z-score 回看窗口")
    entry_z = param(2.0, min=0.5, max=5.0, description="开仓阈值")
    exit_z = param(0.5, min=0.0, max=2.0, description="平仓阈值")

    def on_bar(self, ctx: Context) -> None:
        a, b = self.symbol_a, self.symbol_b
        if not a or not b or a not in ctx.data or b not in ctx.data:
            return
        da, db = ctx.data[a], ctx.data[b]
        merged = pd.concat(
            [da["close"].rename("a"), db["close"].rename("b")], axis=1
        ).dropna()
        if len(merged) < int(self.lookback) + 1:
            return
        spread = merged["a"] / merged["b"]
        mean = spread.rolling(int(self.lookback)).mean()
        std = spread.rolling(int(self.lookback)).std()
        z_now = float(((spread - mean) / std).iloc[-1])
        if math.isnan(z_now):
            return
        if z_now < -float(self.entry_z):
            ctx.order_target_percent(a, 1.0)
            ctx.order_target_percent(b, 0.0)
        elif z_now > float(self.entry_z):
            ctx.order_target_percent(a, 0.0)
            ctx.order_target_percent(b, 1.0)
        elif abs(z_now) < float(self.exit_z):
            ctx.order_target_percent(a, 0.0)
            ctx.order_target_percent(b, 0.0)

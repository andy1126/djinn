"""买入并持有 + 定期再平衡策略(等权基准)。"""

from __future__ import annotations

from djinn.strategy.base import SCOPE_PORTFOLIO, Context, Strategy, param


class BuyAndHold(Strategy):
    """买入并持有 + 定期等权再平衡(基准)。

    首个调仓日等权买入全部标的,之后每隔 ``rebalance_freq`` 日把权重调回等权;
    永不主动平仓(除非被风控 / 资金约束拦截)。适合作为策略表现的对标基准。
    """

    scope = SCOPE_PORTFOLIO

    rebalance_freq = param(20, min=1, max=250, description="再平衡间隔(交易日)")

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self._bars_seen = 0

    def on_bar(self, ctx: Context) -> None:
        n = self._bars_seen
        self._bars_seen += 1
        if n % int(self.rebalance_freq) != 0:
            return
        symbols = [s for s in ctx.data.symbols if len(ctx.data[s]) > 0]
        if not symbols:
            return
        w = 1.0 / len(symbols)
        for s in symbols:
            ctx.order_target_percent(s, w)

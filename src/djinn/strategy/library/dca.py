"""定投策略(Dollar-Cost Averaging):按固定周期等额买入。

DCA 需要状态(已投期数),且买入动作不依赖信号反转,故覆写 on_bar 而非 signals。
"""

from __future__ import annotations

from djinn.strategy.base import Context, Strategy, param


class DCA(Strategy):
    """定期定额买入。

    每 ``frequency`` 个交易日投入 ``amount`` 元到 ``symbol``(等权分配到多标的时,
    每个 symbol 投 amount/len(symbols))。
    """

    symbol = param(None, description="定投标的;None 表示对全部成分等额定投")
    frequency = param(20, min=1, max=250, description="定投间隔(交易日)")
    amount = param(1000.0, min=1.0, description="每期投入金额(元)")

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self._bars_since_last = 0

    def on_bar(self, ctx: Context) -> None:
        self._bars_since_last += 1
        if self._bars_since_last < int(self.frequency):
            return
        self._bars_since_last = 0
        symbols = [self.symbol] if self.symbol else list(ctx.data.symbols)
        symbols = [s for s in symbols if s in ctx.data]
        if not symbols:
            return
        per = float(self.amount) / len(symbols)
        for s in symbols:
            # 以当前净值百分比表达"投入 per 元":percent = per / equity
            eq = ctx.portfolio.equity
            if eq <= 0:
                continue
            ctx.buy(s, percent=per / eq)

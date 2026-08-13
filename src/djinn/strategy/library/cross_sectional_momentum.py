"""横截面动量轮动策略:定期买入过去 N 日涨幅最高的 Top K 标的。"""

from __future__ import annotations

import pandas as pd

from djinn.strategy.base import SCOPE_PORTFOLIO, Context, Strategy, param


class CrossSectionalMomentum(Strategy):
    """横截面动量轮动(纯价格驱动)。

    每个调仓日,按过去 ``lookback`` 日收益率排序,等权买入 Top ``n_stocks``;
    落选者清仓。与 :class:`FactorPortfolioStrategy` 同属组合型策略,但只用价格动量。
    """

    scope = SCOPE_PORTFOLIO

    lookback = param(20, min=2, max=250, description="回看周期(交易日)")
    n_stocks = param(5, min=1, max=50, description="持有 Top N")
    rebalance_freq = param(20, min=1, max=250, description="调仓间隔(交易日)")

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self._bars_seen = 0

    def on_bar(self, ctx: Context) -> None:
        n = self._bars_seen
        self._bars_seen += 1
        if n % int(self.rebalance_freq) != 0:
            return
        # 逐标的算过去 lookback 日收益率
        rets: dict[str, float] = {}
        for s in ctx.data.symbols:
            df = ctx.data[s]
            if len(df) < int(self.lookback) + 1:
                continue
            r = float(df["close"].pct_change(int(self.lookback)).iloc[-1])
            if pd.notna(r):
                rets[s] = r
        if not rets:
            return
        top = sorted(rets, key=lambda s: rets[s], reverse=True)[: int(self.n_stocks)]
        selected = set(top)
        # 落选者清仓
        for s, pos in ctx.portfolio.positions.items():
            if pos.qty > 0 and s not in selected:
                ctx.order_target_percent(s, 0.0)
        # 等权调入 Top N
        w = 1.0 / len(top)
        for s in top:
            ctx.order_target_percent(s, w)

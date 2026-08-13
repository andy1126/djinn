"""双动量策略:绝对动量过滤 + 相对动量轮动。"""

from __future__ import annotations

import pandas as pd

from djinn.strategy.base import SCOPE_PORTFOLIO, Context, Strategy, param


class DualMomentum(Strategy):
    """双动量(绝对 + 相对)。

    每个调仓日,选过去 ``lookback`` 日收益率最高的标的;仅当其收益 > 0(绝对
    动量过滤)才全仓持有,否则全部空仓(回现金)。
    """

    scope = SCOPE_PORTFOLIO

    lookback = param(20, min=2, max=250, description="回看周期(交易日)")
    rebalance_freq = param(20, min=1, max=250, description="调仓间隔(交易日)")

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self._bars_seen = 0

    def on_bar(self, ctx: Context) -> None:
        n = self._bars_seen
        self._bars_seen += 1
        if n % int(self.rebalance_freq) != 0:
            return
        rets: dict[str, float] = {}
        for s in ctx.data.symbols:
            df = ctx.data[s]
            if len(df) < int(self.lookback) + 1:
                continue
            r = float(df["close"].pct_change(int(self.lookback)).iloc[-1])
            if pd.notna(r):
                rets[s] = r
        best = max(rets, key=lambda s: rets[s]) if rets else None
        # 绝对动量:最佳资产收益须 > 0,否则空仓
        target = best if (best is not None and rets[best] > 0) else None
        for s, pos in ctx.portfolio.positions.items():
            if pos.qty > 0 and s != target:
                ctx.order_target_percent(s, 0.0)
        if target is not None:
            ctx.order_target_percent(target, 1.0)

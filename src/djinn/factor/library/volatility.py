"""波动因子:历史波动率与相对基准 beta。"""

from __future__ import annotations

import pandas as pd

from djinn.factor.base import Factor, Panel, PanelDict, param


class VolatilityFactor(Factor):
    """N 日日收益波动率(标准差)。"""

    name = "volatility"
    category = "volatility"
    period = param(20, min=5, max=250, description="波动率回看窗口(交易日)")

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        ret = prices.pct_change()
        return ret.rolling(int(self.period)).std()


class BetaFactor(Factor):
    """N 日 beta(对截面等权"市场"收益)。"""

    name = "beta"
    category = "volatility"
    period = param(60, min=20, max=250, description="beta 回看窗口(交易日)")

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        p = int(self.period)
        ret = prices.pct_change()
        market = ret.mean(axis=1)  # 等权市场代理
        cov = ret.rolling(p).cov(market)
        var = market.rolling(p).var()
        return cov.div(var.replace(0.0, pd.NA), axis=0)

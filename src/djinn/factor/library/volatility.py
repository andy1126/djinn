"""波动因子:历史波动率与相对基准 beta。"""

from __future__ import annotations

import numpy as np

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
        return cov.div(var.where(var != 0), axis=0)


class DownsideVolatilityFactor(Factor):
    """N 日下行波动率(仅负收益的标准差)。"""

    name = "downside_volatility"
    category = "volatility"
    period = param(20, min=5, max=250, description="回看窗口(交易日)")

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        ret = prices.pct_change()
        downside = ret.where(ret < 0)
        return downside.rolling(int(self.period)).std()


class MaxLotteryFactor(Factor):
    """MAX 彩票:窗口内最大的 5 个日收益均值(彩票型偏好)。"""

    name = "max_lottery"
    category = "volatility"
    period = param(21, min=5, max=60, description="回看窗口(交易日)")

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        def _top5_mean(x: np.ndarray) -> float:
            x = x[~np.isnan(x)]
            if len(x) < 5:
                return float("nan")
            return float(np.sort(x)[-5:].mean())

        ret = prices.pct_change()
        return ret.rolling(int(self.period)).apply(_top5_mean, raw=True)


class IdioVolFactor(Factor):
    """特质波动率:日收益对市场代理(截面等权)滚动 OLS 残差 std。"""

    name = "idio_vol"
    category = "volatility"
    period = param(60, min=20, max=250, description="滚动窗口(交易日)")

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        p = int(self.period)
        ret = prices.pct_change()
        market = ret.mean(axis=1)  # 等权市场代理
        cov = ret.rolling(p).cov(market)
        var = market.rolling(p).var()
        beta = cov.div(var.where(var != 0), axis=0)
        resid = ret.sub(beta.mul(market, axis=0), axis=0)
        return resid.rolling(p).std()

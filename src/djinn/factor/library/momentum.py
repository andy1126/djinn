"""动量 / 反转因子。

- 动量 ``momentum``:过去 ``period`` 日累计收益,跳过最近 ``skip`` 日(剔除短期反转);
- 反转 ``reversal``:过去 ``period`` 日累计收益取负(短期超跌反弹)。
均为价格向量化,``date t`` 仅用 ``≤ t`` 收盘。
"""

from __future__ import annotations

from djinn.factor.base import Factor, Panel, PanelDict, param


class MomentumFactor(Factor):
    """N 日动量(可跳过近 1 月)。"""

    name = "momentum"
    category = "momentum"
    period = param(20, min=2, max=250, description="动量回看窗口(交易日)")
    skip = param(0, min=0, max=60, description="跳过最近 N 日(剔除短期反转)")

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        p = int(self.period)
        s = int(self.skip)
        base = prices.shift(s)
        past = prices.shift(s + p)
        return base / past - 1.0


class ReversalFactor(Factor):
    """N 日反转(负的短期累计收益)。"""

    name = "reversal"
    category = "momentum"
    period = param(5, min=1, max=60, description="反转回看窗口(交易日)")

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return -prices.pct_change(int(self.period))


class High52WFactor(Factor):
    """52 周高点距离 = close / 滚动最大 close - 1(负值,越接近 0 越强)。"""

    name = "high_52w"
    category = "momentum"
    window = param(252, min=20, max=500, description="回看窗口(交易日)")

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return prices / prices.rolling(int(self.window)).max() - 1.0

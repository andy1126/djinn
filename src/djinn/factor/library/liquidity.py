"""流动性因子:换手率(成交额 / 流通市值)。"""

from __future__ import annotations

import pandas as pd

from djinn.data.schema import COL_AMOUNT, COL_FLOAT_CAP
from djinn.factor.base import Factor, Panel, PanelDict, param


class TurnoverFactor(Factor):
    """N 日平均换手率 = mean(amount) / float_cap(缺流通市值则 NaN)。"""

    name = "turnover"
    category = "liquidity"
    period = param(20, min=1, max=120, description="换手率平滑窗口(交易日)")
    required_fundamentals = (COL_FLOAT_CAP,)
    required_ohlcv = (COL_AMOUNT,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        amount = ohlcv.get(COL_AMOUNT)
        float_cap = fundamentals.get(COL_FLOAT_CAP)
        if amount is None or float_cap is None:
            return pd.DataFrame(
                float("nan"), index=prices.index, columns=prices.columns
            )
        mean_amount = amount.rolling(int(self.period)).mean()
        cap = float_cap.reindex(index=prices.index, columns=prices.columns)
        return mean_amount / cap.replace(0.0, pd.NA)


class AmihudFactor(Factor):
    """Amihud 非流动性 = N 日均值(|日收益| / 成交额)。"""

    name = "amihud"
    category = "liquidity"
    period = param(20, min=1, max=120, description="平滑窗口(交易日)")
    required_ohlcv = (COL_AMOUNT,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        amount = ohlcv.get(COL_AMOUNT)
        if amount is None:
            return pd.DataFrame(
                float("nan"), index=prices.index, columns=prices.columns
            )
        amt = amount.reindex(index=prices.index, columns=prices.columns)
        illiq = prices.pct_change().abs() / amt.where(amt > 0)
        return illiq.rolling(int(self.period)).mean()


class TurnoverChangeFactor(Factor):
    """换手率变化率 = 短期均换手 / 长期均换手 - 1。"""

    name = "turnover_chg"
    category = "liquidity"
    short = param(20, min=5, max=60, description="短期窗口(交易日)")
    long = param(120, min=30, max=250, description="长期窗口(交易日)")
    required_fundamentals = (COL_FLOAT_CAP,)
    required_ohlcv = (COL_AMOUNT,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        amount = ohlcv.get(COL_AMOUNT)
        float_cap = fundamentals.get(COL_FLOAT_CAP)
        if amount is None or float_cap is None:
            return pd.DataFrame(
                float("nan"), index=prices.index, columns=prices.columns
            )
        cap = float_cap.reindex(index=prices.index, columns=prices.columns)
        daily_to = amount / cap.replace(0.0, pd.NA)  # 日换手率
        short_to = daily_to.rolling(int(self.short)).mean()
        long_to = daily_to.rolling(int(self.long)).mean()
        return short_to / long_to.replace(0.0, pd.NA) - 1.0

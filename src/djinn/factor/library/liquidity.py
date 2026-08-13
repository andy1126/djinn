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

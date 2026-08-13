"""质量因子:ROE、毛利率、净利率。"""

from __future__ import annotations

from djinn.data.schema import COL_GROSS_MARGIN, COL_NET_PROFIT, COL_REVENUE, COL_ROE
from djinn.factor.base import Factor, Panel, PanelDict
from djinn.factor.library._util import fund_panel


class ROEFactor(Factor):
    """净资产收益率 ROE(point-in-time 财报口径)。"""

    name = "roe"
    category = "quality"

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return fund_panel(fundamentals, COL_ROE, prices)


class GrossMarginFactor(Factor):
    """销售毛利率。"""

    name = "gross_margin"
    category = "quality"

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return fund_panel(fundamentals, COL_GROSS_MARGIN, prices)


class NetProfitMarginFactor(Factor):
    """净利率 = 净利润 / 营业收入(营收 ≤ 0 → NaN)。"""

    name = "net_profit_margin"
    category = "quality"

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        profit = fund_panel(fundamentals, COL_NET_PROFIT, prices)
        revenue = fund_panel(fundamentals, COL_REVENUE, prices)
        return profit / revenue.where(revenue > 0)

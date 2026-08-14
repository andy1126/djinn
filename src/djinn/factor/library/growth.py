"""成长因子:营收同比、净利同比。"""

from __future__ import annotations

from djinn.data.schema import COL_PROFIT_YOY, COL_REVENUE_YOY
from djinn.factor.base import Factor, Panel, PanelDict
from djinn.factor.library._util import fund_panel


class RevenueGrowthFactor(Factor):
    """营业收入同比增长率。"""

    name = "revenue_yoy"
    category = "growth"
    required_fundamentals = (COL_REVENUE_YOY,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return fund_panel(fundamentals, COL_REVENUE_YOY, prices)


class ProfitGrowthFactor(Factor):
    """净利润同比增长率。"""

    name = "profit_yoy"
    category = "growth"
    required_fundamentals = (COL_PROFIT_YOY,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return fund_panel(fundamentals, COL_PROFIT_YOY, prices)

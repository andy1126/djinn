"""质量因子:ROE、毛利率、净利率、应计、资产增长。"""

from __future__ import annotations

from djinn.data.schema import (
    COL_GROSS_MARGIN,
    COL_NET_PROFIT,
    COL_OCF,
    COL_REVENUE,
    COL_ROE,
    COL_TOTAL_ASSETS,
)
from djinn.factor.base import Factor, Panel, PanelDict
from djinn.factor.library._util import fund_panel


class ROEFactor(Factor):
    """净资产收益率 ROE(point-in-time 财报口径)。"""

    name = "roe"
    category = "quality"
    max_lookback: int = 1  # D3:基本面直读,无滚动窗口
    required_fundamentals = (COL_ROE,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return fund_panel(fundamentals, COL_ROE, prices)


class GrossMarginFactor(Factor):
    """销售毛利率。"""

    name = "gross_margin"
    category = "quality"
    max_lookback: int = 1  # D3:基本面直读,无滚动窗口
    required_fundamentals = (COL_GROSS_MARGIN,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return fund_panel(fundamentals, COL_GROSS_MARGIN, prices)


class NetProfitMarginFactor(Factor):
    """净利率 = 净利润 / 营业收入(营收 ≤ 0 → NaN)。"""

    name = "net_profit_margin"
    category = "quality"
    max_lookback: int = 1  # D3:基本面直读,无滚动窗口
    required_fundamentals = (COL_NET_PROFIT, COL_REVENUE)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        profit = fund_panel(fundamentals, COL_NET_PROFIT, prices)
        revenue = fund_panel(fundamentals, COL_REVENUE, prices)
        return profit / revenue.where(revenue > 0)


class AccrualsFactor(Factor):
    """应计因子 = (净利润 - 经营现金流) / 总资产 的期间变化率。

    低应计(利润质量高)为优质信号;因子值取应计比率的环比变化,使用者常给负权重。
    """

    name = "accruals"
    category = "quality"
    max_lookback: int = 1  # D3:环比变化仅需 1 期前值
    required_fundamentals = (COL_NET_PROFIT, COL_OCF, COL_TOTAL_ASSETS)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        net_profit = fund_panel(fundamentals, COL_NET_PROFIT, prices)
        ocf = fund_panel(fundamentals, COL_OCF, prices)
        ta = fund_panel(fundamentals, COL_TOTAL_ASSETS, prices)
        accrual_ratio = (net_profit - ocf) / ta.where(ta != 0)
        # PIT 面板为阶梯函数,环比即相邻报告期变化率
        return accrual_ratio.pct_change()


class AssetGrowthFactor(Factor):
    """资产增长率 = 总资产环比变化率(强负向异象,权重示例给负)。"""

    name = "asset_growth"
    category = "quality"
    max_lookback: int = 1  # D3:环比变化仅需 1 期前值
    required_fundamentals = (COL_TOTAL_ASSETS,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        ta = fund_panel(fundamentals, COL_TOTAL_ASSETS, prices)
        return ta.pct_change()

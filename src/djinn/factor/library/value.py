"""价值因子:EP / BP / SP(估值倒数,越高越便宜)。"""

from __future__ import annotations

from djinn.data.schema import COL_MARKET_CAP, COL_OCF, COL_PB, COL_PE, COL_PS
from djinn.factor.base import Factor, Panel, PanelDict
from djinn.factor.library._util import fund_panel


def _reciprocal(fundamentals: PanelDict, key: str, like: Panel) -> Panel:
    raw = fund_panel(fundamentals, key, like)
    pos = raw.where(raw > 0)  # 负 / 零估值无意义 → NaN
    return 1.0 / pos


class EPFactor(Factor):
    """盈利收益率 EP = 1 / PE。"""

    name = "ep"
    category = "value"
    required_fundamentals = (COL_PE,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return _reciprocal(fundamentals, COL_PE, prices)


class BPFactor(Factor):
    """账面市值比 BP = 1 / PB。"""

    name = "bp"
    category = "value"
    required_fundamentals = (COL_PB,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return _reciprocal(fundamentals, COL_PB, prices)


class SPFactor(Factor):
    """营收市值比 SP = 1 / PS。"""

    name = "sp"
    category = "value"
    required_fundamentals = (COL_PS,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return _reciprocal(fundamentals, COL_PS, prices)


class CFPFactor(Factor):
    """现金市值比 = 经营现金流 / 总市值(越高越便宜)。"""

    name = "cfp"
    category = "value"
    required_fundamentals = (COL_OCF, COL_MARKET_CAP)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        ocf = fund_panel(fundamentals, COL_OCF, prices)
        cap = fund_panel(fundamentals, COL_MARKET_CAP, prices)
        return ocf / cap.where(cap > 0)

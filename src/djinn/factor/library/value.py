"""价值因子:EP / BP / SP(估值倒数,越高越便宜)。"""

from __future__ import annotations

from djinn.data.schema import COL_PB, COL_PE, COL_PS
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

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return _reciprocal(fundamentals, COL_PE, prices)


class BPFactor(Factor):
    """账面市值比 BP = 1 / PB。"""

    name = "bp"
    category = "value"

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return _reciprocal(fundamentals, COL_PB, prices)


class SPFactor(Factor):
    """营收市值比 SP = 1 / PS。"""

    name = "sp"
    category = "value"

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        return _reciprocal(fundamentals, COL_PS, prices)

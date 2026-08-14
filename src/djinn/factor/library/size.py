"""规模因子:对数总市值。"""

from __future__ import annotations

import numpy as np

from djinn.data.schema import COL_MARKET_CAP
from djinn.factor.base import Factor, Panel, PanelDict
from djinn.factor.library._util import fund_panel


class SizeFactor(Factor):
    """规模 = ln(market_cap)(市值 ≤ 0 → NaN)。"""

    name = "size"
    category = "size"
    required_fundamentals = (COL_MARKET_CAP,)

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        cap = fund_panel(fundamentals, COL_MARKET_CAP, prices)
        pos = cap.where(cap > 0)
        return pos.map(np.log)

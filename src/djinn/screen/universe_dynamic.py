"""动态股票池:``date → symbols`` 的时变成分映射。

由打分历史(:func:`~djinn.screen.scoring.top_n` 逐日截面)或条件筛选历史构建,
供选股回测按日取当时成分;成分中途增删时,引擎靠并集日历 + 前向填充估值容忍
缺失行情(见 ``event_engine`` 的 ``calendar="union"``)。
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date

import pandas as pd

from djinn.screen.scoring import top_n
from djinn.screen.screener import ScreenCondition, Screener


class DynamicUniverse:
    """``date → list[symbol]`` 的动态成分表。"""

    def __init__(self, mapping: dict[date, list[str]]) -> None:
        self._map: dict[date, list[str]] = {d: list(s) for d, s in mapping.items()}
        self._dates: list[date] = sorted(self._map)

    @property
    def dates(self) -> list[date]:
        """有成分记录的全部日期(升序)。"""
        return list(self._dates)

    @property
    def all_symbols(self) -> list[str]:
        """历史出现过的不重复标的(按首次出现排序)。"""
        out: list[str] = []
        seen: set[str] = set()
        for d in self._dates:
            for s in self._map[d]:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out

    def symbols_on(self, when: date) -> list[str]:
        """``when`` 当日(或之前最近记录日)的成分;无记录返回空列表。"""
        if not self._dates:
            return []
        # D9:二分定位 ``≤ when`` 的最近记录日(O(log T),替代线性扫)
        pos = bisect_right(self._dates, when)
        if pos == 0:
            return []
        return list(self._map[self._dates[pos - 1]])

    def __len__(self) -> int:
        return len(self._dates)

    # ── 构建 ────────────────────────────────────────────
    @classmethod
    def from_score_history(cls, score_df: pd.DataFrame, n: int) -> DynamicUniverse:
        """由 ``date × symbol`` 综合得分宽表,逐日取 TopN 构建。"""
        mapping = {ts.date(): top_n(score_df, ts, n) for ts in score_df.index}
        return cls(mapping)

    @classmethod
    def from_condition_history(
        cls,
        conditions: list[ScreenCondition],
        fundamentals_by_date: dict[date, pd.DataFrame],
        ohlcv_derived_by_date: dict[date, pd.DataFrame] | None = None,
    ) -> DynamicUniverse:
        """由逐日基本面截面 + 条件筛选构建。"""
        mapping: dict[date, list[str]] = {}
        for d, fdf in fundamentals_by_date.items():
            mkt = (ohlcv_derived_by_date or {}).get(d)
            mapping[d] = Screener.apply(conditions, fdf, mkt)
        return cls(mapping)

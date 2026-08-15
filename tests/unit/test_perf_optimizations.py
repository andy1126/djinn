"""D4 / D8 性能优化的等价性测试(新旧输出逐值相等)。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from djinn.data.market_data import MarketData
from djinn.data.schema import Adjust, Market
from djinn.engine.event_engine import EventDrivenEngine
from djinn.strategy.base import DataView


def _md(symbol: str, n: int = 10) -> MarketData:
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": [50.0] * n,
            "high": [51.0] * n,
            "low": [49.0] * n,
            "close": [50.0 + i for i in range(n)],
            "volume": [1.0e6] * n,
        },
        index=idx,
    )
    return MarketData(symbol=symbol, market=Market.US, df=df, adjust=Adjust.BACKWARD)


def test_dataview_searchsorted_equal() -> None:
    """D4:DataView.__getitem__ 的 searchsorted 切片与 df.loc[:now] 完全相等。"""
    md = _md("S")
    nows = [
        date(2024, 1, 2),
        date(2024, 1, 5),
        date(2024, 1, 6),  # 周六(非交易日)
        date(2024, 1, 20),
    ]
    for now in nows:
        view = DataView({"S": md}, now)
        got = view["S"]
        expected = md.df.loc[: pd.Timestamp(now)]
        pd.testing.assert_frame_equal(got, expected)


def test_bars_at_pos_map_equivalence() -> None:
    """D8:_bars_at 用预计算 pos_map 与 bar_at 结果逐字段相等。"""
    md = _md("S")
    data = {"S": md}
    eng = EventDrivenEngine()
    pos_maps = {s: {t: i for i, t in enumerate(m.df.index)} for s, m in data.items()}
    for ts in pd.DatetimeIndex(md.df.index):
        bar = eng._bars_at(data, ts, pos_maps)["S"]
        expected = md.bar_at(ts.date())
        assert bar is not None and expected is not None
        assert bar.timestamp == expected.timestamp
        assert bar.open == expected.open
        assert bar.close == expected.close
        assert bar.volume == expected.volume
    # 非交易日 → None
    assert eng._bars_at(data, pd.Timestamp("2024-01-07"), pos_maps)["S"] is None

"""H1 引擎暖机:EngineConfig.start(账本从指定日开,暖机数据仅供因子 lookback)。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from djinn.data.market_data import MarketData
from djinn.data.schema import Adjust, Market
from djinn.engine.event_engine import EngineConfig, EventDrivenEngine
from djinn.strategy.base import Strategy


def _md(start: str = "2024-01-01", end: str = "2024-03-31") -> MarketData:
    idx = pd.bdate_range(start, end)
    n = len(idx)
    closes = [100.0 + i for i in range(n)]  # 线性上行
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1.0e6] * n,
        },
        index=idx,
    )
    return MarketData(symbol="A", market=Market.US, df=df, adjust=Adjust.BACKWARD)


class _Recorder(Strategy):
    """记录首个 on_bar 日 ctx.data 可见的最早日期(验证暖机历史可见)。"""

    def __init__(self) -> None:
        super().__init__()
        self.first_seen_earliest: pd.Timestamp | None = None
        self.n_bars = 0

    def on_bar(self, ctx) -> None:  # type: ignore[no-untyped-def]
        self.n_bars += 1
        if self.first_seen_earliest is None:
            self.first_seen_earliest = ctx.data["A"].index.min()


def _engine(start: date | None = None) -> EngineConfig:
    return EngineConfig(initial_cash=100000.0, start=start)


def test_start_filters_equity_index() -> None:
    """start 之后才开账:净值 index 从 start 起,首值 == initial_cash。"""
    res = EventDrivenEngine(_engine(date(2024, 2, 1))).run(_Recorder(), {"A": _md()})
    assert res.equity_curve.index[0] == pd.Timestamp("2024-02-01")
    assert res.equity_curve.iloc[0] == pytest.approx(100000.0)
    # start 过滤生效:净值只覆盖 start 之后,短于完整数据跨度
    assert len(res.equity_curve) < len(_md().df.index)


def test_warmup_history_visible() -> None:
    """暖机数据对策略可见:首日 on_bar 能看到早于 start 的行情。"""
    strat = _Recorder()
    EventDrivenEngine(_engine(date(2024, 2, 1))).run(strat, {"A": _md()})
    # 数据首日 2024-01-01,引擎 2024-02-01 开账;策略首日即见 01-01 起的历史
    assert strat.first_seen_earliest == pd.Timestamp("2024-01-01")
    idx = _md().df.index
    assert strat.n_bars == int((idx >= pd.Timestamp("2024-02-01")).sum())


def test_no_start_equivalent_to_data_start() -> None:
    """start=None 行为与数据首日开账一致(回归安全)。"""
    res = EventDrivenEngine(_engine(None)).run(_Recorder(), {"A": _md()})
    assert res.equity_curve.index[0] == pd.Timestamp("2024-01-01")


def test_start_after_data_end_raises() -> None:
    """start 晚于所有数据 → ValueError(避免空回测静默成功)。"""
    with pytest.raises(ValueError, match="无交易日可回测"):
        EventDrivenEngine(_engine(date(2025, 1, 1))).run(_Recorder(), {"A": _md()})

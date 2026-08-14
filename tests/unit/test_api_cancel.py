"""E4:任务取消(状态标志 + 引擎协作式中断 + 端点)。"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from djinn.api.jobs import JobRegistry
from djinn.data.market_data import MarketData
from djinn.data.schema import Adjust, Market
from djinn.engine import EngineConfig, EventDrivenEngine
from djinn.engine.commission import USCommissionModel
from djinn.engine.slippage import ZeroSlippage
from djinn.strategy import MACrossover
from djinn.utils.exceptions import BacktestCancelled


def _md(symbol: str, n: int) -> MarketData:
    idx = pd.bdate_range("2024-01-01", periods=n)
    return MarketData(
        symbol=symbol,
        market=Market.US,
        df=pd.DataFrame(
            {
                "open": [10.0] * n,
                "high": [10.0] * n,
                "low": [10.0] * n,
                "close": [10.0 + i * 0.01 for i in range(n)],
                "volume": [1e6] * n,
            },
            index=idx,
        ),
        adjust=Adjust.BACKWARD,
    )


def test_registry_cancel_flags(tmp_path) -> None:
    registry = JobRegistry(db_path=tmp_path / "jobs.db")
    job = registry.create("backtest", meta={})
    assert registry.request_cancel(job.job_id) is True
    assert registry.is_cancel_requested(job.job_id) is True
    registry.clear_cancel(job.job_id)
    assert registry.is_cancel_requested(job.job_id) is False


def test_cancel_non_pending_fails(tmp_path) -> None:
    registry = JobRegistry(db_path=tmp_path / "jobs.db")
    job = registry.create("backtest", meta={})
    registry.update(job.job_id, status="done")
    assert registry.request_cancel(job.job_id) is False


def test_engine_should_stop_raises() -> None:
    """should_stop 第 3 日返回 True → 引擎抛 BacktestCancelled。"""
    md = _md("AAPL", 30)
    cfg = EngineConfig(
        initial_cash=Decimal("100000"),
        commission=USCommissionModel(rate=0.0, min_commission=0.0),
        slippage=ZeroSlippage(),
    )
    strategy = MACrossover(fast=5, slow=15)
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    with pytest.raises(BacktestCancelled):
        EventDrivenEngine(cfg).run(strategy, {"AAPL": md}, should_stop=should_stop)
    assert calls["n"] >= 3

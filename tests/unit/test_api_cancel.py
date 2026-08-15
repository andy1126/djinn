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


def test_run_backtest_job_cancel(tmp_path) -> None:
    """E4:run_backtest_job 在取消标志下 → status=cancelled,__meta__ 保留,无报告落盘。"""
    from pathlib import Path

    from djinn.api.jobs import run_backtest_job
    from djinn.data import ProviderRegistry
    from djinn.data.provider import DataProvider

    class _SlowProvider(DataProvider):
        name = "slow"

        def supports(self, symbol, market=None):
            return True

        def get_ohlcv(self, symbol, start, end, adjust=Adjust.BACKWARD, market=None):
            idx = pd.bdate_range(start, end)
            n = len(idx)
            df = pd.DataFrame(
                {
                    "open": [100.0 + 0.1 * i for i in range(n)],
                    "high": [101.0 + 0.1 * i for i in range(n)],
                    "low": [99.0 + 0.1 * i for i in range(n)],
                    "close": [100.0 + 0.1 * i for i in range(n)],
                    "volume": [1.0e6] * n,
                },
                index=idx,
            )
            return MarketData(symbol=symbol, market=Market.US, df=df, adjust=adjust)

    reg = JobRegistry(db_path=str(tmp_path / "jobs.db"))
    cfg = {
        "universe": {"symbols": ["A", "B", "C", "D"], "market": "US"},
        "period": {"start": "2020-01-01", "end": "2022-12-31"},
        "strategy": {"name": "MACrossover", "params": {"fast": 5, "slow": 15}},
    }
    job = reg.create("backtest", meta={"config": cfg, "title": "cancel-test"})
    reg.request_cancel(job.job_id)
    run_backtest_job(
        reg, job.job_id, provider_registry=ProviderRegistry([_SlowProvider()])
    )
    j = reg.get(job.job_id)
    assert j.status == "cancelled", f"应为 cancelled,实际 {j.status} {j.error}"
    # __meta__ 保留(可重新提交)
    assert (j.result or {}).get("__meta__") is not None
    # 无报告落盘
    assert not list(Path(".cache/djinn_results").glob(f"{job.job_id}.json"))


def test_progress_callback_throttles_writes() -> None:
    """E12:高频 update 经 0.5s 节流后 DB 写入次数显著少于调用次数。"""
    from djinn.api.jobs import ProgressCallback

    class _CountingRegistry:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, job_id, **kwargs) -> None:
            self.calls += 1

    reg = _CountingRegistry()
    cb = ProgressCallback(job_id="x", registry=reg, min_interval_sec=0.05)  # type: ignore[arg-type]
    for i in range(100):
        cb.update(i / 100.0, f"stage-{i}")
    # 瞬时 100 次调用(间隔 < 0.05s)→ 节流到远少于 100 次 DB 写
    assert reg.calls < 50, f"节流失效:写入 {reg.calls} 次"
    assert reg.calls >= 1  # 至少首写
    # force 跳过节流(终态强写)
    cb.update(1.0, "done", force=True)
    assert reg.calls >= 2

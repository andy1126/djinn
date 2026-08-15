"""H6 Walk-Forward API 测试:/walk-forwards 创建/查询 + 孤儿恢复接线。

不触网:注入确定性 stub provider;TestClient 同步执行后台任务,POST 后即可轮询结果。
"""

from __future__ import annotations

import math
import os
from datetime import date

import pandas as pd
from fastapi.testclient import TestClient

os.environ.setdefault("DJINN_TEST", "1")

from djinn.api.deps import get_job_registry, get_registry
from djinn.api.jobs import JobRegistry
from djinn.api.main import app
from djinn.data.market_data import MarketData
from djinn.data.provider import DataProvider, ProviderRegistry
from djinn.data.schema import Adjust, Market

_SYMBOL = "AAPL"
_START = "2020-01-01"
_END = "2021-12-31"
_GRID = {"fast": [5, 10], "slow": [20, 30]}


def _synth_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    """线性趋势 + 正弦波动(网格可区分,与 test_walk_forward 同口径)。"""
    idx = pd.bdate_range(start, end)
    n = len(idx)
    closes = [100.0 + 0.08 * i + 6.0 * math.sin(i / 12.0) for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1.0e6] * n,
            "amount": [1.0e8] * n,
        },
        index=idx,
    )


class _StubProvider(DataProvider):
    name = "stub"
    market = Market.US

    def supports(self, symbol: str, market: Market | None = None) -> bool:
        return True

    def get_ohlcv(
        self, symbol: str, start: date, end: date, adjust: Adjust = Adjust.BACKWARD
    ) -> MarketData:
        return MarketData(
            symbol=symbol,
            market=Market.US,
            df=_synth_ohlcv(symbol, start, end),
            adjust=adjust,
        )


_stub_registry = ProviderRegistry([_StubProvider()])
_test_registry = JobRegistry(db_path=".cache/test_jobs_walk.db")

client = TestClient(app)


def setup_module() -> None:
    """注入 stub registry(walk-forward 任务经 Depends(get_registry) 命中)。"""
    app.dependency_overrides[get_job_registry] = lambda: _test_registry
    app.dependency_overrides[get_registry] = lambda: _stub_registry


def teardown_module() -> None:
    app.dependency_overrides.clear()


def _payload(
    *,
    grid: dict | None = None,
    is_days: int = 126,
    oos_days: int = 63,
    min_is_sharpe: float | None = None,
    include_wf: bool = True,
) -> dict:
    cfg: dict = {
        "universe": {"symbols": [_SYMBOL]},
        "period": {"start": _START, "end": _END},
        "account": {"initial_cash": 100000, "currency": "USD"},
        "strategy": {"name": "MACrossover", "params": {}},
    }
    if include_wf:
        cfg["walk_forward"] = {
            "is_days": is_days,
            "oos_days": oos_days,
            "grid": _GRID,
            "min_is_sharpe": min_is_sharpe,
        }
    return {"config": cfg, "grid": grid, "target": None, "parallel": False}


def _wait_done(job_id: str) -> dict:
    """轮询任务直至终态(TestClient 同步执行,通常首轮即 done)。"""
    for _ in range(50):
        got = client.get(f"/walk-forwards/{job_id}")
        assert got.status_code == 200
        body = got.json()
        if body["status"] in ("done", "error", "cancelled"):
            return body
    raise AssertionError("walk-forward 任务超时未完成")


def test_create_and_poll() -> None:
    """POST → job done,结果含逐窗口 + 拼接指标,可 JSON。"""
    resp = client.post("/walk-forwards", json=_payload())
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    body = _wait_done(job_id)
    assert body["status"] == "done", body.get("error")
    result = body["result"]
    assert result["__meta__"]["config"] is not None  # __meta__ 保留
    report = result["report"]
    assert isinstance(report["windows"], list) and len(report["windows"]) >= 2
    assert report["metrics"] is not None
    assert report["equity_curve"] is not None
    deployed = [w for w in report["windows"] if w["deployed"]]
    assert len(deployed) >= 1
    for w in deployed:
        assert w["best_params"] in [
            {"fast": f, "slow": s} for f in _GRID["fast"] for s in _GRID["slow"]
        ]
        assert w["oos_equity"]["index"] and w["oos_equity"]["values"]


def test_request_grid_overrides_config() -> None:
    """请求体 grid 覆盖配置:只用 fast=5/slow=20 一种组合。"""
    over = {"fast": [5], "slow": [20]}
    resp = client.post("/walk-forwards", json=_payload(grid=over))
    assert resp.status_code == 200
    body = _wait_done(resp.json()["job_id"])
    assert body["status"] == "done", body.get("error")
    for w in body["result"]["report"]["windows"]:
        assert w["best_params"] in [{"fast": 5, "slow": 20}] or not w["deployed"]


def test_invalid_grid_key_400() -> None:
    """非法扫轴前缀 → 400(复用 sweep 的网格校验)。"""
    resp = client.post("/walk-forwards", json=_payload(grid={"bogus.prefix": [1]}))
    assert resp.status_code == 400
    assert "未知扫轴前缀" in resp.json()["detail"]


def test_missing_walk_forward_job_errors() -> None:
    """配置缺 walk_forward 段 → 任务 error(异步校验,非 400)。"""
    resp = client.post("/walk-forwards", json=_payload(include_wf=False))
    assert resp.status_code == 200
    body = _wait_done(resp.json()["job_id"])
    assert body["status"] == "error"
    assert "walk_forward" in body["error"]


def test_orphan_recovery_includes_walk_forward(tmp_path, monkeypatch) -> None:
    """孤儿恢复能识别 walk-forward 任务(_RUNNERS 已注册)。"""
    import djinn.api.jobs as jobs_mod

    monkeypatch.delenv("DJINN_TEST", raising=False)
    monkeypatch.setattr(jobs_mod, "_RUNNERS", {"walk-forward": lambda *a, **k: None})
    jobs_mod._recovered_jobs.clear()
    reg = JobRegistry(db_path=str(tmp_path / "walk.db"))
    j = reg.create("walk-forward", meta={"config": {"x": 1}})
    reg.update(j.job_id, status="running")

    assert jobs_mod.recover_orphaned_jobs(reg) == 1
    jobs_mod._recovered_jobs.clear()

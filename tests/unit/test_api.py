"""FastAPI 端点测试:策略/回测/扫描/数据/报告 + 配置校验。

使用 TestClient 直连 app,不依赖外部 uvicorn 进程。
"""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

# 使用临时 DB 避免污染真实 job 库
os.environ.setdefault("DJINN_TEST", "1")

from djinn.api.deps import get_cache, get_job_registry
from djinn.api.jobs import JobRegistry
from djinn.api.main import app
from djinn.data import DataCache

# 注入临时 registry(独立 DB)与隔离缓存(避免 test_clear_cache 清掉真实 .cache/djinn/)
_test_registry = JobRegistry(db_path=".cache/test_jobs.db")
app.dependency_overrides[get_job_registry] = lambda: _test_registry
_test_cache = DataCache(cache_dir=".cache/test_api_cache")
app.dependency_overrides[get_cache] = lambda: _test_cache

client = TestClient(app)

VALID_BACKTEST_CONFIG = {
    "universe": {"symbols": ["NVDA"], "benchmark": "^GSPC", "market": "US"},
    "period": {"start": "2024-01-01", "end": "2024-06-30"},
    "account": {"initial_cash": 100000, "currency": "USD"},
    "strategy": {"name": "MACrossover", "params": {"fast": 10, "slow": 30}},
    "costs": {
        "commission": {"type": "us"},
        "slippage": {"type": "fixed_bps", "bps": 5},
    },
    "portfolio": {"mode": "single", "allocation": "equal"},
    "output": {"export": [], "report": "none"},
    "adjust": "backward",
}


def teardown_module() -> None:
    app.dependency_overrides.clear()


# ── 健康检查 ──────────────────────────────────────────
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── 策略端点 ──────────────────────────────────────────
def test_list_strategies():
    resp = client.get("/strategies")
    assert resp.status_code == 200
    data = resp.json()
    names = [s["name"] for s in data["strategies"]]
    assert "MACrossover" in names
    assert "DCA" in names
    # 参数 schema 必须是 list[dict] 且含必要字段
    ma = next(s for s in data["strategies"] if s["name"] == "MACrossover")
    assert isinstance(ma["params"], list)
    assert ma["params"][0]["name"] == "fast"
    assert "type" in ma["params"][0]


def test_get_strategy():
    resp = client.get("/strategies/MACrossover")
    assert resp.status_code == 200
    assert resp.json()["name"] == "MACrossover"


def test_unknown_strategy_404():
    resp = client.get("/strategies/Bogus")
    assert resp.status_code == 404


# ── 回测端点 ──────────────────────────────────────────
def test_create_backtest_returns_id() -> None:
    resp = client.post("/backtests", json={"config": VALID_BACKTEST_CONFIG})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert len(data["job_id"]) == 12


def test_backtest_crud_and_list() -> None:
    resp = client.post("/backtests", json={"config": VALID_BACKTEST_CONFIG})
    job_id = resp.json()["job_id"]

    # 查询
    got = client.get(f"/backtests/{job_id}")
    assert got.status_code == 200
    assert got.json()["job_id"] == job_id
    # 可读 title 由配置派生
    assert got.json()["title"] == "MACrossover · NVDA · 2024-01-01~2024-06-30"

    # 列表含本任务且带 title
    lst = client.get("/backtests")
    assert lst.status_code == 200
    matches = [j for j in lst.json() if j["job_id"] == job_id]
    assert matches
    assert matches[0]["title"] == "MACrossover · NVDA · 2024-01-01~2024-06-30"


def test_backtest_404() -> None:
    assert client.get("/backtests/deadbeef").status_code == 404


def test_invalid_export_format_400() -> None:
    resp = client.post("/backtests", json={"config": VALID_BACKTEST_CONFIG})
    job_id = resp.json()["job_id"]
    assert client.get(f"/backtests/{job_id}/export/json").status_code == 400


def test_export_unfinished_400() -> None:
    resp = client.post("/backtests", json={"config": VALID_BACKTEST_CONFIG})
    job_id = resp.json()["job_id"]
    # 任务进入 running/done 前立即导出(可能已被后台处理,故重试一次)
    export_resp = client.get(f"/backtests/{job_id}/export/csv")
    # 若已完成则 200,未完成则 400;不崩即正确行为
    assert export_resp.status_code in (200, 400)


# ── 扫描端点 ──────────────────────────────────────────
def test_create_sweep() -> None:
    resp = client.post(
        "/sweeps",
        json={
            "config": VALID_BACKTEST_CONFIG,
            "grid": {"fast": [5, 10], "slow": [20, 30]},
            "target": "sharpe",
            "parallel": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert len(data["job_id"]) == 12


def test_sweep_list_and_404() -> None:
    client.post(
        "/sweeps",
        json={
            "config": VALID_BACKTEST_CONFIG,
            "grid": {"fast": [5]},
            "target": "sharpe",
            "parallel": False,
        },
    )
    lst = client.get("/sweeps")
    assert lst.status_code == 200
    # 扫描 title 带 "参数扫描" 前缀和目标
    titles = [j["title"] for j in lst.json() if j["title"]]
    assert any(
        t.startswith("参数扫描 MACrossover · NVDA") and "目标=sharpe" in t
        for t in titles
    )
    assert client.get("/sweeps/deadbeef").status_code == 404


# ── 数据端点 ──────────────────────────────────────────
def test_list_cache() -> None:
    resp = client.get("/data/cache")
    assert resp.status_code == 200
    assert "entries" in resp.json()


def test_clear_cache() -> None:
    resp = client.delete("/data/cache")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"


def test_fetch_data_invalid_dates_400() -> None:
    resp = client.post(
        "/data/fetch",
        json={
            "symbols": ["NVDA"],
            "market": "US",
            "start": "not-a-date",
            "end": "2024-06-30",
            "adjust": "backward",
        },
    )
    assert resp.status_code == 400  # date.fromisoformat 失败


# ── 配置校验 ──────────────────────────────────────────
def test_validation_empty_symbols_422() -> None:
    resp = client.post(
        "/backtests",
        json={
            "config": {
                "universe": {"symbols": [], "market": "US"},
                "period": {"start": "2024-01-01", "end": "2024-06-30"},
                "strategy": {"name": "MACrossover", "params": {}},
            }
        },
    )
    assert resp.status_code == 422


def test_validation_bad_period_422() -> None:
    resp = client.post(
        "/backtests",
        json={
            "config": {
                "universe": {"symbols": ["NVDA"], "market": "US"},
                "period": {"start": "2024-12-31", "end": "2024-01-01"},
                "strategy": {"name": "MACrossover", "params": {}},
            }
        },
    )
    assert resp.status_code == 422


def test_validation_missing_fields_422() -> None:
    assert client.post("/backtests", json={"config": {}}).status_code == 422

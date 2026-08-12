"""策略端点测试:统一列表(含 FactorPortfolio)+ 用户策略 CRUD + validate。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from djinn.api.deps import get_strategy_store
from djinn.api.main import app
from djinn.strategy.store import StrategyStore

_DB = Path(".cache/test_strategies.db")

_store = StrategyStore(db_path=_DB)
app.dependency_overrides[get_strategy_store] = lambda: _store

client = TestClient(app)

SIGNALS_SRC = """
fast = param(10, min=2, max=100)

def signals(self, data):
    close = data["close"]
    up = cross_over(sma(close, self.fast), sma(close, self.fast * 2))
    down = cross_under(sma(close, self.fast), sma(close, self.fast * 2))
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[up] = 1
    sig[down] = -1
    return state_from_signals(sig)
"""


def setup_module() -> None:
    global _store
    if _DB.exists():
        _DB.unlink()
    _store = StrategyStore(db_path=_DB)
    app.dependency_overrides[get_strategy_store] = lambda: _store


def teardown_module() -> None:
    app.dependency_overrides.clear()


def test_list_includes_factor_portfolio():
    names = [s["name"] for s in client.get("/strategies").json()["strategies"]]
    assert "FactorPortfolio" in names  # 修复注册表漂移
    assert "MACrossover" in names


def test_create_and_list_user_strategy():
    resp = client.post(
        "/strategies/user",
        json={"name": "MyMAC", "source_code": SIGNALS_SRC, "kind": "python"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "MyMAC"
    assert [p["name"] for p in data["params"]] == ["fast"]

    names = [s["name"] for s in client.get("/strategies").json()["strategies"]]
    assert "MyMAC" in names


def test_duplicate_name_409():
    client.post("/strategies/user", json={"name": "Dup", "source_code": SIGNALS_SRC})
    resp = client.post(
        "/strategies/user", json={"name": "Dup", "source_code": SIGNALS_SRC}
    )
    assert resp.status_code == 409


def test_builtin_name_conflict_409():
    resp = client.post(
        "/strategies/user", json={"name": "MACrossover", "source_code": SIGNALS_SRC}
    )
    assert resp.status_code == 409


def test_invalid_code_400():
    resp = client.post(
        "/strategies/user",
        json={
            "name": "Bad",
            "source_code": "def signals(self, d):\n    import os\n    return d['close']*0",
        },
    )
    assert resp.status_code == 400


def test_validate_endpoint():
    ok = client.post(
        "/strategies/user/validate",
        json={"name": "V", "source_code": SIGNALS_SRC, "kind": "python"},
    ).json()
    assert ok["valid"] is True
    assert [p["name"] for p in ok["params"]] == ["fast"]

    bad = client.post(
        "/strategies/user/validate",
        json={"name": "V", "source_code": "x = 1", "kind": "python"},
    ).json()
    assert bad["valid"] is False
    assert bad["error"]


def test_update_and_delete():
    created = client.post(
        "/strategies/user", json={"name": "U", "source_code": SIGNALS_SRC}
    ).json()
    sid = created["strategy_id"]
    up = client.put(f"/strategies/user/{sid}", json={"name": "U2"})
    assert up.status_code == 200
    assert up.json()["name"] == "U2"
    assert client.delete(f"/strategies/user/{sid}").status_code == 204
    assert client.get("/strategies/U2").status_code == 404

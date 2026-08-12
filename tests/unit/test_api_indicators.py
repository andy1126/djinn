"""指标端点测试:指标库列表 + 用户指标 CRUD + validate。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from djinn.api.deps import get_indicator_store
from djinn.api.main import app
from djinn.indicators.store import IndicatorStore

_DB = Path(".cache/test_indicators.db")

_store = IndicatorStore(db_path=_DB)
app.dependency_overrides[get_indicator_store] = lambda: _store

client = TestClient(app)


def _src(name: str) -> str:
    return f"def {name}(close, n=5):\n    return close / close.shift(n) - 1\n"


def setup_module() -> None:
    global _store
    if _DB.exists():
        _DB.unlink()
    _store = IndicatorStore(db_path=_DB)
    app.dependency_overrides[get_indicator_store] = lambda: _store


def teardown_module() -> None:
    app.dependency_overrides.clear()


def test_list_includes_builtin_and_user():
    client.post(
        "/indicators/user", json={"name": "my_roc", "source_code": _src("my_roc")}
    )
    names = [i["name"] for i in client.get("/indicators").json()["indicators"]]
    assert "sma" in names and "rsi" in names and "my_roc" in names


def test_create_and_signature():
    r = client.post(
        "/indicators/user", json={"name": "my_roc2", "source_code": _src("my_roc2")}
    )
    assert r.status_code == 201
    assert "close" in r.json()["signature"]


def test_duplicate_409():
    client.post("/indicators/user", json={"name": "dup", "source_code": _src("dup")})
    resp = client.post(
        "/indicators/user", json={"name": "dup", "source_code": _src("dup")}
    )
    assert resp.status_code == 409


def test_builtin_conflict_409():
    resp = client.post(
        "/indicators/user", json={"name": "sma", "source_code": _src("sma")}
    )
    assert resp.status_code == 409


def test_validate_endpoint():
    ok = client.post(
        "/indicators/user/validate", json={"name": "v", "source_code": _src("v")}
    ).json()
    assert ok["valid"] is True
    assert ok["signature"]
    bad = client.post(
        "/indicators/user/validate", json={"name": "v", "source_code": "x = 1"}
    ).json()
    assert bad["valid"] is False
    assert bad["error"]


def test_delete():
    created = client.post(
        "/indicators/user", json={"name": "to_delete", "source_code": _src("to_delete")}
    ).json()
    assert (
        client.delete(f"/indicators/user/{created['indicator_id']}").status_code == 204
    )

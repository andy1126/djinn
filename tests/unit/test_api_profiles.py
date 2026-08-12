"""标的 profile 端点测试:CRUD 全链路 + 去重/重名校验。

使用 TestClient 直连 app,注入临时 ProfileRegistry(独立 SQLite),不污染真实库。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from djinn.api.deps import get_profile_registry
from djinn.api.main import app
from djinn.api.profiles import ProfileRegistry

_DB_PATH = Path(".cache/test_profiles.db")

_registry = ProfileRegistry(db_path=_DB_PATH)
app.dependency_overrides[get_profile_registry] = lambda: _registry

client = TestClient(app)


def setup_module() -> None:
    """每次运行前清掉旧库,保证干净状态。"""
    global _registry
    if _DB_PATH.exists():
        _DB_PATH.unlink()
    _registry = ProfileRegistry(db_path=_DB_PATH)
    app.dependency_overrides[get_profile_registry] = lambda: _registry


def teardown_module() -> None:
    app.dependency_overrides.clear()


def test_create_profile():
    resp = client.post(
        "/profiles",
        json={
            "name": "美股科技",
            "symbols": ["NVDA", "", "AAPL", "NVDA"],
            "market": "US",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "美股科技"
    assert data["market"] == "US"
    # 去空串 + 保序去重
    assert data["symbols"] == ["NVDA", "AAPL"]
    assert data["profile_id"]


def test_create_duplicate_name_conflict():
    client.post("/profiles", json={"name": "重复名", "symbols": ["AAPL"]})
    resp = client.post("/profiles", json={"name": "重复名", "symbols": ["MSFT"]})
    assert resp.status_code == 409


def test_list_profiles():
    client.post(
        "/profiles", json={"name": "港股", "symbols": ["0700.HK"], "market": "HK"}
    )
    resp = client.get("/profiles")
    assert resp.status_code == 200
    names = [p["name"] for p in resp.json()]
    assert "港股" in names
    # 按名称升序
    assert names == sorted(names)


def test_get_profile_and_404():
    created = client.post(
        "/profiles", json={"name": "单标的", "symbols": ["NVDA"]}
    ).json()
    resp = client.get(f"/profiles/{created['profile_id']}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "单标的"
    assert client.get("/profiles/does-not-exist").status_code == 404


def test_update_profile():
    created = client.post(
        "/profiles", json={"name": "旧名", "symbols": ["AAPL"], "market": "US"}
    ).json()
    resp = client.put(
        f"/profiles/{created['profile_id']}",
        json={"name": "新名", "symbols": ["MSFT", "GOOG"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "新名"
    assert data["symbols"] == ["MSFT", "GOOG"]
    # 未传 market → 保持不变
    assert data["market"] == "US"


def test_delete_profile():
    created = client.post(
        "/profiles", json={"name": "待删", "symbols": ["NVDA"]}
    ).json()
    pid = created["profile_id"]
    resp = client.delete(f"/profiles/{pid}")
    assert resp.status_code == 204
    assert client.get(f"/profiles/{pid}").status_code == 404
    # 再删 → 404
    assert client.delete(f"/profiles/{pid}").status_code == 404

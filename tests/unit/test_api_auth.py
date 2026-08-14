"""E8:可选 Bearer token 鉴权 + CORS env 化测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from djinn.api import main as api_main


def test_auth_middleware_enforced(monkeypatch) -> None:
    """设置 token 后:无/错 header 401、对 token 放行、/health 免密。"""
    monkeypatch.setattr(api_main, "_API_TOKEN", "secret-token")
    client = TestClient(api_main.app)
    assert client.get("/strategies").status_code == 401
    assert (
        client.get("/strategies", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert (
        client.get(
            "/strategies", headers={"Authorization": "Bearer secret-token"}
        ).status_code
        == 200
    )
    # /health 免密
    assert client.get("/health").status_code == 200


def test_auth_disabled_by_default(monkeypatch) -> None:
    """未设置 token:零配置放行(现有流程不破坏)。"""
    monkeypatch.setattr(api_main, "_API_TOKEN", None)
    client = TestClient(api_main.app)
    assert client.get("/health").status_code == 200
    assert client.get("/strategies").status_code == 200

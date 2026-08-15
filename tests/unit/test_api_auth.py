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


def test_ws_auth_enforced(monkeypatch) -> None:
    """E8:WS 握手无/错 token → 4001 拒绝;对 token → 鉴权通过(任务不存在再 4004)。"""
    import pytest
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("DJINN_API_TOKEN", "secret-token")
    client = TestClient(api_main.app)
    # 无 token
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect("/jobs/nonexistent/progress"),
    ):
        pass
    assert exc.value.code == 4001
    # 错 token
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect("/jobs/nonexistent/progress?token=wrong"),
    ):
        pass
    assert exc.value.code == 4001
    # 对 token:鉴权通过(握手成功,不再 4001 拒绝)
    with client.websocket_connect("/jobs/nonexistent/progress?token=secret-token"):
        pass


def test_ws_auth_disabled_by_default(monkeypatch) -> None:
    """E8:未设置 token 时 WS 零配置放行(握手成功,不 4001 拒绝)。"""
    monkeypatch.delenv("DJINN_API_TOKEN", raising=False)
    client = TestClient(api_main.app)
    with client.websocket_connect("/jobs/nonexistent/progress"):
        pass

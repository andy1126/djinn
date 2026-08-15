"""WebSocket 进度推送共享实现(E10)。

把「订阅 job 状态 → 推送初始态 + 增量更新 + 心跳 → 终态关闭」抽成公共函数,
供 ``/backtests/{id}/progress``(旧)与 ``/jobs/{id}/progress``(新,通用)复用。
"""

from __future__ import annotations

import asyncio
import os

from fastapi import WebSocket, WebSocketDisconnect

from djinn.api.jobs import JobRecord, JobRegistry

# 终态:任务结束后关闭连接。
_TERMINAL = ("done", "error", "cancelled")


def _ws_authorized(websocket: WebSocket) -> bool:
    """E8:WS 鉴权(设置 DJINN_API_TOKEN 后,握手须带 ``?token=`` 或 Bearer 头)。

    HTTP 中间件只覆盖普通请求;WebSocket 握手不走中间件,需在订阅前单独校验。
    未设置 token 时零配置放行。
    """
    token = os.environ.get("DJINN_API_TOKEN")
    if not token:
        return True
    if websocket.query_params.get("token") == token:
        return True
    return websocket.headers.get("authorization", "") == f"Bearer {token}"


async def stream_job_progress(
    websocket: WebSocket, registry: JobRegistry, job_id: str
) -> None:
    """接受 WS 连接,推送 job 的当前态与后续增量(含心跳),终态后关闭。"""
    if not _ws_authorized(websocket):
        await websocket.close(code=4001, reason="未授权")
        return
    await websocket.accept()
    job = registry.get(job_id)
    if not job:
        await websocket.close(code=4004, reason="任务不存在")
        return
    await websocket.send_json(job.to_dict())
    if job.status in _TERMINAL:
        await websocket.close()
        return

    queue: asyncio.Queue[JobRecord] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(updated_job: JobRecord) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, updated_job)

    registry.subscribe(job_id, callback)
    try:
        while True:
            try:
                updated = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(updated.to_dict())
                if updated.status in _TERMINAL:
                    break
            except TimeoutError:
                # 心跳(保持连接,并让前端感知仍在线)
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        registry.unsubscribe(job_id, callback)

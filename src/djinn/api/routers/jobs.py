"""统一任务管理端点(E4:取消 / E6:清理 / E10:通用进度 WS)。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, WebSocket

from djinn.api.deps import get_job_registry
from djinn.api.jobs import JobRegistry
from djinn.api.ws import stream_job_progress

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/{job_id}/cancel")
def cancel_job(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> dict[str, str]:
    """请求取消任务(仅 pending / running 可取消)。

    回测 / 扫描在协作式检查点检测到取消标志后置为 ``cancelled``;已结束任务
    返回 409。
    """
    if not registry.request_cancel(job_id):
        raise HTTPException(status_code=409, detail="任务不存在或已结束,无法取消")
    return {"status": "cancel_requested"}


@router.post("/purge")
def purge_jobs(
    days: int = 30, registry: JobRegistry = Depends(get_job_registry)
) -> dict[str, int]:
    """清理 ``days`` 天前已终态的任务记录 + 报告缓存 + 导出文件(E6)。"""
    if days <= 0:
        raise HTTPException(status_code=400, detail="days 必须为正整数")
    removed = registry.purge_older_than(days)
    return {"removed": removed}


@router.websocket("/{job_id}/progress")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """通用任务进度推送(sweep / factor-analysis / factor-matrix / screen 等)。

    与 ``/backtests/{id}/progress`` 共用同一实现;前端各任务页统一走本端点。
    """
    await stream_job_progress(websocket, get_job_registry(), job_id)

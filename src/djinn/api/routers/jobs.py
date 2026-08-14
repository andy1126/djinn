"""统一任务管理端点(E4:取消)。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from djinn.api.deps import get_job_registry
from djinn.api.jobs import JobRegistry

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

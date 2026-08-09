"""选股路由:截面条件过滤 + 可选多因子打分排序(POST /screens)。

选股为长任务,后台线程执行(见 :func:`~djinn.api.jobs.run_screen_job`),结果
(股票列表 + 得分)存入 job result,经 ``/{job_id}`` 以 JobStatus 取出。
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from djinn.api.deps import get_job_registry, get_registry
from djinn.api.jobs import JobRegistry, run_screen_job
from djinn.api.schemas import JobCreated, JobStatus, ScreenRequest
from djinn.data.provider import ProviderRegistry

router = APIRouter(prefix="/screens", tags=["screens"])


@router.post("", response_model=JobCreated)
async def create_screen(
    req: ScreenRequest,
    background_tasks: BackgroundTasks,
    registry: JobRegistry = Depends(get_job_registry),
    preg: ProviderRegistry = Depends(get_registry),
) -> JobCreated:
    """创建截面选股任务(异步执行)。"""
    if not req.symbols and not req.index:
        raise HTTPException(
            status_code=400, detail="需提供 symbols 或 index 作为候选池"
        )
    if req.top_n and not req.scores:
        raise HTTPException(status_code=400, detail="top_n 需配合 scores 使用")
    meta = req.model_dump(mode="json")
    universe_desc = req.index or f"{len(req.symbols or [])}只"
    meta["title"] = f"选股 · {len(req.conditions)}条件 · {universe_desc}"
    job = registry.create("screen", meta=meta)
    background_tasks.add_task(run_screen_job, registry, job.job_id, preg)
    return JobCreated(job_id=job.job_id, status="pending")


@router.get("/{job_id}", response_model=JobStatus)
async def get_screen(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> JobStatus:
    """查询选股任务状态与结果(股票列表 + 得分在 ``result.results``)。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return JobStatus(**job.to_dict())

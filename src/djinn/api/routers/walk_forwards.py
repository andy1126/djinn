"""Walk-Forward 分析路由:创建/查询(H 计划)。"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from djinn.api.deps import get_job_registry, get_registry
from djinn.api.jobs import JobRegistry, make_title, run_walk_forward_job
from djinn.api.routers.sweeps import _validate_grid_keys
from djinn.api.schemas import JobCreated, JobStatus, WalkForwardRequest
from djinn.data.provider import ProviderRegistry

router = APIRouter(prefix="/walk-forwards", tags=["walk-forwards"])


@router.post("", response_model=JobCreated)
async def create_walk_forward(
    req: WalkForwardRequest,
    background_tasks: BackgroundTasks,
    registry: JobRegistry = Depends(get_job_registry),
    provider_registry: ProviderRegistry = Depends(get_registry),
) -> JobCreated:
    """创建 walk-forward 分析任务(逐窗口 IS 独立选参 + OOS 评估,异步执行)。"""
    if req.grid:
        _validate_grid_keys(req.grid)
    cfg_json = req.config.model_dump(mode="json")
    job = registry.create(
        "walk-forward",
        meta={
            "config": cfg_json,
            "grid": req.grid,
            "target": req.target,
            "parallel": req.parallel,
            "title": make_title(cfg_json, kind="walk-forward", target=req.target),
        },
    )
    background_tasks.add_task(
        run_walk_forward_job, registry, job.job_id, None, provider_registry
    )
    return JobCreated(job_id=job.job_id, status="pending")


@router.get("", response_model=list[JobStatus])
async def list_walk_forwards(
    limit: int = 50, registry: JobRegistry = Depends(get_job_registry)
) -> list[JobStatus]:
    """列出历史 walk-forward 任务。"""
    jobs = registry.list(limit=limit, kind="walk-forward")
    return [JobStatus(**job.to_dict()) for job in jobs]


@router.get("/{job_id}", response_model=JobStatus)
async def get_walk_forward(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> JobStatus:
    """查询 walk-forward 任务状态与结果。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return JobStatus(**job.to_dict())

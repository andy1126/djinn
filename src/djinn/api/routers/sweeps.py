"""参数扫描路由:创建/查询。"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from djinn.api.deps import get_job_registry
from djinn.api.jobs import JobRegistry, make_title, run_sweep_job
from djinn.api.schemas import JobCreated, JobStatus, SweepRequest

router = APIRouter(prefix="/sweeps", tags=["sweeps"])


@router.post("", response_model=JobCreated)
async def create_sweep(
    req: SweepRequest,
    background_tasks: BackgroundTasks,
    registry: JobRegistry = Depends(get_job_registry),
):
    """创建参数扫描任务(异步执行)。"""
    cfg_json = req.config.model_dump(mode="json")
    job = registry.create(
        "sweep",
        meta={
            "config": cfg_json,
            "grid": req.grid,
            "target": req.target,
            "parallel": req.parallel,
            "title": make_title(cfg_json, kind="sweep", target=req.target),
        },
    )
    background_tasks.add_task(run_sweep_job, registry, job.job_id)
    return JobCreated(job_id=job.job_id, status="pending")


@router.get("", response_model=list[JobStatus])
async def list_sweeps(
    limit: int = 50, registry: JobRegistry = Depends(get_job_registry)
):
    """列出历史扫描任务。"""
    jobs = registry.list(limit=limit, kind="sweep")
    return [JobStatus(**job.to_dict()) for job in jobs]


@router.get("/{job_id}", response_model=JobStatus)
async def get_sweep(job_id: str, registry: JobRegistry = Depends(get_job_registry)):
    """查询扫描任务状态与结果。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return JobStatus(**job.to_dict())

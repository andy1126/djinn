"""参数扫描路由:创建/查询。"""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from djinn.api.deps import get_job_registry, get_registry
from djinn.api.jobs import JobRegistry, make_title, run_sweep_job
from djinn.api.schemas import JobCreated, JobStatus, SweepRequest
from djinn.cli.sweep import ALLOWED_SWEEP_AXES
from djinn.data.provider import ProviderRegistry

router = APIRouter(prefix="/sweeps", tags=["sweeps"])


def _validate_grid_keys(grid: dict[str, list[Any]]) -> None:
    """grid key 校验:必须落在已知轴前缀内,否则 400 带允许列表。

    条款宽松:裸策略参数(无前缀、如 ``fast``)默认视为 ``strategy.params.*``
    顶层参数,允许通过——这样旧形 sweep 配置不被破坏。仅拦截明显非法 key。
    """
    for key in grid:
        if "." in key:
            prefix = key.split(".", 1)[0]
            if prefix not in (
                "universe",
                "strategy",
                "portfolio",
                "risk",
                "account",
                "costs",
                "output",
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"未知扫轴前缀: {key}。"
                        f"允许的轴(裸策略参数直接写名即可):"
                        f"{', '.join(ALLOWED_SWEEP_AXES)}"
                    ),
                )


@router.post("", response_model=JobCreated)
async def create_sweep(
    req: SweepRequest,
    background_tasks: BackgroundTasks,
    registry: JobRegistry = Depends(get_job_registry),
    provider_registry: ProviderRegistry = Depends(get_registry),
) -> JobCreated:
    """创建参数扫描任务(异步执行)。"""
    _validate_grid_keys(req.grid)
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
    background_tasks.add_task(
        run_sweep_job, registry, job.job_id, None, provider_registry
    )
    return JobCreated(job_id=job.job_id, status="pending")


@router.get("", response_model=list[JobStatus])
async def list_sweeps(
    limit: int = 50, registry: JobRegistry = Depends(get_job_registry)
) -> list[JobStatus]:
    """列出历史扫描任务。"""
    jobs = registry.list(limit=limit, kind="sweep")
    return [JobStatus(**job.to_dict()) for job in jobs]


@router.get("/{job_id}", response_model=JobStatus)
async def get_sweep(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> JobStatus:
    """查询扫描任务状态与结果。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return JobStatus(**job.to_dict())

"""因子路由:因子库 schema(GET /factors)+ 单因子分析任务(/factor-analysis)。

因子库复用 :data:`~djinn.factor.library.FACTOR_REGISTRY` 与策略同一套
:func:`~djinn.strategy.parameter.param_schema`,前端可动态生成参数表单。
因子分析为长任务,后台线程执行(见 :func:`~djinn.api.jobs.run_factor_analysis_job`),
结果(报告 dict)直接存入 job result,经 ``/{job_id}/report`` 取出。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from djinn.api.deps import get_job_registry, get_registry
from djinn.api.jobs import JobRegistry, run_factor_analysis_job, run_factor_matrix_job
from djinn.api.schemas import (
    FactorAnalysisRequest,
    FactorInfo,
    FactorListResponse,
    FactorMatrixRequest,
    JobCreated,
    JobStatus,
)
from djinn.data.provider import ProviderRegistry
from djinn.factor.library import FACTOR_REGISTRY, get_factor_class
from djinn.strategy.parameter import param_schema

router = APIRouter(prefix="/factors", tags=["factors"])
analysis_router = APIRouter(prefix="/factor-analysis", tags=["factor-analysis"])
matrix_router = APIRouter(prefix="/factor-matrix", tags=["factor-matrix"])


def _factor_info(name: str) -> FactorInfo:
    cls = get_factor_class(name)
    doc = (cls.__doc__ or "").strip().splitlines()
    desc = doc[0].strip() if doc else ""
    category = str(getattr(cls, "category", "generic"))
    return FactorInfo(
        name=name,
        category=category,
        description=desc,
        params=[p.to_dict() for p in param_schema(cls)],
    )


@router.get("", response_model=FactorListResponse)
async def list_factors() -> FactorListResponse:
    """列出全部内置因子及其参数 schema。"""
    return FactorListResponse(factors=[_factor_info(n) for n in FACTOR_REGISTRY])


@router.get("/{name}", response_model=FactorInfo)
async def get_factor(name: str) -> FactorInfo:
    """获取单个因子的详细信息。"""
    if name not in FACTOR_REGISTRY:
        raise HTTPException(status_code=404, detail=f"因子 {name} 不存在")
    return _factor_info(name)


@analysis_router.post("", response_model=JobCreated)
async def create_factor_analysis(
    req: FactorAnalysisRequest,
    background_tasks: BackgroundTasks,
    registry: JobRegistry = Depends(get_job_registry),
    preg: ProviderRegistry = Depends(get_registry),
) -> JobCreated:
    """创建单因子分析任务(异步执行)。"""
    if req.factor not in FACTOR_REGISTRY:
        raise HTTPException(status_code=404, detail=f"因子 {req.factor} 不存在")
    if not req.symbols and not req.index:
        raise HTTPException(
            status_code=400, detail="需提供 symbols 或 index 作为标的池"
        )
    meta = req.model_dump(mode="json")
    universe_desc = req.index or f"{len(req.symbols or [])}只"
    meta["title"] = f"因子分析 {req.factor} · {universe_desc} · {req.start}~{req.end}"
    job = registry.create("factor-analysis", meta=meta)
    background_tasks.add_task(run_factor_analysis_job, registry, job.job_id, preg)
    return JobCreated(job_id=job.job_id, status="pending")


@analysis_router.get("", response_model=list[JobStatus])
async def list_factor_analyses(
    limit: int = 50, registry: JobRegistry = Depends(get_job_registry)
) -> list[JobStatus]:
    """列出历史因子分析任务。"""
    jobs = registry.list(limit=limit, kind="factor-analysis")
    return [JobStatus(**job.to_dict()) for job in jobs]


@analysis_router.get("/{job_id}", response_model=JobStatus)
async def get_factor_analysis(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> JobStatus:
    """查询因子分析任务状态与结果。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return JobStatus(**job.to_dict())


@analysis_router.get("/{job_id}/report")
async def get_factor_analysis_report(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> dict[str, Any]:
    """取因子分析报告(IC / 分层 / 衰减 / 换手,JSON 友好)。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="任务未完成")
    report = (job.result or {}).get("report")
    if not isinstance(report, dict):
        raise HTTPException(status_code=400, detail="任务缺少报告数据")
    return dict(report)


# ── 多因子诊断(/factor-matrix)─────────────────────────────
@matrix_router.post("", response_model=JobCreated)
async def create_factor_matrix(
    req: FactorMatrixRequest,
    background_tasks: BackgroundTasks,
    registry: JobRegistry = Depends(get_job_registry),
    preg: ProviderRegistry = Depends(get_registry),
) -> JobCreated:
    """创建多因子诊断任务(异步执行):N 因子 × universe × 区间 → 相关矩阵。"""
    if not req.symbols and not req.index:
        raise HTTPException(
            status_code=400, detail="需提供 symbols 或 index 作为标的池"
        )
    unknown = [p.factor for p in req.factors if p.factor not in FACTOR_REGISTRY]
    if unknown:
        raise HTTPException(status_code=404, detail=f"未知因子: {unknown}")
    if len(req.factors) < 2 or len(req.factors) > 8:
        raise HTTPException(status_code=400, detail="多因子诊断需 2~8 个因子")
    meta = req.model_dump(mode="json")
    universe_desc = req.index or f"{len(req.symbols or [])}只"
    names = ",".join(p.factor for p in req.factors)
    meta["title"] = f"多因子诊断 {names} · {universe_desc} · {req.start}~{req.end}"
    job = registry.create("factor-matrix", meta=meta)
    background_tasks.add_task(run_factor_matrix_job, registry, job.job_id, preg)
    return JobCreated(job_id=job.job_id, status="pending")


@matrix_router.get("", response_model=list[JobStatus])
async def list_factor_matrices(
    limit: int = 50, registry: JobRegistry = Depends(get_job_registry)
) -> list[JobStatus]:
    """列出历史多因子诊断任务。"""
    jobs = registry.list(limit=limit, kind="factor-matrix")
    return [JobStatus(**job.to_dict()) for job in jobs]


@matrix_router.get("/{job_id}", response_model=JobStatus)
async def get_factor_matrix(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> JobStatus:
    """查询多因子诊断任务状态。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return JobStatus(**job.to_dict())


@matrix_router.get("/{job_id}/report")
async def get_factor_matrix_report(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> dict[str, Any]:
    """取多因子诊断报告(相关矩阵 + 各因子 IC 汇总 + 换手)。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="任务未完成")
    report = (job.result or {}).get("report")
    if not isinstance(report, dict):
        raise HTTPException(status_code=400, detail="任务缺少报告数据")
    return dict(report)

"""回测路由:创建/查询/导出/WebSocket 进度。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse

from djinn.api.deps import get_job_registry, get_registry
from djinn.api.jobs import JobRecord, JobRegistry, make_title, run_backtest_job
from djinn.api.report_store import densify_payload, rebuild_report
from djinn.api.report_store import load as load_report
from djinn.api.schemas import BacktestRequest, JobCreated, JobStatus
from djinn.data.provider import ProviderRegistry
from djinn.io import export_csv, export_excel

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("", response_model=JobCreated)
async def create_backtest(
    req: BacktestRequest,
    background_tasks: BackgroundTasks,
    registry: JobRegistry = Depends(get_job_registry),
    provider_registry: ProviderRegistry = Depends(get_registry),
) -> JobCreated:
    """创建回测任务(异步执行)。"""
    cfg_json = req.config.model_dump(mode="json")
    job = registry.create(
        "backtest",
        meta={"config": cfg_json, "title": make_title(cfg_json, kind="backtest")},
    )
    background_tasks.add_task(
        run_backtest_job, registry, job.job_id, provider_registry=provider_registry
    )
    return JobCreated(job_id=job.job_id, status="pending")


@router.get("", response_model=list[JobStatus])
async def list_backtests(
    limit: int = 50, registry: JobRegistry = Depends(get_job_registry)
) -> list[JobStatus]:
    """列出历史回测任务。"""
    jobs = registry.list(limit=limit, kind="backtest")
    return [JobStatus(**job.to_dict()) for job in jobs]


@router.get("/{job_id}", response_model=JobStatus)
async def get_backtest(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> JobStatus:
    """查询回测任务状态与结果。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return JobStatus(**job.to_dict())


@router.get("/{job_id}/report")
async def get_backtest_report(
    job_id: str,
    registry: JobRegistry = Depends(get_job_registry),
    provider_registry: ProviderRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """获取完整回测报告(指标 + 曲线 + 交易 + 持仓 + 归因),用于前端结果页。

    报告由后台任务完成后落盘(见 ``api/report_store.py``);此处直接读缓存,
    不再重跑回测。无缓存时回退为重跑(兼容旧任务)。
    """
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="任务未完成")
    payload = load_report(job_id)
    if payload is not None:
        return {"job_id": job_id, **densify_payload(payload)}
    # 回退:旧任务无缓存 → 重跑一次并落盘
    meta = (job.result or {}).get("__meta__", {})
    config_dict = meta.get("config", {})
    if not config_dict:
        raise HTTPException(status_code=400, detail="任务缺少配置元数据")
    try:
        from djinn.api.report_store import save, serialize_report
        from djinn.cli.runner import run_backtest
        from djinn.config import load_config

        cfg = load_config(data=config_dict)
        result = await asyncio.to_thread(
            run_backtest, cfg, registry=provider_registry, with_attribution=True
        )
        payload = serialize_report(result.report)
        save(job_id, payload)
        return {"job_id": job_id, **densify_payload(payload)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


@router.get("/{job_id}/export/{fmt}", response_model=None)
async def export_backtest(
    job_id: str,
    fmt: str,
    registry: JobRegistry = Depends(get_job_registry),
    provider_registry: ProviderRegistry = Depends(get_registry),
) -> dict[str, Any] | FileResponse:
    """导出回测结果为 CSV/Excel。从 ``report_store`` 读缓存报告重建后导出,不重跑。"""
    if fmt not in ("csv", "excel"):
        raise HTTPException(status_code=400, detail="fmt 必须是 csv 或 excel")
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="任务未完成")
    payload = load_report(job_id)
    if payload is None:
        # 回退:旧任务无缓存 → 重跑一次并落盘
        meta = (job.result or {}).get("__meta__", {})
        config_dict = meta.get("config", {})
        if not config_dict:
            raise HTTPException(status_code=400, detail="任务缺少配置元数据")
        from djinn.api.report_store import save, serialize_report
        from djinn.cli.runner import run_backtest
        from djinn.config import load_config

        cfg = load_config(data=config_dict)
        result = await asyncio.to_thread(
            run_backtest, cfg, registry=provider_registry, with_attribution=True
        )
        payload = serialize_report(result.report)
        save(job_id, payload)
    report = rebuild_report(payload)
    output_dir = Path(".cache/exports") / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        await asyncio.to_thread(export_csv, report, output_dir)
        return {
            "status": "ok",
            "path": str(output_dir),
            "files": [str(p) for p in output_dir.glob("*.csv")],
        }
    else:
        path = output_dir / "report.xlsx"
        await asyncio.to_thread(export_excel, report, path)
        return FileResponse(
            path,
            filename=f"{job_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


@router.websocket("/{job_id}/progress")
async def backtest_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """WebSocket 推送回测进度。"""
    await websocket.accept()
    registry = get_job_registry()
    # 先发送当前状态
    job = registry.get(job_id)
    if not job:
        await websocket.close(code=4004, reason="任务不存在")
        return
    await websocket.send_json(job.to_dict())
    if job.status in ("done", "error"):
        await websocket.close()
        return
    # 订阅更新
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
                if updated.status in ("done", "error"):
                    break
            except TimeoutError:
                # 心跳
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        registry.unsubscribe(job_id, callback)

"""回测路由:创建/查询/导出/WebSocket 进度。"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import math
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse

from djinn.api.deps import get_job_registry
from djinn.api.jobs import JobRegistry, make_title, run_backtest_job
from djinn.api.schemas import BacktestRequest, JobCreated, JobStatus
from djinn.cli.runner import run_backtest
from djinn.config import load_config
from djinn.io import export_csv, export_excel

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("", response_model=JobCreated)
async def create_backtest(
    req: BacktestRequest,
    background_tasks: BackgroundTasks,
    registry: JobRegistry = Depends(get_job_registry),
):
    """创建回测任务(异步执行)。"""
    cfg_json = req.config.model_dump(mode="json")
    job = registry.create(
        "backtest",
        meta={"config": cfg_json, "title": make_title(cfg_json, kind="backtest")},
    )
    background_tasks.add_task(run_backtest_job, registry, job.job_id)
    return JobCreated(job_id=job.job_id, status="pending")


@router.get("", response_model=list[JobStatus])
async def list_backtests(
    limit: int = 50, registry: JobRegistry = Depends(get_job_registry)
):
    """列出历史回测任务。"""
    jobs = registry.list(limit=limit, kind="backtest")
    return [JobStatus(**job.to_dict()) for job in jobs]


@router.get("/{job_id}", response_model=JobStatus)
async def get_backtest(job_id: str, registry: JobRegistry = Depends(get_job_registry)):
    """查询回测任务状态与结果。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return JobStatus(**job.to_dict())


@router.get("/{job_id}/report")
async def get_backtest_report(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
):
    """获取完整回测报告(指标 + 曲线 + 交易 + 持仓),用于前端结果页。

    重新运行回测生成报告(简化实现,生产环境应缓存 BacktestResult)。
    """
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="任务未完成")
    meta = (job.result or {}).get("__meta__", {})
    config_dict = meta.get("config", {})
    if not config_dict:
        raise HTTPException(status_code=400, detail="任务缺少配置元数据")
    try:
        cfg = load_config(data=config_dict)
        import asyncio

        result = await asyncio.to_thread(run_backtest, cfg)
        report = result.report

        # 序列化为 JSON 友好结构

        def _series_to_list(s):
            if s is None or len(s) == 0:
                return {"index": [], "values": []}
            return {
                "index": [str(x) for x in s.index],
                "values": [_safe_float(x) for x in s.values],
            }

        def _df_to_dict(df):
            if df is None or df.empty:
                return {
                    "index": [],
                    "columns": list(df.columns) if df is not None else [],
                    "data": [],
                }
            return {
                "index": [str(x) for x in df.index],
                "columns": list(df.columns),
                "data": [[_sanitize(v) for v in row] for row in df.values.tolist()],
            }

        trades_out = []
        for t in report.trades:
            if dataclasses.is_dataclass(t):
                d = {k: _trade_val(v) for k, v in dataclasses.asdict(t).items()}
                trades_out.append(d)
            elif isinstance(t, dict):
                trades_out.append(_sanitize(t))

        return {
            "job_id": job_id,
            "symbols": report.symbols,
            "metrics": _dictify(report.metrics),
            "trade_stats": _dictify(report.trade_stats),
            "benchmark_stats": (
                _dictify(report.benchmark_stats)
                if report.benchmark_stats is not None
                else None
            ),
            "equity_curve": _series_to_list(report.equity_curve),
            "benchmark_curve": _series_to_list(report.benchmark_curve),
            "drawdown_curve": _series_to_list(report.drawdown_curve),
            "monthly_returns": _df_to_dict(report.monthly_returns),
            "yearly_returns": _series_to_list(report.yearly_returns),
            "rolling_sharpe": _series_to_list(report.rolling_sharpe),
            "rolling_volatility": _series_to_list(report.rolling_volatility),
            "trades": trades_out,
            "rejections": [_dictify(r) for r in report.rejections],
            "positions": _df_to_dict(report.positions),
            "weights": _df_to_dict(report.weights),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


def _jsonable(v) -> bool:
    """判断值是否 JSON 可序列化(用于过滤)。"""
    if v is None or isinstance(v, (str, int, bool)):
        return True
    if isinstance(v, float):
        return math.isfinite(v)
    try:
        import json

        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _sanitize(v):
    """把 NaN/Inf 转 None(float),其余递归清洗;JSON 默认编码不接受 NaN/Inf。"""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, list):
        return [_sanitize(x) for x in v]
    if isinstance(v, tuple):
        return [_sanitize(x) for x in v]
    if isinstance(v, dict):
        return {k: _sanitize(val) for k, val in v.items()}
    return v


def _safe_float(v) -> float | None:
    """Series value → finite float,否则 None。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _trade_val(v):
    """单笔交易字段值的 JSON 友好化:date/DateTime → isoformat, NaN/Inf → None。"""
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, list):
        return [_trade_val(x) for x in v]
    return v


def _dictify(obj) -> dict:
    """把 dataclass/对象转为 dict,过滤非 JSON 可序列化字段,NaN/Inf → None。"""
    if obj is None:
        return {}
    raw: dict
    if isinstance(obj, dict):
        raw = obj
    elif hasattr(obj, "__dict__"):
        raw = dict(vars(obj))
    else:
        return {}
    out: dict = {}
    for k, v in raw.items():
        if not _jsonable(v):
            continue
        if isinstance(v, float):
            out[k] = v if math.isfinite(v) else None
        else:
            out[k] = v
    return out


@router.get("/{job_id}/export/{fmt}")
async def export_backtest(
    job_id: str, fmt: str, registry: JobRegistry = Depends(get_job_registry)
):
    """导出回测结果为 CSV/Excel。"""
    if fmt not in ("csv", "excel"):
        raise HTTPException(status_code=400, detail="fmt 必须是 csv 或 excel")
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="任务未完成")
        # 重新运行回测获取报告(简化实现,生产环境应缓存)
    meta = (job.result or {}).get("__meta__", {})
    config_dict = meta.get("config", {})
    cfg = load_config(data=config_dict)
    result = run_backtest(cfg)
    output_dir = Path(".cache/exports") / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        export_csv(result.report, output_dir)
        return {
            "status": "ok",
            "path": str(output_dir),
            "files": list(output_dir.glob("*.csv")),
        }
    else:
        path = output_dir / "report.xlsx"
        export_excel(result.report, path)
        return FileResponse(
            path,
            filename=f"{job_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


@router.websocket("/{job_id}/progress")
async def backtest_progress_ws(websocket: WebSocket, job_id: str):
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
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(updated_job):
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

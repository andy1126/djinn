"""选股路由:截面条件过滤 + 可选多因子打分排序(POST /screens)。

选股为长任务,后台线程执行(见 :func:`~djinn.api.jobs.run_screen_job`),结果
(股票列表 + 得分)存入 job result,经 ``/{job_id}`` 以 JobStatus 取出。
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from djinn.api.deps import get_job_registry, get_registry
from djinn.api.jobs import JobRegistry, run_screen_job
from djinn.api.schemas import (
    JobCreated,
    JobStatus,
    ScreenField,
    ScreenFieldsResponse,
    ScreenMarket,
    ScreenMarketsResponse,
    ScreenRequest,
)
from djinn.data.provider import ProviderRegistry

router = APIRouter(prefix="/screens", tags=["screens"])

# 选股页各市场可用性。A 股估值字段(pe/pb/ps/市值)依赖东财
# ``stock_zh_a_spot_em`` 接口,当前网络不可达(降级为 NaN),选股页据此置灰 A 股;
# 财务字段(roe/毛利率/营收同比等)虽走新浪源可用,但估值筛选是本页主路径,
# 直接置灰 A 股避免用户误选后得到空结果。东财恢复后把 CN 改回 available=True。
_SCREEN_MARKETS: list[ScreenMarket] = [
    ScreenMarket(
        market="CN",
        label="A股",
        available=False,
        reason="A股估值数据源(东财)暂不可达",
    ),
    ScreenMarket(market="HK", label="港股", available=True),
    ScreenMarket(market="US", label="美股", available=True),
]

# 可筛选字段(与 djinn.data.schema.FUNDAMENTAL_VALUE_COLUMNS 对齐)。
# (name, label, group, description):
# - valuation:估值字段,依赖东财估值接口,A 股当前可能不可达(降级为 NaN);
# - financial:财务字段,走新浪财务指标源,A 股可用。
_SCREEN_FIELDS: list[tuple[str, str, Literal["valuation", "financial"], str]] = [
    ("market_cap", "总市值", "valuation", "单位:元"),
    ("float_cap", "流通市值", "valuation", "单位:元"),
    ("pe", "市盈率 PE", "valuation", "股价 / 每股收益"),
    ("pb", "市净率 PB", "valuation", "股价 / 每股净资产"),
    ("ps", "市销率 PS", "valuation", "总市值 / 营业收入"),
    ("roe", "净资产收益率 ROE", "financial", "净利润 / 净资产,单位 %"),
    ("gross_margin", "毛利率", "financial", "毛利 / 营收,单位 %"),
    ("revenue", "营业收入", "financial", "单位:元"),
    ("net_profit", "净利润", "financial", "单位:元"),
    ("ocf", "经营现金流", "financial", "经营活动现金流净额,单位:元"),
    ("total_assets", "总资产", "financial", "单位:元"),
    ("revenue_yoy", "营收同比", "financial", "同比增速,单位 %"),
    ("profit_yoy", "净利同比", "financial", "同比增速,单位 %"),
]


@router.get("/fields", response_model=ScreenFieldsResponse)
async def list_screen_fields() -> ScreenFieldsResponse:
    """列出截面可筛选字段(供前端筛选条件下拉枚举)。"""
    return ScreenFieldsResponse(
        fields=[
            ScreenField(name=name, label=label, group=group, description=desc)
            for name, label, group, desc in _SCREEN_FIELDS
        ]
    )


@router.get("/markets", response_model=ScreenMarketsResponse)
async def list_screen_markets() -> ScreenMarketsResponse:
    """列出选股页各市场可用性(不可用市场前端置灰)。"""
    return ScreenMarketsResponse(markets=_SCREEN_MARKETS)


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


@router.get("", response_model=list[JobStatus])
async def list_screens(
    limit: int = 50, registry: JobRegistry = Depends(get_job_registry)
) -> list[JobStatus]:
    """列出历史选股任务(按更新时间倒序,刷新页面不丢任务)。"""
    jobs = registry.list(limit=limit, kind="screen")
    return [JobStatus(**job.to_dict()) for job in jobs]


@router.get("/{job_id}", response_model=JobStatus)
async def get_screen(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> JobStatus:
    """查询选股任务状态与结果(股票列表 + 得分在 ``result.results``)。"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return JobStatus(**job.to_dict())

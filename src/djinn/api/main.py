"""FastAPI 应用入口。"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from djinn.api.deps import get_cache, get_job_registry
from djinn.api.jobs import recover_orphaned_jobs
from djinn.api.routers import (
    backtests,
    data,
    factors,
    indicators,
    jobs,
    profiles,
    screens,
    stocks,
    strategies,
    sweeps,
    universe,
    walk_forwards,
)
from djinn.data import default_registry
from djinn.utils.logging import get_logger

logger = get_logger(__name__)

# 可选 Bearer token 鉴权(最小防线):设置 DJINN_API_TOKEN 后,除健康/文档端点外
# 所有请求须带 ``Authorization: Bearer <token>``;未设置则零配置放行。
_API_TOKEN = os.environ.get("DJINN_API_TOKEN")
_AUTH_EXEMPT = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


_PURGE_MARKER = ".cache/djinn_last_purge"


def _auto_purge_old_jobs(registry: Any) -> None:
    """E6:每日一次自动清理过期任务(保留 ``DJINN_JOB_RETENTION_DAYS`` 天)。

    用 marker 文件记录上次清理日期,避免每次启动(或热重载)都重复清理。
    """
    import datetime
    from pathlib import Path

    days = int(os.environ.get("DJINN_JOB_RETENTION_DAYS", "30"))
    marker = Path(_PURGE_MARKER)
    try:
        last = datetime.date.fromisoformat(marker.read_text().strip())
    except Exception:
        last = None
    if last == datetime.date.today():
        return
    try:
        n = registry.purge_older_than(days)
        if n:
            logger.info("自动清理 %d 个过期任务(保留 %d 天)", n, days)
    except Exception as e:  # 清理失败不影响启动
        logger.warning("自动清理过期任务失败: %s", e)
    from contextlib import suppress

    with suppress(Exception):
        marker.write_text(datetime.date.today().isoformat())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动钩子:恢复孤儿任务 + 每日一次自动清理过期任务。

    测试环境(TestClient 不用 ``with`` 不触发 lifespan,且 ``DJINN_TEST=1``
    守卫)不执行恢复,避免误恢复真实任务。
    """
    registry = get_job_registry()
    preg = default_registry(cache=get_cache())
    recovered = recover_orphaned_jobs(registry, preg)
    if recovered:
        logger.info("启动恢复 %d 个孤儿任务", recovered)
    _auto_purge_old_jobs(registry)
    # 预热用户策略编译(初始化 pynescript/antlr),首次请求策略列表不再卡 2.3s
    try:
        import time as _t

        from djinn.api.routers.strategies import warm_up_user_strategies

        _t0 = _t.perf_counter()
        warm_up_user_strategies()
        logger.info("用户策略编译预热完成(%.2fs)", _t.perf_counter() - _t0)
    except Exception as e:  # 预热失败不影响启动
        logger.warning("用户策略预热失败(首次请求将现场编译): %s", e)
    yield


app = FastAPI(
    title="Djinn Backtesting API",
    description="多市场量化回测框架 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS 配置(env 化,逗号分隔;默认允许本地 Vite dev server)
_cors_origins = [
    o.strip()
    for o in os.environ.get(
        "DJINN_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _auth_middleware(
    request: Request, call_next: Callable[[Request], Any]
) -> Any:
    """可选 Bearer token 鉴权(未设置 DJINN_API_TOKEN 时零开销放行)。"""
    if _API_TOKEN and request.url.path not in _AUTH_EXEMPT:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {_API_TOKEN}":
            return JSONResponse(status_code=401, content={"detail": "未授权"})
    return await call_next(request)


# 注册路由
app.include_router(strategies.router)
app.include_router(data.router)
app.include_router(backtests.router)
app.include_router(sweeps.router)
app.include_router(walk_forwards.router)
app.include_router(universe.router)
app.include_router(factors.router)
app.include_router(factors.analysis_router)
app.include_router(factors.matrix_router)
app.include_router(screens.router)
app.include_router(stocks.router)
app.include_router(profiles.router)
app.include_router(indicators.router)
app.include_router(jobs.router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """API 根路径健康检查。"""
    return {"status": "ok", "message": "Djinn Backtesting API is running"}


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """健康检查端点。"""
    return {"status": "healthy"}

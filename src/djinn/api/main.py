"""FastAPI 应用入口。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from djinn.api.deps import get_cache, get_job_registry
from djinn.api.jobs import recover_orphaned_jobs
from djinn.api.routers import (
    backtests,
    data,
    factors,
    screens,
    strategies,
    sweeps,
    universe,
)
from djinn.data import default_registry
from djinn.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动钩子:恢复进程重启前被中断的后台任务(见 recover_orphaned_jobs)。

    测试环境(TestClient 不用 ``with`` 不触发 lifespan,且 ``DJINN_TEST=1``
    守卫)不执行恢复,避免误恢复真实任务。
    """
    registry = get_job_registry()
    preg = default_registry(cache=get_cache())
    recovered = recover_orphaned_jobs(registry, preg)
    if recovered:
        logger.info("启动恢复 %d 个孤儿任务", recovered)
    yield


app = FastAPI(
    title="Djinn Backtesting API",
    description="多市场量化回测框架 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS 配置(允许前端访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(strategies.router)
app.include_router(data.router)
app.include_router(backtests.router)
app.include_router(sweeps.router)
app.include_router(universe.router)
app.include_router(factors.router)
app.include_router(factors.analysis_router)
app.include_router(factors.matrix_router)
app.include_router(screens.router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """API 根路径健康检查。"""
    return {"status": "ok", "message": "Djinn Backtesting API is running"}


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """健康检查端点。"""
    return {"status": "healthy"}

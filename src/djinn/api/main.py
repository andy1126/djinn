"""FastAPI 应用入口。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from djinn.api.routers import backtests, data, strategies, sweeps
from djinn.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Djinn Backtesting API",
    description="多市场量化回测框架 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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


@app.get("/", tags=["health"])
async def root():
    """API 根路径健康检查。"""
    return {"status": "ok", "message": "Djinn Backtesting API is running"}


@app.get("/health", tags=["health"])
async def health_check():
    """健康检查端点。"""
    return {"status": "healthy"}

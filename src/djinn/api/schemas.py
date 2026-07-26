"""FastAPI 请求 / 响应 pydantic 模型。

复用内核的 BacktestConfig 作为回测请求体;结果按需序列化为 JSON 友好结构。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from djinn.config.models import BacktestConfig


# ── 请求 ──────────────────────────────────────────────
class BacktestRequest(BaseModel):
    """创建回测任务请求(直接复用 BacktestConfig)。"""

    config: BacktestConfig


class SweepRequest(BaseModel):
    """参数扫描请求。"""

    config: BacktestConfig
    grid: dict[str, list[Any]]
    target: str = "sharpe"
    parallel: bool = True


class DataFetchRequest(BaseModel):
    """数据拉取请求。"""

    symbols: list[str]
    market: str | None = None
    start: str
    end: str
    adjust: str = "backward"
    csv_dir: str | None = None


# ── 响应 ──────────────────────────────────────────────
class JobCreated(BaseModel):
    job_id: str
    status: str = "pending"


class JobStatus(BaseModel):
    job_id: str
    title: str = ""
    status: str  # pending / running / done / error
    progress: float = 0.0
    stage: str = ""
    error: str | None = None
    result_path: str | None = None
    result: dict[str, Any] | None = None
    kind: str = ""


class StrategyInfo(BaseModel):
    name: str
    description: str = ""
    params: list[dict[str, Any]]


class StrategyListResponse(BaseModel):
    strategies: list[StrategyInfo]


class MetricsResponse(BaseModel):
    metrics: dict[str, Any]
    trade_stats: dict[str, Any]
    benchmark_stats: dict[str, Any] | None = None
    symbols: list[str]


class CurveResponse(BaseModel):
    """曲线数据(前端绘图用,index 为 ISO 日期字符串)。"""

    index: list[str]
    equity: list[float]
    benchmark: list[float] | None = None
    drawdown: list[float]


class PositionsResponse(BaseModel):
    index: list[str]
    positions: dict[str, list[float]]
    weights: dict[str, list[float]]


class MonthlyReturnsResponse(BaseModel):
    years: list[int]
    months: list[str]
    values: list[list[float | None]]


class SweepResultResponse(BaseModel):
    results: list[dict[str, Any]]
    target: str


class CacheEntry(BaseModel):
    file: str
    rows: int
    start: str | None = None
    end: str | None = None
    error: bool = False


class CacheResponse(BaseModel):
    entries: list[CacheEntry]


class SymbolSearchResult(BaseModel):
    symbol: str
    market: str
    name: str = ""


class SymbolSearchResponse(BaseModel):
    query: str
    results: list[SymbolSearchResult]

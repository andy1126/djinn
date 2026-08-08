"""FastAPI 请求 / 响应 pydantic 模型。

复用内核的 BacktestConfig 作为回测请求体;结果按需序列化为 JSON 友好结构。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from djinn.config.models import BacktestConfig
from djinn.screen.scoring import FactorScore
from djinn.screen.screener import ScreenCondition


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


# ── 股票池(universe)────────────────────────────────────
class UniverseStock(BaseModel):
    symbol: str
    name: str = ""
    market: str = ""


class UniverseStockListResponse(BaseModel):
    market: str | None = None
    count: int
    stocks: list[UniverseStock]


class IndexInfo(BaseModel):
    key: str
    name: str
    market: str


class IndexListResponse(BaseModel):
    indexes: list[IndexInfo]


class IndexComponentsResponse(BaseModel):
    index: str
    count: int
    symbols: list[str]


class IndustryCount(BaseModel):
    name: str
    count: int


class IndustryListResponse(BaseModel):
    industries: list[IndustryCount]


# ── 因子库 / 因子分析 ───────────────────────────────────
class FactorInfo(BaseModel):
    name: str
    category: str = "generic"
    description: str = ""
    params: list[dict[str, Any]]


class FactorListResponse(BaseModel):
    factors: list[FactorInfo]


class FactorAnalysisRequest(BaseModel):
    """单因子分析请求(后台任务):universe × 区间 → IC / 分层 / 衰减报告。"""

    factor: str = Field(..., description="因子名(见 GET /factors)")
    params: dict[str, Any] = Field(default_factory=dict, description="因子参数覆盖")
    index: str | None = Field(default=None, description="宽基指数键(如 CSI300)")
    symbols: list[str] | None = Field(
        default=None, description="显式标的池(优先于 index)"
    )
    market: str | None = None
    start: str
    end: str
    adjust: str = "backward"
    ic_method: str = Field(
        default="spearman", description="IC 相关方法 spearman/pearson"
    )
    n_quantiles: int = Field(default=5, ge=2, le=20, description="分层数")
    periods: list[int] = Field(default=[1, 5, 10], description="前向收益持有期(交易日)")


# ── 选股(screen)────────────────────────────────────────
class ScreenRequest(BaseModel):
    """截面选股请求(后台任务):条件过滤 + 可选多因子打分排序。"""

    conditions: list[ScreenCondition] = Field(
        default_factory=list, description="截面筛选条件(取交集)"
    )
    scores: list[FactorScore] = Field(
        default_factory=list, description="可选打分因子(空则不打分)"
    )
    top_n: int | None = Field(default=None, ge=1, description="按得分取前 N(需 scores)")
    index: str | None = Field(default=None, description="宽基指数键(如 CSI300)")
    symbols: list[str] | None = Field(
        default=None, description="显式候选池(优先于 index)"
    )
    market: str | None = None
    when: str | None = Field(default=None, description="截面日期(默认最近交易日)")
    lookback_days: int = Field(default=120, ge=20, description="打分行情回看窗口(日)")


class ScreenResultResponse(BaseModel):
    count: int
    results: list[dict[str, Any]]


# ── 多因子诊断(factor matrix)────────────────────────────
class FactorMatrixPoint(BaseModel):
    """多因子诊断中的一个因子条目(因子名 + 权重 + 方向 + 可选参数覆盖)。"""

    factor: str = Field(..., description="因子名(见 GET /factors)")
    weight: float = Field(default=1.0, description="权重(诊断展示用,不影响相关矩阵)")
    direction: Literal[1, -1] = Field(default=1, description="1=值越高越好,-1=越低越好")
    params: dict[str, Any] = Field(default_factory=dict, description="因子参数覆盖")


class FactorMatrixRequest(BaseModel):
    """多因子诊断请求(后台任务):universe × 区间 → 因子相关矩阵 + 各因子 IC 汇总。"""

    factors: list[FactorMatrixPoint] = Field(
        ..., min_length=2, max_length=8, description="2~8 个因子"
    )
    index: str | None = Field(default=None, description="宽基指数键(如 CSI300)")
    symbols: list[str] | None = Field(
        default=None, description="显式标的池(优先于 index)"
    )
    market: str | None = None
    start: str
    end: str
    adjust: str = "backward"
    ic_method: str = Field(
        default="spearman", description="IC 相关方法 spearman/pearson"
    )
    periods: list[int] = Field(default=[1, 5, 10], description="前向收益持有期(交易日)")

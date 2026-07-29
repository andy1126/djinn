"""pydantic 配置模型:BacktestConfig 为唯一权威配置。

CLI 与(Phase 2)FastAPI 都构造同一个 BacktestConfig 调用同一内核。
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from djinn.data.schema import Adjust, Market
from djinn.screen.screener import ScreenCondition


class UniverseConfig(BaseModel):
    """标的池与基准。

    标的来源(可叠加,解析后合并去重):
    - ``symbols``:显式标的列表;
    - ``index``:指数代码 / 名,取其成分股入池;
    - ``screen``:截面筛选条件,作用于上述候选池。

    ``factors`` + ``n_stocks`` 声明动态打分池(供选股策略 / API 解析,默认 None)。
    """

    model_config = ConfigDict(extra="forbid")
    symbols: list[str] = Field(default_factory=list, description="标的代码列表")
    benchmark: str | None = Field(
        default=None, description="基准代码(如 ^GSPC / 000300.SH)"
    )
    market: Market | None = Field(default=None, description="强制市场;None 自动推断")
    # Phase 3:动态股票池来源(均向后兼容,默认 None)
    index: str | None = Field(default=None, description="指数代码/名,取成分股入池")
    screen: list[ScreenCondition] | None = Field(
        default=None, description="截面筛选条件列表(作用于候选池)"
    )
    factors: dict[str, float] | None = Field(
        default=None, description="动态打分池:因子名 → 权重"
    )
    n_stocks: int | None = Field(default=None, gt=0, description="动态池大小(TopN)")

    @model_validator(mode="after")
    def _check_source(self) -> UniverseConfig:
        if not (self.symbols or self.index or self.screen):
            raise ValueError("universe 需至少提供 symbols / index / screen 之一")
        return self


class PeriodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: date
    end: date

    @model_validator(mode="after")
    def _check_range(self) -> PeriodConfig:
        if self.start >= self.end:
            raise ValueError(f"start({self.start}) 必须早于 end({self.end})")
        return self


class AccountConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_cash: float = Field(default=100000.0, gt=0)
    currency: str = Field(default="USD")
    t_plus_1: bool | None = Field(default=None, description="A 股自动启用;显式覆盖")


class SlippageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["zero", "none", "fixed_bps", "fixed", "random", "volume_share"] = (
        "zero"
    )
    bps: float = Field(default=5.0, ge=0)


class CommissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["default", "china", "us", "hk"] = "default"
    rate: float | None = Field(default=None, ge=0)
    min_commission: float | None = Field(default=None, ge=0)
    stamp_duty_rate: float | None = Field(default=None, ge=0)
    transfer_fee_rate: float | None = Field(default=None, ge=0)


class CostsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    commission: CommissionConfig = Field(default_factory=CommissionConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    enforce_price_limit: bool = True
    enforce_suspension: bool = True
    enforce_lot: bool = True


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="策略类名(如 MACrossover)")
    params: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    # Phase 3:因子组合策略(scope=portfolio,均向后兼容默认 None)
    scope: Literal["per_symbol", "portfolio"] | None = Field(
        default=None, description="策略作用域;portfolio 为整体调仓(选股)"
    )
    factor_weights: dict[str, float] | None = Field(
        default=None, description="因子名 → 权重(负值 = 因子值越低越好)"
    )
    n_stocks: int | None = Field(default=None, gt=0, description="选股数(TopN)")
    rebalance_freq: int | None = Field(
        default=None, gt=0, description="调仓间隔(交易日)"
    )


class RebalanceConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    period: Literal["none", "daily", "weekly", "monthly", "quarterly", "yearly"] = (
        "none"
    )
    threshold: float = Field(default=0.0, ge=0, le=1)
    min_hold_days: int = Field(default=0, ge=0)


class PortfolioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["single", "portfolio"] = "single"
    # 与 djinn.portfolio.allocation.AllocationType 保持一致(内联以免 config 重量级依赖)
    allocation: Literal[
        "equal",
        "market_cap",
        "custom",
        "score",
        "risk_parity",
        "min_variance",
        "mean_variance",
    ] = "equal"
    weights: dict[str, float] | None = None
    rebalance: RebalanceConfigModel = Field(default_factory=RebalanceConfigModel)


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_single_weight: float = Field(default=1.0, ge=0, le=1)
    max_total_position: float = Field(default=1.0, ge=0, le=1)
    max_sector_weight: float | None = None
    sector_map: dict[str, str] | None = None
    max_turnover: float | None = Field(default=None, ge=0)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dir: str = Field(default="./results")
    export: list[Literal["csv", "excel"]] = Field(default=["csv"])
    report: Literal["html", "none"] = "none"
    rolling_window: int = Field(default=63, ge=5)


class BacktestConfig(BaseModel):
    """回测完整配置(唯一权威)。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    universe: UniverseConfig
    period: PeriodConfig
    account: AccountConfig = Field(default_factory=AccountConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    strategy: StrategyConfig
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    adjust: Adjust = Adjust.BACKWARD
    risk_free_rate: float = Field(default=0.0, ge=0, le=1)

    @field_validator("adjust", mode="before")
    @classmethod
    def _parse_adjust(cls, v: object) -> Adjust:
        if isinstance(v, Adjust):
            return v
        return Adjust(str(v))

    def resolved_market(self) -> Market:
        """确定回测市场(universe.market,或由标的 / 指数推断)。"""
        if self.universe.market is not None:
            return self.universe.market
        from djinn.data.schema import detect_market

        if self.universe.symbols:
            return detect_market(self.universe.symbols[0])
        # 无显式标的(纯 index / screen 池):默认 A 股(akshare 免费主线)
        return Market.CN

    def resolved_t_plus_1(self) -> bool:
        if self.account.t_plus_1 is not None:
            return self.account.t_plus_1
        return self.resolved_market() is Market.CN

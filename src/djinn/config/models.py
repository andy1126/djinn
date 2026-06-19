"""pydantic 配置模型:BacktestConfig 为唯一权威配置。

CLI 与(Phase 2)FastAPI 都构造同一个 BacktestConfig 调用同一内核。
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from djinn.data.schema import Adjust, Market


class UniverseConfig(BaseModel):
    """标的池与基准。"""

    model_config = ConfigDict(extra="forbid")
    symbols: list[str] = Field(..., min_length=1, description="标的代码列表")
    benchmark: str | None = Field(
        default=None, description="基准代码(如 ^GSPC / 000300.SH)"
    )
    market: Market | None = Field(default=None, description="强制市场;None 自动推断")


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
    allocation: Literal["equal", "market_cap", "custom"] = "equal"
    weights: dict[str, float] | None = None
    rebalance: RebalanceConfigModel = Field(default_factory=RebalanceConfigModel)


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_single_weight: float = Field(default=1.0, ge=0, le=1)
    max_total_position: float = Field(default=1.0, ge=0, le=1)
    max_sector_weight: float | None = None
    sector_map: dict[str, str] | None = None


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
        """确定回测市场(universe.market 或由首个标的推断)。"""
        if self.universe.market is not None:
            return self.universe.market
        from djinn.data.schema import detect_market

        return detect_market(self.universe.symbols[0])

    def resolved_t_plus_1(self) -> bool:
        if self.account.t_plus_1 is not None:
            return self.account.t_plus_1
        return self.resolved_market() is Market.CN

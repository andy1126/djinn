"""pydantic 配置模型:BacktestConfig 为唯一权威配置。

CLI 与(Phase 2)FastAPI 都构造同一个 BacktestConfig 调用同一内核。
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from djinn.config.screen_models import ScreenCondition
from djinn.data.schema import Adjust, Market

# 市场 → 默认币种(currency=None 时按 resolved_market 映射)。
_CURRENCY_BY_MARKET: dict[Market, str] = {
    Market.CN: "CNY",
    Market.HK: "HKD",
    Market.US: "USD",
}


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
    # E11:None 时由 BacktestConfig 按 resolved_market() 映射(CN→CNY/HK→HKD/US→USD)
    currency: str | None = Field(default=None)
    t_plus_1: bool | None = Field(default=None, description="A 股自动启用;显式覆盖")


class SlippageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["zero", "fixed_bps", "fixed", "random", "volume_share"] = "zero"
    bps: float = Field(default=5.0, ge=0)

    @field_validator("type", mode="before")
    @classmethod
    def _migrate_none(cls, v: object) -> object:
        # E11:"none" 别名 → "zero"(旧配置迁移)
        if isinstance(v, str) and v.lower() == "none":
            warnings.warn(
                "slippage.type='none' 已废弃,请改用 'zero'",
                DeprecationWarning,
                stacklevel=2,
            )
            return "zero"
        return v


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
    fill_ref: Literal["open", "close", "vwap"] = "open"  # 成交参考价
    max_volume_share: float = Field(default=0.0, ge=0, le=1)  # 单笔成交占当日成交量上限


class SelectionConfig(BaseModel):
    """选股流水线增强(组合策略,G 计划)。"""

    model_config = ConfigDict(extra="forbid")
    min_amount: float | None = None  # 20 日平均成交额下限(元)
    min_list_days: int | None = None  # 上市最少交易日数
    exclude_st: bool = False
    neutralize: bool = False  # C5:打分前行业/市值中性化(需 industry_map + 市值面板)
    industry_neutral: bool = False
    max_sector_weight: float | None = None  # 行业暴露上限(0,1]
    min_score_diff: float = 0.0  # 换手惩罚阈值(zscore σ)


class TimingConfig(BaseModel):
    """择时覆盖层(组合策略,G 计划)。"""

    model_config = ConfigDict(extra="forbid")
    market_filter: dict[str, Any] | None = None  # {type:sma,window:200,floor:0.3}
    exit_rule: dict[str, Any] | None = None  # {type:sma_break|atr_trail,...}
    entry_confirm: dict[str, Any] | None = None  # {type:above_sma,window:20}
    cooldown_days: int = 5


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
    # C9:因子加权方式(static=手填 factor_weights;icir=滚动 ICIR 自动加权,符号自适配)
    weighting: Literal["static", "icir"] = "static"
    icir_window: int = Field(default=60, gt=0, description="滚动 ICIR 窗口(交易日)")
    icir_min_periods: int = Field(default=20, gt=0, description="滚动 ICIR 最少观测数")
    n_stocks: int | None = Field(default=None, gt=0, description="选股数(TopN)")
    rebalance_freq: int | None = Field(
        default=None, gt=0, description="调仓间隔(交易日)"
    )
    # G 计划:选股流水线 + 择时覆盖层(均可选)
    selection: SelectionConfig | None = None
    timing: TimingConfig | None = None


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
    # E11:默认不导出(避免 API 后台任务脏写 ./results);CLI 示例显式声明
    export: list[Literal["csv", "excel"]] = Field(default_factory=list)
    report: Literal["html", "none"] = "none"
    rolling_window: int = Field(default=63, ge=5)


class WalkForwardConfig(BaseModel):
    """Walk-Forward 分析:滚动样本外验证(H 计划)。

    ``period`` 为全区间,窗口在其内滚动;每个窗口:样本内(IS)网格搜索选参
    (按窗口独立选参),样本外(OOS)用 IS 最优参数评估;所有 OOS 段拼接成
    无前视的样本外净值。``min_is_sharpe`` 不达标则该窗口不部署(OOS 空仓)。
    """

    model_config = ConfigDict(extra="forbid")
    is_days: int = Field(default=250, gt=0, description="样本内(训练)窗口,交易日")
    oos_days: int = Field(default=125, gt=0, description="样本外(验证)窗口,交易日")
    step: int | None = Field(
        default=None, gt=0, description="滚动步长,默认=oos_days(非重叠)"
    )
    n_windows: int | None = Field(
        default=None, gt=0, description="窗口数上限,默认由区间推导"
    )
    target: str = Field(default="sharpe", description="IS 优化目标(与 sweep 同语义)")
    grid: dict[str, list[Any]] = Field(
        default_factory=dict, description="参数网格(与 sweep --grid 同格式)"
    )
    top_k: int = Field(
        default=1, gt=0, description="部署 IS 最优前 k 个组合(1=只部署最优)"
    )
    min_is_sharpe: float | None = Field(
        default=None, description="IS 目标不达标则该窗口 OOS 空仓(防过拟合)"
    )
    warmup_days: int = Field(
        default=300,
        ge=0,
        description="每窗口前置暖机交易日(≥ 因子 max_lookback 最稳)",
    )


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
    walk_forward: WalkForwardConfig | None = Field(
        default=None, description="Walk-Forward 分析配置(None=不启用)"
    )

    @field_validator("adjust", mode="before")
    @classmethod
    def _parse_adjust(cls, v: object) -> Adjust:
        if isinstance(v, Adjust):
            return v
        return Adjust(str(v))

    @model_validator(mode="after")
    def _resolve_currency(self) -> BacktestConfig:
        # E11:currency=None 时按市场映射币种
        if self.account.currency is None:
            self.account.currency = _CURRENCY_BY_MARKET.get(
                self.resolved_market(), "USD"
            )
        return self

    def resolved_market(self) -> Market:
        """确定回测市场(universe.market,或由标的 / 指数推断)。"""
        if self.universe.market is not None:
            return self.universe.market
        from djinn.data.schema import detect_market

        if self.universe.symbols:
            return detect_market(self.universe.symbols[0])
        # E11:纯 index 池按指数映射市场(HSI→HK、SP500/...→US);查不到再默认 CN
        if self.universe.index:
            from djinn.data.universe import UNIVERSE_INDEX_MAP

            entry = UNIVERSE_INDEX_MAP.get(self.universe.index.upper())
            if entry is not None:
                return entry["market"]  # type: ignore[return-value]
        # 无显式标的(纯 screen 池):默认 A 股(akshare 免费主线)
        return Market.CN

    def resolved_t_plus_1(self) -> bool:
        if self.account.t_plus_1 is not None:
            return self.account.t_plus_1
        return self.resolved_market() is Market.CN

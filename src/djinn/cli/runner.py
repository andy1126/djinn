"""回测运行器:把 BacktestConfig 串成完整回测流程。

CLI(`djinn run`)与(Phase 2)FastAPI 端点共用此模块,保证结果一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from djinn.analytics import build_report
from djinn.analytics.report import Report
from djinn.config.models import BacktestConfig
from djinn.data import (
    DataCache,
    MarketData,
    ProviderRegistry,
    default_registry,
    load_benchmark,
)
from djinn.data.schema import Market
from djinn.engine import (
    EngineConfig,
    EventDrivenEngine,
    TradeConstraints,
)
from djinn.engine.commission import make_commission
from djinn.engine.slippage import make_slippage
from djinn.io import export
from djinn.portfolio import (
    Allocation,
    CustomWeight,
    EqualWeight,
    MarketCapWeight,
    RebalanceConfig,
    Rebalancer,
    RiskLimits,
    RiskManager,
)
from djinn.strategy import Strategy
from djinn.strategy.library import get_strategy_class
from djinn.utils.exceptions import ConfigError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class RunResult:
    """单次回测运行结果(含报告与导出文件)。"""

    report: Report
    config: BacktestConfig
    exported_files: list[Path]


def build_engine_config(cfg: BacktestConfig) -> EngineConfig:
    """从 BacktestConfig 构造 EngineConfig(费用/滑点/约束/组合/风控)。"""
    market = cfg.resolved_market()

    # 佣金
    cc = cfg.costs.commission
    type_to_market = {"china": Market.CN, "us": Market.US, "hk": Market.HK}
    if cc.type == "default":
        comm = make_commission(market)
    else:
        comm_market = type_to_market.get(cc.type, market)
        kwargs: dict[str, float] = {}
        if cc.rate is not None:
            kwargs["rate"] = cc.rate
        if cc.min_commission is not None:
            kwargs["min_commission"] = cc.min_commission
        if cc.stamp_duty_rate is not None:
            kwargs["stamp_duty_rate"] = cc.stamp_duty_rate
        if cc.transfer_fee_rate is not None:
            kwargs["transfer_fee_rate"] = cc.transfer_fee_rate
        comm = make_commission(comm_market, **kwargs)

    # 滑点
    sc = cfg.costs.slippage
    slip_kwargs: dict[str, float] = {}
    if sc.bps is not None:
        slip_kwargs["bps"] = sc.bps
    slippage = make_slippage(sc.type, **slip_kwargs)

    # 约束
    con = TradeConstraints(
        market=market,
        enforce_lot=cfg.costs.enforce_lot,
        enforce_price_limit=cfg.costs.enforce_price_limit,
        enforce_suspension=cfg.costs.enforce_suspension,
        enforce_t_plus_1=cfg.resolved_t_plus_1(),
    )

    # 组合
    allocation: Allocation | None = None
    if cfg.portfolio.mode == "portfolio":
        if cfg.portfolio.allocation == "equal":
            allocation = EqualWeight()
        elif cfg.portfolio.allocation == "market_cap":
            allocation = MarketCapWeight()
        elif cfg.portfolio.allocation == "custom":
            if not cfg.portfolio.weights:
                raise ConfigError("custom 分配需要 portfolio.weights")
            allocation = CustomWeight(cfg.portfolio.weights)

    rebalancer = None
    rc = cfg.portfolio.rebalance
    if rc.period != "none" or rc.threshold > 0:
        rebalancer = Rebalancer(
            RebalanceConfig(
                period=rc.period, threshold=rc.threshold, min_hold_days=rc.min_hold_days
            )
        )

    risk = RiskManager(
        RiskLimits(
            max_single_weight=cfg.risk.max_single_weight,
            max_total_position=cfg.risk.max_total_position,
            max_sector_weight=cfg.risk.max_sector_weight,
            sector_map=cfg.risk.sector_map or {},
        )
    )

    return EngineConfig(
        initial_cash=cfg.account.initial_cash,
        currency=cfg.account.currency,
        commission=comm,
        slippage=slippage,
        constraints=con,
        allocation=allocation,
        rebalance=rebalancer,
        risk=risk,
    )


def build_strategy(cfg: BacktestConfig) -> Strategy:
    """按配置实例化策略。"""
    cls = get_strategy_class(cfg.strategy.name)
    return cls(**{k: v for k, v in cfg.strategy.params.items() if v is not None})


def run_backtest(
    cfg: BacktestConfig,
    *,
    registry: ProviderRegistry | None = None,
    csv_dir: str | None = None,
    cache: DataCache | None = None,
) -> RunResult:
    """执行完整回测:数据 → 引擎 → 报告 → 导出。"""
    market = cfg.resolved_market()
    if registry is None:
        registry = default_registry(csv_dir=csv_dir, cache=cache)

    # 拉取数据
    data: dict[str, MarketData] = {}
    for sym in cfg.universe.symbols:
        _log.info("拉取数据 %s [%s ~ %s]", sym, cfg.period.start, cfg.period.end)
        md = registry.get_ohlcv(
            sym, cfg.period.start, cfg.period.end, cfg.adjust, market=market
        )
        data[sym] = md

    # 基准
    benchmark: MarketData | None = None
    if cfg.universe.benchmark:
        try:
            benchmark = load_benchmark(
                registry,
                cfg.universe.benchmark,
                cfg.period.start,
                cfg.period.end,
                market=market,
                adjust=cfg.adjust,
            )
        except Exception as e:
            _log.warning("基准加载失败,跳过: %s", e)

    # 策略 + 引擎
    strategy = build_strategy(cfg)
    engine_cfg = build_engine_config(cfg)
    engine = EventDrivenEngine(engine_cfg)
    result = engine.run(strategy, data, benchmark=benchmark)

    # 报告
    report = build_report(
        result,
        market=market.value,
        rf=cfg.risk_free_rate,
        rolling_window=cfg.output.rolling_window,
    )

    # 导出
    out_dir = Path(cfg.output.dir)
    files: list[Path] = []
    if cfg.output.export:
        files = export(report, out_dir, cfg.output.export)
    if cfg.output.report == "html":
        from djinn.viz import save_html_report

        files.append(save_html_report(report, out_dir / "report.html"))
    return RunResult(report=report, config=cfg, exported_files=files)

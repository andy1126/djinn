"""回测运行器:把 BacktestConfig 串成完整回测流程。

CLI(`djinn run`)与(Phase 2)FastAPI 端点共用此模块,保证结果一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

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
from djinn.data.providers.fundamentals_router import FundamentalsRouter
from djinn.data.schema import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_VOLUME,
    Market,
)
from djinn.engine import (
    EngineConfig,
    EventDrivenEngine,
    TradeConstraints,
)
from djinn.engine.commission import make_commission
from djinn.engine.slippage import make_slippage
from djinn.factor import make_factor
from djinn.io import export
from djinn.portfolio import (
    Allocation,
    RebalanceConfig,
    Rebalancer,
    RiskLimits,
    RiskManager,
    make_allocation,
)
from djinn.screen import FactorScore, Screener
from djinn.strategy import Strategy
from djinn.strategy.library import get_strategy_class
from djinn.strategy.library.factor_portfolio import FactorPortfolioStrategy
from djinn.utils.exceptions import ConfigError, StrategyError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class RunResult:
    """单次回测运行结果(含报告与导出文件)。"""

    report: Report
    config: BacktestConfig
    exported_files: list[Path]


def _build_allocation(cfg: BacktestConfig) -> Allocation:
    """按 ``cfg.portfolio.allocation`` 构造分配器(默认等权)。

    ``score`` / ``risk_parity`` / ``min_variance`` / ``mean_variance`` 所需的
    ``scores`` / ``cov`` 由策略在调仓时传入(见 :class:`FactorPortfolioStrategy`)。
    """
    try:
        return make_allocation(cfg.portfolio.allocation, cfg.portfolio.weights)
    except StrategyError as e:
        raise ConfigError(str(e)) from e


def _is_portfolio_scope(cfg: BacktestConfig) -> bool:
    """是否为整体调仓(选股)策略——由策略自行再平衡,引擎不再注入调仓单。"""
    return cfg.strategy.scope == "portfolio" or cfg.strategy.name == "FactorPortfolio"


def build_engine_config(cfg: BacktestConfig) -> EngineConfig:
    """从 BacktestConfig 构造 EngineConfig(费用/滑点/约束/组合/风控)。"""
    market = cfg.resolved_market()
    portfolio_scope = _is_portfolio_scope(cfg)

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

    # 组合:选股(portfolio)策略自行再平衡,引擎侧不注入分配 / 调仓单
    allocation: Allocation | None = None
    if not portfolio_scope and cfg.portfolio.mode == "portfolio":
        allocation = _build_allocation(cfg)

    rebalancer = None
    rc = cfg.portfolio.rebalance
    if not portfolio_scope and (rc.period != "none" or rc.threshold > 0):
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
            max_turnover=cfg.risk.max_turnover,
        )
    )

    # 选股回测用并集日历(成分时变 + 前向填充估值);普通回测仍取交集
    calendar: Literal["intersection", "union"] = (
        "union" if portfolio_scope else "intersection"
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
        calendar=calendar,
    )


def build_strategy(
    cfg: BacktestConfig, *, fundamentals: dict[str, pd.DataFrame] | None = None
) -> Strategy:
    """按配置实例化策略(因子组合策略走专属构造)。"""
    if _is_portfolio_scope(cfg):
        return _build_factor_portfolio(cfg, fundamentals=fundamentals)
    cls = get_strategy_class(cfg.strategy.name)
    return cls(**{k: v for k, v in cfg.strategy.params.items() if v is not None})


def _build_factor_portfolio(
    cfg: BacktestConfig, *, fundamentals: dict[str, pd.DataFrame] | None = None
) -> FactorPortfolioStrategy:
    """构造多因子打分 TopN 组合策略。

    因子权重来自 ``strategy.factor_weights``(或 ``universe.factors``);权重为负表示
    "因子值越低越好"(如波动率)。因子用默认参数实例化,定制参数请直接构造策略。
    """
    fw = cfg.strategy.factor_weights or cfg.universe.factors or {}
    if not fw:
        raise ConfigError(
            "FactorPortfolio 需要 strategy.factor_weights(或 universe.factors)"
        )
    factors = [make_factor(name) for name in fw]
    scores = [FactorScore(factor=name, weight=float(w)) for name, w in fw.items()]
    n_stocks = cfg.strategy.n_stocks or cfg.universe.n_stocks or 10
    rebalance_freq = cfg.strategy.rebalance_freq or 20
    return FactorPortfolioStrategy(
        factors=factors,
        scores=scores,
        n_stocks=n_stocks,
        rebalance_freq=rebalance_freq,
        allocation=_build_allocation(cfg),
        fundamentals=fundamentals,
    )


def _resolve_universe_symbols(
    cfg: BacktestConfig, registry: ProviderRegistry, market: Market
) -> list[str]:
    """解析标的池:symbols ∪ index 成分,再经 screen 截面筛选。"""
    symbols = list(cfg.universe.symbols)
    if cfg.universe.index:
        comps = _index_components(cfg.universe.index, registry)
        symbols = list(dict.fromkeys([*symbols, *comps]))
    if cfg.universe.screen:
        if not symbols:
            raise ConfigError("screen 需配合 symbols 或 index 提供候选池")
        snap = FundamentalsRouter(registry.providers).get_snapshot(
            symbols, cfg.period.end, market
        )
        symbols = Screener.apply(list(cfg.universe.screen), snap)
    if not symbols:
        raise ConfigError("universe 解析后为空标的")
    _log.info("universe 解析:%d 只标的", len(symbols))
    return symbols


def _index_components(index: str, registry: ProviderRegistry) -> list[str]:
    """从首个支持指数成分的 provider 取成分股。"""
    for p in registry.providers:
        try:
            comps = p.get_index_components(index)
        except NotImplementedError:
            continue
        except Exception as e:
            _log.warning("provider %s 取指数 %s 成分失败: %s", p.name, index, e)
            continue
        if comps:
            _log.info("指数 %s 成分 %d 只(来自 %s)", index, len(comps), p.name)
            return comps
    raise ConfigError(f"无 provider 能提供指数 {index!r} 的成分股")


def _try_fundamental_panels(
    cfg: BacktestConfig,
    data: dict[str, MarketData],
    registry: ProviderRegistry,
    market: Market,
) -> dict[str, pd.DataFrame] | None:
    """为因子组合策略构建 point-in-time 基本面宽表(失败退化为纯行情因子)。"""
    from djinn.factor.engine import DEFAULT_FUNDAMENTAL_FIELDS, FactorEngine

    symbols = list(data.keys())
    idx = pd.DatetimeIndex([])
    for md in data.values():
        idx = idx.union(pd.DatetimeIndex(md.df.index))
    eng = FactorEngine()
    try:
        return eng._fundamental_panels(
            DEFAULT_FUNDAMENTAL_FIELDS,
            symbols,
            idx.sort_values(),
            cfg.period.start,
            cfg.period.end,
            FundamentalsRouter(registry.providers),
            market,
        )
    except Exception as e:
        _log.warning("基本面面板构建失败,仅用行情类因子: %s", e)
        return None


# ── 归因(Phase 5 接线)──────────────────────────────────
def _industry_map_safe(
    registry: ProviderRegistry, symbols: list[str]
) -> dict[str, str]:
    """尽力取 symbol → 行业映射(任一 provider 成功即用,全失败返回 {})。"""
    for p in registry.providers:
        try:
            m = p.get_industry_map(symbols)
        except NotImplementedError:
            continue
        except Exception as e:
            _log.warning("provider %s 取行业映射失败: %s", p.name, e)
            continue
        if m:
            return {str(k): str(v) for k, v in m.items()}
    return {}


def _ohlcv_from_data(
    data: dict[str, MarketData],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """由已拉取的 ``{symbol: MarketData}`` 组收盘价宽表与 OHLCV 字段宽表(无 I/O)。"""
    closes = {s: md.df[COL_CLOSE] for s, md in data.items()}
    prices = pd.DataFrame(closes).sort_index()
    ohlcv: dict[str, pd.DataFrame] = {}
    for c in (COL_OPEN, COL_HIGH, COL_LOW, COL_VOLUME, COL_AMOUNT):
        cols = {s: md.df[c] for s, md in data.items() if c in md.df.columns}
        if cols:
            ohlcv[c] = pd.DataFrame(cols).reindex(prices.index)
    return prices, ohlcv


def _attribution_payloads(
    weights: pd.DataFrame,
    data: dict[str, MarketData],
    industry_map: dict[str, str],
    factor_panels: dict[str, pd.DataFrame] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """计算 Brinson 行业归因 + (可选)因子暴露/行业分布,返回已序列化 dict。

    Brinson 基准取**交易宇宙等权篮子**(指数基准无成分权重,这是最贴近"选股 /
    配置是否跑赢 naive 篮子"的诚实口径)。任一环节失败退化为 None,不影响主报告。
    """
    from djinn.analytics import brinson_attribution, build_exposure_report

    symbols = [str(c) for c in weights.columns]
    if not symbols:
        return None, None
    rets = {s: data[s].df[COL_CLOSE].pct_change() for s in symbols if s in data}
    returns = pd.DataFrame(rets)
    brinson_d: dict[str, Any] | None = None
    if not returns.empty:
        bench_w = dict.fromkeys(symbols, 1.0 / len(symbols))
        try:
            brinson_d = brinson_attribution(
                weights, bench_w, returns, industry_map
            ).to_dict()
        except Exception as e:
            _log.warning("Brinson 行业归因失败: %s", e)
    exposure_d: dict[str, Any] | None = None
    if factor_panels:
        try:
            exposure_d = build_exposure_report(
                weights, factor_panels, industry_map
            ).to_dict()
        except Exception as e:
            _log.warning("因子暴露 / 行业分布报告失败: %s", e)
    return brinson_d, exposure_d


def _attach_attribution(
    report: Report,
    cfg: BacktestConfig,
    result: Any,
    data: dict[str, MarketData],
    registry: ProviderRegistry,
    market: Market,
    fundamentals: dict[str, pd.DataFrame] | None,
) -> None:
    """把 Brinson 行业归因 + 因子暴露/行业分布填充进 report(in-place)。"""
    weights = result.weights_curve
    if weights is None or weights.empty:
        return
    symbols = [str(c) for c in weights.columns]
    industry_map = _industry_map_safe(registry, symbols)
    # 因子暴露仅对声明了因子集的选股策略有意义
    factor_panels: dict[str, pd.DataFrame] | None = None
    fw = cfg.strategy.factor_weights or cfg.universe.factors
    if _is_portfolio_scope(cfg) and fw:
        try:
            prices, ohlcv = _ohlcv_from_data(data)
            fund = fundamentals or {}
            factor_panels = {
                f.name: f.compute(prices, ohlcv, fund)
                for f in (make_factor(n) for n in fw)
            }
        except Exception as e:
            _log.warning("因子面板构建失败,跳过因子暴露: %s", e)
            factor_panels = None
    brinson_d, exposure_d = _attribution_payloads(
        weights, data, industry_map, factor_panels
    )
    report.attribution = brinson_d
    report.factor_exposure = exposure_d


def run_backtest(
    cfg: BacktestConfig,
    *,
    registry: ProviderRegistry | None = None,
    csv_dir: str | None = None,
    cache: DataCache | None = None,
    with_attribution: bool = False,
) -> RunResult:
    """执行完整回测:数据 → 引擎 → 报告 → 导出。

    ``with_attribution=True`` 时额外计算 Brinson 行业归因与(因子组合策略)
    因子暴露 / 行业分布,填充进 ``report.attribution`` / ``report.factor_exposure``
    (供 Web 报告端点;CLI 默认关闭以避免额外的行业映射网络开销)。
    """
    market = cfg.resolved_market()
    if registry is None:
        registry = default_registry(csv_dir=csv_dir, cache=cache)

    # 解析标的池(symbols ∪ index 成分,再经 screen 截面筛选)
    symbols = _resolve_universe_symbols(cfg, registry, market)

    # 拉取数据
    data: dict[str, MarketData] = {}
    for sym in symbols:
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

    # 策略(因子组合策略注入 point-in-time 基本面面板)+ 引擎
    fundamentals: dict[str, pd.DataFrame] | None = None
    if _is_portfolio_scope(cfg):
        fundamentals = _try_fundamental_panels(cfg, data, registry, market)
    strategy = build_strategy(cfg, fundamentals=fundamentals)
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

    # 归因(可选,Web 报告端点开启)
    if with_attribution:
        try:
            _attach_attribution(
                report, cfg, result, data, registry, market, fundamentals
            )
        except Exception as e:
            _log.warning("归因计算失败(主报告不受影响): %s", e)

    # 导出
    out_dir = Path(cfg.output.dir)
    files: list[Path] = []
    if cfg.output.export:
        files = export(report, out_dir, cfg.output.export)
    if cfg.output.report == "html":
        from djinn.viz import save_html_report

        files.append(save_html_report(report, out_dir / "report.html"))
    return RunResult(report=report, config=cfg, exported_files=files)

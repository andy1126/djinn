"""djinn sweep:参数扫描(joblib 并行)。

支持多轴扫描(白名单见 ``ALLOWED_SWEEP_AXES``):策略裸参数 / universe.index
/ strategy.factor_weights / portfolio.allocation / strategy.n_stocks /
strategy.rebalance_freq。非白名单 key 在 ``_apply_param`` 走旧形(顶层
``strategy.params``)兜底,但路由层会先于此后拦截未知轴。
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import typer

from djinn.analytics import build_report
from djinn.cli.runner import build_engine_config, build_strategy
from djinn.config import BacktestConfig, load_config
from djinn.data import DataCache, ProviderRegistry, default_registry
from djinn.engine import EventDrivenEngine
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

# sweep 可扫轴白名单(前后端共享语义;前端硬编码同列表并注释同步)。
# 裸策略参数(fast/slow 等)不在此列——_apply_param 兜底写入 strategy.params。
ALLOWED_SWEEP_AXES: list[str] = [
    "universe.index",
    "strategy.factor_weights",
    "portfolio.allocation",
    "strategy.n_stocks",
    "strategy.rebalance_freq",
    "strategy.min_score_diff",
]

# 目标指标中"值越小越好"的集合:命中则升序(最优在前)。仅放波动率类;
# ``max_drawdown`` **不在此列**——它存为 ≤0 的负值,值越大(越接近 0)回撤越浅
# 越好,故与其余指标同走默认降序。若误放此处升序,最深的回撤会排到最前。
REVERSE_MIN_TARGETS: set[str] = {"volatility", "annual_volatility"}


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """参数网格展开为所有组合。"""
    keys = list(grid.keys())
    if not keys:
        return [{}]
    return [
        dict(zip(keys, vals, strict=False))
        for vals in itertools.product(*[grid[k] for k in keys])
    ]


def _index_symbols(index: str, registry: ProviderRegistry) -> list[str]:
    """从首个支持指数成分的 provider 取成分股(复用 runner 逻辑,宽松失败)。"""
    for p in registry.providers:
        try:
            comps = p.get_index_components(index)
        except NotImplementedError:
            continue
        except Exception as e:
            _log.warning("provider %s 取指数 %s 成分失败: %s", p.name, index, e)
            continue
        if comps:
            return [str(s) for s in comps]
    return []


def _apply_param(cfg: BacktestConfig, key: str, value: Any) -> None:
    """扫一个轴;path-prefix 写入对应子字段,兜底走 strategy.params。

    ``universe.index`` 扫轴时仅改 index,symbols 由 ``_run_one`` 在执行前重新解析
    (成分股随 index 变化),避免此处重复查询。
    """
    if key == "universe.index":
        cfg.universe.index = str(value)
    elif key == "strategy.factor_weights":
        cfg.strategy.factor_weights = (
            {str(k): float(v) for k, v in dict(value).items()}
            if value is not None
            else None
        )
    elif key == "portfolio.allocation":
        cfg.portfolio.allocation = str(value)  # type: ignore[assignment]
    elif key == "strategy.n_stocks":
        cfg.strategy.n_stocks = int(value)
    elif key == "strategy.rebalance_freq":
        cfg.strategy.rebalance_freq = int(value)
    elif key == "strategy.min_score_diff":
        # G 计划:换手惩罚阈值;selection 为 None 时先建默认实例
        if cfg.strategy.selection is None:
            from djinn.config.models import SelectionConfig

            cfg.strategy.selection = SelectionConfig()
        cfg.strategy.selection.min_score_diff = float(value)
    else:
        # 兼容旧形:顶层策略参数(如 fast/slow)
        cfg.strategy.params[key] = value


def _config_summary(cfg: BacktestConfig) -> dict[str, Any]:
    """本次组合真用上的关键轴摘要(供结果表展示)。"""
    return {
        "strategy": cfg.strategy.name,
        "universe.index": cfg.universe.index,
        "n_symbols": len(cfg.universe.symbols),
        "strategy.factor_weights": cfg.strategy.factor_weights,
        "portfolio.allocation": cfg.portfolio.allocation,
        "strategy.n_stocks": cfg.strategy.n_stocks,
        "strategy.rebalance_freq": cfg.strategy.rebalance_freq,
        "strategy.params": dict(cfg.strategy.params),
    }


def _run_one(
    cfg: BacktestConfig,
    registry: ProviderRegistry,
    params: dict[str, Any],
    target: str = "sharpe",
) -> dict[str, Any]:
    """单次参数组合回测,返回 {params, config_summary, target, ...metrics}。"""
    # 应用可扫轴(可能改 universe.index / factor_weights / allocation / n_stocks / rebalance_freq)
    for k, v in params.items():
        _apply_param(cfg, k, v)
    # 若扫了 universe.index,需在此重新解析成分股写入 universe.symbols。
    if cfg.universe.index and params.get("universe.index") is not None:
        comps = _index_symbols(cfg.universe.index, registry)
        if comps:
            cfg.universe.symbols = list(dict.fromkeys(comps))
    strategy = build_strategy(cfg)
    engine_cfg = build_engine_config(cfg)
    engine = EventDrivenEngine(engine_cfg)
    market = cfg.resolved_market()
    data = {
        sym: registry.get_ohlcv(
            sym, cfg.period.start, cfg.period.end, cfg.adjust, market=market
        )
        for sym in cfg.universe.symbols
    }
    result = engine.run(strategy, data)
    report = build_report(result, market=market.value, rf=cfg.risk_free_rate)
    m = report.metrics
    metric_val = getattr(m, target, 0.0)
    if metric_val is None:
        metric_val = 0.0
    return {
        "params": params,
        "config_summary": _config_summary(cfg),
        target: float(metric_val),
        "total_return": m.total_return,
        "annual_return": m.annual_return,
        "max_drawdown": m.max_drawdown,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "calmar": m.calmar,
        "volatility": m.annual_volatility,
        "n_trades": m.n_trades,
    }


def sweep_command(
    config: Path = typer.Option(..., "-c", "--config", help="YAML 配置文件"),
    grid: str = typer.Option(
        ..., "--grid", help='参数网格 JSON,如 \'{"fast":[5,10],"slow":[20,30]}\''
    ),
    target: str = typer.Option("sharpe", "--target", help="优化目标指标"),
    top: int = typer.Option(10, "--top", help="显示前 N 个最优组合"),
    output: Path | None = typer.Option(None, "-o", "--output", help="导出结果 JSON"),
    parallel: bool = typer.Option(True, "--parallel/--no-parallel", help="joblib 并行"),
    csv_dir: Path | None = typer.Option(None, "--csv-dir", help="本地 CSV 数据目录"),
) -> None:
    """参数扫描:在策略参数网格上搜索最优组合。"""
    cfg = load_config(config)
    try:
        grid_dict = json.loads(grid)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"grid JSON 解析失败: {e}") from e
    combos = _expand_grid(grid_dict)
    typer.echo(f"参数网格:{len(combos)} 种组合,目标指标={target}")

    cache = DataCache()
    registry = default_registry(csv_dir=str(csv_dir) if csv_dir else None, cache=cache)

    # 预拉数据:universe.symbols + 所有扫到的 index 的成分(实际各组合符号会覆盖)
    market = cfg.resolved_market()
    all_symbols: set[str] = set(cfg.universe.symbols)
    index_vals = grid_dict.get("universe.index", []) or []
    for idx in index_vals:
        all_symbols.update(_index_symbols(str(idx), registry))
    for sym in all_symbols:
        registry.get_ohlcv(
            sym, cfg.period.start, cfg.period.end, cfg.adjust, market=market
        )

    if parallel and len(combos) > 1:
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=-1)(
                delayed(_run_one)(cfg, registry, c, target) for c in combos
            )
        except ImportError:
            results = [_run_one(cfg, registry, c, target) for c in combos]
    else:
        results = [_run_one(cfg, registry, c, target) for c in combos]

    # 排序:REVERSE_MIN_TARGETS 越小越好 → 升序;默认降序。
    # B4:NaN/缺失目标排最后(升序时用 +inf,降序时用 -inf)。
    reverse = target not in REVERSE_MIN_TARGETS
    nan_val = float("-inf") if reverse else float("inf")

    def _key(r: dict[str, Any]) -> float:
        v = r.get(target)
        if v is None:
            return nan_val
        try:
            f = float(v)
        except (TypeError, ValueError):
            return nan_val
        return f if math.isfinite(f) else nan_val

    results.sort(key=_key, reverse=reverse)
    direction = "升序" if not reverse else "降序"
    typer.echo(f"\n=== Top {min(top, len(results))} 组合(按 {target} {direction})===")
    for i, r in enumerate(results[:top]):
        typer.echo(
            f"{i + 1}. params={r['params']}  {target}={r.get(target, 0):.3f}  "
            f"sharpe={r['sharpe']:.3f} sortino={r['sortino']:.3f} calmar={r['calmar']:.3f}  "
            f"ret={r['total_return']:.2%}  mdd={r['max_drawdown']:.2%}  trades={r['n_trades']}"
        )

    if output:
        Path(output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str)
        )
        typer.echo(f"\n结果已导出: {output}")

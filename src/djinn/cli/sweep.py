"""djinn sweep:参数扫描(joblib 并行)。"""

from __future__ import annotations

import itertools
import json
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


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """参数网格展开为所有组合。"""
    keys = list(grid.keys())
    if not keys:
        return [{}]
    return [
        dict(zip(keys, vals, strict=False))
        for vals in itertools.product(*[grid[k] for k in keys])
    ]


def _run_one(
    cfg: BacktestConfig,
    registry: ProviderRegistry,
    params: dict[str, Any],
    target: str = "sharpe",
) -> dict[str, Any]:
    """单次参数组合回测,返回 {params, metric}。"""
    # 覆盖策略参数
    cfg.strategy.params.update(params)
    strategy = build_strategy(cfg)
    engine_cfg = build_engine_config(cfg)
    engine = EventDrivenEngine(engine_cfg)
    # 数据已在 registry 缓存
    market = cfg.resolved_market()
    data = {
        sym: registry.get_ohlcv(
            sym, cfg.period.start, cfg.period.end, cfg.adjust, market=market
        )
        for sym in cfg.universe.symbols
    }
    result = engine.run(strategy, data)
    report = build_report(result, market=market.value, rf=cfg.risk_free_rate)
    metric_val = getattr(report.metrics, target, 0.0)
    return {
        "params": params,
        target: metric_val,
        "total_return": report.metrics.total_return,
        "max_drawdown": report.metrics.max_drawdown,
        "n_trades": report.metrics.n_trades,
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

    # 预拉取数据(避免每个 worker 重复拉取)
    market = cfg.resolved_market()
    for sym in cfg.universe.symbols:
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

    # 排序
    results.sort(key=lambda r: r.get(target, 0.0), reverse=True)
    typer.echo(f"\n=== Top {min(top, len(results))} 组合(按 {target} 降序)===")
    for i, r in enumerate(results[:top]):
        typer.echo(
            f"{i + 1}. params={r['params']}  {target}={r.get(target, 0):.3f}  "
            f"ret={r['total_return']:.2%}  mdd={r['max_drawdown']:.2%}  trades={r['n_trades']}"
        )

    if output:
        Path(output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str)
        )
        typer.echo(f"\n结果已导出: {output}")

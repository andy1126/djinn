"""djinn data:数据拉取 / 缓存管理。"""

from __future__ import annotations

from pathlib import Path

import typer

from djinn.config import load_config
from djinn.data import DataCache, default_registry
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

data_app = typer.Typer(name="data", help="数据拉取与缓存管理", no_args_is_help=True)


@data_app.command("fetch")
def fetch(
    config: Path = typer.Option(..., "-c", "--config", help="YAML 配置文件"),
    csv_dir: Path | None = typer.Option(None, "--csv-dir", help="本地 CSV 数据目录"),
) -> None:
    """按配置拉取标的 + 基准数据并缓存。"""
    cfg = load_config(config)
    cache = DataCache()
    registry = default_registry(csv_dir=str(csv_dir) if csv_dir else None, cache=cache)
    market = cfg.resolved_market()
    for sym in cfg.universe.symbols:
        typer.echo(f"拉取 {sym} ...")
        md = registry.get_ohlcv(
            sym, cfg.period.start, cfg.period.end, cfg.adjust, market=market
        )
        typer.echo(f"  {md.info()}")
    if cfg.universe.benchmark:
        typer.echo(f"拉取基准 {cfg.universe.benchmark} ...")
        from djinn.data import load_benchmark

        bm = load_benchmark(
            registry,
            cfg.universe.benchmark,
            cfg.period.start,
            cfg.period.end,
            market=market,
            adjust=cfg.adjust,
        )
        typer.echo(f"  {bm.info()}")
    typer.echo("数据拉取完成。")


@data_app.command("cache")
def cache_list() -> None:
    """列出缓存条目。"""
    cache = DataCache()
    entries = cache.list_entries()
    if not entries:
        typer.echo("缓存为空。")
        return
    typer.echo(f"{'文件':<40} {'行数':>8} {'起':<12} {'止':<12}")
    for e in entries:
        typer.echo(
            f"{e.get('file','')!s:<40} {e.get('rows',-1):>8} {e.get('start','')!s:<12} {e.get('end','')!s:<12}"
        )


@data_app.command("clear-cache")
def clear_cache() -> None:
    """清空缓存。"""
    DataCache().clear()
    typer.echo("缓存已清空。")

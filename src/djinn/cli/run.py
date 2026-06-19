"""djinn run:单次回测。"""

from __future__ import annotations

from pathlib import Path

import typer

from djinn.cli.runner import run_backtest
from djinn.config import load_config
from djinn.data import DataCache, default_registry
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


def run_command(
    config: Path = typer.Option(..., "-c", "--config", help="YAML 配置文件"),
    csv_dir: Path | None = typer.Option(
        None, "--csv-dir", help="本地 CSV 数据目录(优先于网络 provider)"
    ),
    no_export: bool = typer.Option(False, "--no-export", help="不导出文件,仅打印指标"),
    print_metrics: bool = typer.Option(
        True, "--print-metrics/--no-print-metrics", help="打印指标摘要"
    ),
) -> None:
    """按配置文件运行回测,输出指标与导出文件。"""
    cfg = load_config(config)
    cache = DataCache()
    registry = default_registry(csv_dir=str(csv_dir) if csv_dir else None, cache=cache)
    result = run_backtest(cfg, registry=registry, cache=cache)
    report = result.report

    if print_metrics:
        typer.echo("\n=== 回测摘要 ===")
        typer.echo(f"标的: {', '.join(report.symbols)}")
        typer.echo(f"交易日数: {report.metrics.n_days}")
        typer.echo(
            f"成交笔数: {report.metrics.n_trades}  拒单: {len(report.rejections)}"
        )
        m = report.metrics
        typer.echo(f"累计收益: {m.total_return:.2%}")
        typer.echo(f"年化收益: {m.annual_return:.2%}")
        typer.echo(f"年化波动: {m.annual_volatility:.2%}")
        typer.echo(f"夏普比率: {m.sharpe:.3f}")
        typer.echo(f"索提诺:   {m.sortino:.3f}")
        typer.echo(f"最大回撤: {m.max_drawdown:.2%}")
        typer.echo(f"Calmar:   {m.calmar:.3f}")
        typer.echo(f"换手率:   {m.turnover:.2f}")
        if report.benchmark_stats is not None:
            b = report.benchmark_stats
            typer.echo("--- 基准对比 ---")
            typer.echo(f"基准收益: {b.benchmark_return:.2%}")
            typer.echo(f"超额收益: {b.excess_return:.2%}")
            typer.echo(f"Beta:     {b.beta:.3f}")
            typer.echo(f"跟踪误差: {b.tracking_error:.2%}")
            typer.echo(f"信息比率: {b.information_ratio:.3f}")

    if not no_export and result.exported_files:
        typer.echo("\n=== 导出 ===")
        for f in result.exported_files:
            typer.echo(f"  {f}")
    typer.echo("\n回测完成。")

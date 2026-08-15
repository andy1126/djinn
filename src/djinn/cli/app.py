"""djinn CLI:run / sweep / data(typer)。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from djinn.cli.data import data_app
from djinn.cli.run import run_command
from djinn.cli.sweep import sweep_command
from djinn.cli.walk_forward import walk_forward
from djinn.config import dump_config, load_config
from djinn.data import DataCache, default_registry
from djinn.utils.logging import get_logger, set_log_level

app = typer.Typer(
    name="djinn",
    help="Djinn — 多市场量化回测框架(run / sweep / walk / data)",
    no_args_is_help=True,
    add_completion=False,
)
app.command(name="run")(run_command)
app.command(name="sweep")(sweep_command)
app.add_typer(data_app, name="data")

_log = get_logger(__name__)


@app.callback()
def main(
    log_level: str = typer.Option("WARNING", "--log-level", "-l", help="日志级别"),
    version: bool = typer.Option(False, "--version", "-V", help="显示版本"),
) -> None:
    """djinn 顶层选项。"""
    if version:
        from djinn import __version__

        typer.echo(f"djinn {__version__}")
        raise typer.Exit()
    set_log_level(log_level)


@app.command(name="show-config")
def show_config(
    config: Path = typer.Option(..., "-c", "--config", help="YAML 配置文件"),
    output: Path | None = typer.Option(None, "-o", "--output", help="导出规范化 YAML"),
) -> None:
    """加载并校验配置,打印或导出规范化 YAML。"""
    cfg = load_config(config)
    if output:
        dump_config(cfg, output)
        typer.echo(f"已导出: {output}")
    else:
        typer.echo(
            json.dumps(
                cfg.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str
            )
        )


def walk_command(
    config: Path = typer.Option(
        ..., "-c", "--config", help="YAML 配置文件(需含 walk_forward 段)"
    ),
    grid: str | None = typer.Option(
        None,
        "--grid",
        help='参数网格 JSON,覆盖配置,如 \'{"fast":[5,10,20],"slow":[20,30,60]}\'',
    ),
    is_days: int | None = typer.Option(None, "--is-days", help="样本内窗口(交易日)"),
    oos_days: int | None = typer.Option(None, "--oos-days", help="样本外窗口(交易日)"),
    step: int | None = typer.Option(
        None, "--step", help="滚动步长(默认=oos-days,非重叠)"
    ),
    target: str | None = typer.Option(None, "--target", help="IS 优化目标指标"),
    min_is_sharpe: float | None = typer.Option(
        None, "--min-is-sharpe", help="IS 目标不达标则该窗口不部署(OOS 空仓)"
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="导出结果 JSON"),
    csv_dir: Path | None = typer.Option(None, "--csv-dir", help="本地 CSV 数据目录"),
) -> None:
    """Walk-Forward 分析:逐窗口 IS 独立选参 + OOS 评估,拼接样本外净值。"""
    cfg = load_config(config)
    if cfg.walk_forward is None:
        raise typer.BadParameter(
            "配置缺少 walk_forward 段,请先配置(参考 configs/walk.example.yaml)"
        )
    wf = cfg.walk_forward
    if is_days is not None:
        wf.is_days = is_days
    if oos_days is not None:
        wf.oos_days = oos_days
    if step is not None:
        wf.step = step
    if target is not None:
        wf.target = target
    if min_is_sharpe is not None:
        wf.min_is_sharpe = min_is_sharpe

    grid_dict: dict[str, list[Any]] | None = None
    if grid:
        try:
            grid_dict = json.loads(grid)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"grid JSON 解析失败: {e}") from e

    registry = default_registry(
        csv_dir=str(csv_dir) if csv_dir else None, cache=DataCache()
    )
    report = walk_forward(cfg, registry=registry, grid=grid_dict)

    typer.echo(
        f"\n=== Walk-Forward({len(report.windows)} 窗口,目标={report.target},"
        f"IS {wf.is_days} 日 / OOS {wf.oos_days} 日)==="
    )
    for w in report.windows:
        params = w.best_params or {}
        if w.deployed and w.oos_metrics:
            oos_m = w.oos_metrics
            typer.echo(
                f"{w.no:>2}. IS {w.is_start}~{w.is_end}  OOS {w.oos_start}~{w.oos_end}\n"
                f"     params={params}\n"
                f"     OOS sharpe={oos_m.get('sharpe', float('nan')):.3f} "
                f"sortino={oos_m.get('sortino', float('nan')):.3f} "
                f"calmar={oos_m.get('calmar', float('nan')):.3f} "
                f"ret={oos_m.get('total_return', 0.0):.2%} "
                f"mdd={oos_m.get('max_drawdown', 0.0):.2%} "
                f"trades={oos_m.get('n_trades', 0)}"
            )
        else:
            typer.echo(
                f"{w.no:>2}. IS {w.is_start}~{w.is_end}  OOS {w.oos_start}~{w.oos_end}"
                "  [未部署:IS 未达标]"
            )

    if report.metrics is not None:
        m = report.metrics
        typer.echo(f"\n=== 拼接样本外净值({len(report.equity_curve)} 个交易日)===")
        typer.echo(
            f"sharpe={m.sharpe:.3f} sortino={m.sortino:.3f} calmar={m.calmar:.3f} "
            f"annual_return={m.annual_return:.2%} max_drawdown={m.max_drawdown:.2%} "
            f"n_trades={m.n_trades}"
        )
    else:
        typer.echo("\n无可部署窗口(全部未达标或无样本外段)")

    if output:
        Path(output).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str)
        )
        typer.echo(f"\n结果已导出: {output}")


app.command(name="walk")(walk_command)


def cli_main() -> None:
    """入口(供 ``python -m djinn``)。"""
    app()


if __name__ == "__main__":
    cli_main()

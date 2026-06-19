"""djinn CLI:run / sweep / data(typer)。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from djinn.cli.data import data_app
from djinn.cli.run import run_command
from djinn.cli.sweep import sweep_command
from djinn.config import dump_config, load_config
from djinn.utils.logging import get_logger, set_log_level

app = typer.Typer(
    name="djinn",
    help="Djinn — 多市场量化回测框架(run / sweep / data)",
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


def cli_main() -> None:
    """入口(供 ``python -m djinn``)。"""
    app()


if __name__ == "__main__":
    cli_main()

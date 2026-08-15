"""净值 / 回撤 / 仓位曲线(matplotlib 静态图)。

供 HTML 报告内嵌与 CLI 导出。前端(ECharts)走 FastAPI 返回数据,不走本模块。
matplotlib 为可选依赖(``djinn[viz]``),延迟到函数内导入(E13)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from djinn.analytics.report import Report

try:
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - 无 viz 环境占位(函数内 _plt 会先抛友好错误)
    Figure = Any  # type: ignore


def _plt() -> Any:
    """延迟导入 matplotlib.pyplot(非交互后端);缺失给出友好报错。"""
    try:
        import matplotlib

        matplotlib.use("Agg")  # 非交互后端,服务端安全
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:  # pragma: no cover - 无 viz 环境
        raise ImportError("可视化需要安装 djinn[viz](pip install djinn[viz])") from e


def plot_equity_curve(report: Report, *, log_scale: bool = False) -> Figure:
    """净值曲线 + 基准叠加。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        report.equity_curve.index,
        np.asarray(report.equity_curve.values),
        label="策略",
        linewidth=1.5,
    )
    if report.benchmark_curve is not None:
        ax.plot(
            report.benchmark_curve.index,
            np.asarray(report.benchmark_curve.values),
            label="基准",
            linewidth=1.0,
            alpha=0.7,
        )
    ax.set_title("净值曲线")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale("log")
    fig.autofmt_xdate()
    fig.tight_layout()
    return cast(Figure, fig)


def plot_drawdown(report: Report) -> Figure:
    """水下回撤曲线(填充)。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 4))
    dd = report.drawdown_curve
    ax.fill_between(dd.index, np.asarray(dd.values), 0, color="red", alpha=0.4)
    ax.set_title(f"回撤曲线(最大回撤 {report.metrics.max_drawdown:.2%})")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return cast(Figure, fig)


def plot_positions(report: Report) -> Figure:
    """各标的仓位权重随时间堆叠面积图 + 现金占比。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 5))
    w = report.weights.copy()
    # 现金 = 1 - Σ 持仓权重
    cash = 1.0 - w.sum(axis=1)
    w["现金"] = cash.clip(lower=0.0)
    w.plot.area(ax=ax, stacked=True, alpha=0.85)
    ax.set_title("仓位权重变化")
    ax.set_xlabel("日期")
    ax.set_ylabel("权重")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=min(len(w.columns), 6))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return cast(Figure, fig)


def save_fig(fig: Figure, path: str | Path) -> Path:
    """保存图到文件(PNG, dpi=120)。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=120, bbox_inches="tight")
    _plt().close(fig)
    return p

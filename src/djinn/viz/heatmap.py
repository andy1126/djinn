"""月度 / 年度收益热力图(seaborn)。seaborn/matplotlib 为可选依赖(E13)。"""

from __future__ import annotations

from typing import Any, cast

from djinn.analytics.report import Report

try:
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - 无 viz 环境占位(函数内会先抛友好错误)
    Figure = Any  # type: ignore


def _sns_and_plt() -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        return sns, plt
    except ImportError as e:  # pragma: no cover - 无 viz 环境
        raise ImportError("可视化需要安装 djinn[viz](pip install djinn[viz])") from e


def plot_monthly_heatmap(report: Report) -> Figure:
    """月度收益热力图(行=年,列=月)。"""
    sns, plt = _sns_and_plt()
    mr = report.monthly_returns
    if mr.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "无月度收益数据", ha="center", va="center")
        ax.axis("off")
        return cast(Figure, fig)
    fig, ax = plt.subplots(figsize=(11, max(3, len(mr) * 0.6)))
    sns.heatmap(
        mr,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "月度收益"},
    )
    ax.set_title("月度收益热力图")
    ax.set_xlabel("月份")
    ax.set_ylabel("年份")
    fig.tight_layout()
    return cast(Figure, fig)

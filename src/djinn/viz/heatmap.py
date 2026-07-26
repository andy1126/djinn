"""月度 / 年度收益热力图(seaborn)。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from djinn.analytics.report import Report


def plot_monthly_heatmap(report: Report) -> Figure:
    """月度收益热力图(行=年,列=月)。"""
    mr = report.monthly_returns
    if mr.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "无月度收益数据", ha="center", va="center")
        ax.axis("off")
        return fig
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
    return fig

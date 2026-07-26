"""djinn.viz — 可视化:静态图(matplotlib/seaborn)+ HTML 报告(Jinja2+plotly)。"""

from __future__ import annotations

from djinn.viz.heatmap import plot_monthly_heatmap
from djinn.viz.html_report import render_html_report, save_html_report
from djinn.viz.plots import plot_drawdown, plot_equity_curve, plot_positions, save_fig

__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_positions",
    "save_fig",
    "plot_monthly_heatmap",
    "render_html_report",
    "save_html_report",
]

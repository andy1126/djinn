"""Jinja2 HTML 报告:汇总指标表 + 内嵌 plotly 交互图为单页 HTML。

plotly 为可选依赖(``djinn[viz]``),延迟到函数内导入(E13)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from djinn.analytics.report import Report
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _go() -> Any:
    """延迟导入 plotly.graph_objects;缺失给出友好报错。"""
    try:
        import plotly.graph_objects as go

        return go
    except ImportError as e:  # pragma: no cover - 无 viz 环境
        raise ImportError("HTML 报告需要安装 djinn[viz](pip install djinn[viz])") from e


def _equity_fig(report: Report) -> str:
    """净值 + 基准叠加 plotly 图(内嵌 HTML)。"""
    go = _go()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=report.equity_curve.index,
            y=report.equity_curve.values,
            mode="lines",
            name="策略",
            line=dict(width=2),
        )
    )
    if report.benchmark_curve is not None:
        fig.add_trace(
            go.Scatter(
                x=report.benchmark_curve.index,
                y=report.benchmark_curve.values,
                mode="lines",
                name="基准",
                line=dict(width=1, dash="dot"),
            )
        )
    fig.update_layout(
        title="净值曲线",
        xaxis_title="日期",
        yaxis_title="净值",
        template="plotly_white",
        height=420,
    )
    return str(fig.to_html(full_html=False, include_plotlyjs="cdn"))


def _drawdown_fig(report: Report) -> str:
    go = _go()
    dd = report.drawdown_curve
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            fill="tozeroy",
            name="回撤",
            line=dict(color="red", width=1),
        )
    )
    fig.update_layout(
        title=f"回撤曲线(最大回撤 {report.metrics.max_drawdown:.2%})",
        xaxis_title="日期",
        yaxis_title="回撤",
        template="plotly_white",
        height=320,
    )
    return str(fig.to_html(full_html=False, include_plotlyjs=False))


def _positions_fig(report: Report) -> str:
    go = _go()
    w = report.weights.copy()
    if w.empty:
        return "<p>无持仓数据</p>"
    cash = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    w["现金"] = cash
    fig = go.Figure()
    for col in w.columns:
        fig.add_trace(go.Scatter(x=w.index, y=w[col], stackgroup="pos", name=col))
    fig.update_layout(
        title="仓位权重",
        xaxis_title="日期",
        yaxis_title="权重",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=380,
    )
    return str(fig.to_html(full_html=False, include_plotlyjs=False))


def _monthly_heatmap_fig(report: Report) -> str:
    go = _go()
    mr = report.monthly_returns
    if mr.empty:
        return "<p>无月度收益数据</p>"
    fig = go.Figure(
        data=go.Heatmap(
            z=mr.values,
            x=list(mr.columns),
            y=[str(y) for y in mr.index],
            colorscale="RdYlGn",
            zmid=0,
            text=[
                [f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in mr.values
            ],
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="月度收益热力图", template="plotly_white", height=max(200, len(mr) * 40)
    )
    return str(fig.to_html(full_html=False, include_plotlyjs=False))


def _trades_df(report: Report) -> pd.DataFrame:
    rows = []
    for t in report.trades:
        rows.append(
            {
                "日期": getattr(t, "timestamp", ""),
                "标的": getattr(t, "symbol", ""),
                "方向": getattr(t, "side", ""),
                "数量": round(float(getattr(t, "qty", 0)), 4),
                "价格": round(float(getattr(t, "price", 0)), 4),
                "佣金": round(float(getattr(t, "commission", 0)), 2),
                "标签": getattr(t, "tag", ""),
            }
        )
    return pd.DataFrame(rows)


def render_html_report(report: Report) -> str:
    """渲染完整 HTML 报告(字符串)。"""
    m = report.metrics
    env = _env()
    tmpl = env.get_template("report.html")
    return tmpl.render(
        report=report,
        metrics=m.to_dict(),
        summary=report.summary(),
        symbols=report.symbols,
        equity_fig=_equity_fig(report),
        drawdown_fig=_drawdown_fig(report),
        positions_fig=_positions_fig(report),
        heatmap_fig=_monthly_heatmap_fig(report),
        trades_html=(
            _trades_df(report).to_html(index=False, classes="trades", border=0)
            if len(report.trades)
            else "<p>无成交</p>"
        ),
        rejections_count=len(report.rejections),
    )


def save_html_report(report: Report, path: str | Path) -> Path:
    """渲染并保存 HTML 报告。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html_report(report), encoding="utf-8")
    _log.info("HTML 报告已保存: %s", p)
    return p

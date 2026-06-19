"""结果导出:CSV / Excel(交易 / 持仓 / 指标)。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from djinn.analytics.report import Report


def trades_to_df(trades: list[Any]) -> pd.DataFrame:
    """成交列表 → DataFrame。"""
    if not trades:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "side", "qty", "price", "commission", "tag"]
        )
    rows = []
    for t in trades:
        rows.append(
            {
                "timestamp": getattr(t, "timestamp", None),
                "symbol": getattr(t, "symbol", ""),
                "side": getattr(t, "side", ""),
                "qty": getattr(t, "qty", 0.0),
                "price": getattr(t, "price", 0.0),
                "commission": getattr(t, "commission", 0.0),
                "tag": getattr(t, "tag", ""),
            }
        )
    return pd.DataFrame(rows)


def rejections_to_df(rejections: list[Any]) -> pd.DataFrame:
    if not rejections:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "side", "reason", "requested_qty", "tag"]
        )
    rows = []
    for r in rejections:
        rows.append(
            {
                "timestamp": getattr(r, "timestamp", None),
                "symbol": getattr(r, "symbol", ""),
                "side": getattr(r, "side", ""),
                "reason": getattr(r, "reason", ""),
                "requested_qty": getattr(r, "requested_qty", 0.0),
                "tag": getattr(r, "tag", ""),
            }
        )
    return pd.DataFrame(rows)


def export_csv(report: Report, out_dir: str | Path) -> list[Path]:
    """导出 CSV:指标、交易、拒单、净值、持仓。返回写入文件列表。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    # 指标
    metrics_df = pd.DataFrame([report.metrics.to_dict()])
    p = out / "metrics.csv"
    metrics_df.to_csv(p, index=False)
    files.append(p)
    # 交易
    p = out / "trades.csv"
    trades_to_df(report.trades).to_csv(p, index=False)
    files.append(p)
    # 拒单
    p = out / "rejections.csv"
    rejections_to_df(report.rejections).to_csv(p, index=False)
    files.append(p)
    # 净值
    p = out / "equity_curve.csv"
    report.equity_curve.to_csv(p, header=["equity"])
    files.append(p)
    # 持仓
    if len(report.positions):
        p = out / "positions.csv"
        report.positions.to_csv(p)
        files.append(p)
    # 权重
    if len(report.weights):
        p = out / "weights.csv"
        report.weights.to_csv(p)
        files.append(p)
    # 月度收益
    if len(report.monthly_returns):
        p = out / "monthly_returns.csv"
        report.monthly_returns.to_csv(p)
        files.append(p)
    return files


def export_excel(report: Report, out_path: str | Path) -> Path:
    """导出多 sheet Excel。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame([report.summary()]).to_excel(
            writer, sheet_name="summary", index=False
        )
        pd.DataFrame([report.metrics.to_dict()]).to_excel(
            writer, sheet_name="metrics", index=False
        )
        trades_to_df(report.trades).to_excel(writer, sheet_name="trades", index=False)
        rejections_to_df(report.rejections).to_excel(
            writer, sheet_name="rejections", index=False
        )
        report.equity_curve.to_frame("equity").to_excel(writer, sheet_name="equity")
        if report.benchmark_curve is not None:
            report.benchmark_curve.to_frame("benchmark").to_excel(
                writer, sheet_name="benchmark"
            )
        if len(report.positions):
            report.positions.to_excel(writer, sheet_name="positions")
        if len(report.weights):
            report.weights.to_excel(writer, sheet_name="weights")
        if len(report.monthly_returns):
            report.monthly_returns.to_excel(writer, sheet_name="monthly_returns")
    return out


def export(report: Report, out_dir: str | Path, fmt: str | Sequence[str]) -> list[Path]:
    """按格式导出。``fmt`` 支持 "csv" / "excel" 或其列表。"""
    fmts = [fmt] if isinstance(fmt, str) else list(fmt)
    out_dir_p = Path(out_dir)
    files: list[Path] = []
    if "csv" in fmts:
        files.extend(export_csv(report, out_dir_p / "csv"))
    if "excel" in fmts:
        files.append(export_excel(report, out_dir_p / "report.xlsx"))
    return files

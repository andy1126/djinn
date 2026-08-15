"""报告对象:把 BacktestResult + 指标 + 基准对比组装成单一数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from djinn.analytics.metrics import (
    Metrics,
    compute_max_drawdown,
    compute_metrics,
    monthly_returns,
    rolling_sharpe,
    rolling_volatility,
    yearly_returns,
)
from djinn.analytics.trades import (
    BenchmarkStats,
    TradeStats,
    compare_benchmark,
    compute_trade_stats,
)
from djinn.engine.event_engine import BacktestResult


@dataclass
class Report:
    """完整回测报告(供 CLI / 导出 / 可视化复用)。"""

    metrics: Metrics
    trade_stats: TradeStats
    benchmark_stats: BenchmarkStats | None = None
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_curve: pd.Series | None = None
    drawdown_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    monthly_returns: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    yearly_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rolling_sharpe: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rolling_volatility: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    trades: list[Any] = field(default_factory=list)
    rejections: list[Any] = field(default_factory=list)
    positions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    weights: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    prices: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame()
    )  # index=交易日, columns=symbol, value=收盘价(mark-to-market 口径)
    config: dict[str, Any] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    # ── 归因(Phase 5,可选;``run_backtest(with_attribution=True)`` 填充)──
    # 均为已序列化的 JSON 友好 dict(BrinsonResult.to_dict / FactorExposureReport.to_dict)。
    attribution: dict[str, Any] | None = None  # Brinson 行业归因(配置/选股/交互)
    factor_exposure: dict[str, Any] | None = None  # 因子暴露时序 + 行业权重分布

    def summary(self) -> dict[str, Any]:
        """一页式摘要(指标 + 关键统计)。"""
        out: dict[str, Any] = dict(self.metrics.to_dict())
        out.update(self.trade_stats.to_dict())
        if self.benchmark_stats is not None:
            out.update(self.benchmark_stats.to_dict())
        out["symbols"] = self.symbols
        return out


def build_report(
    result: BacktestResult,
    *,
    market: str | None = None,
    rf: float = 0.0,
    rolling_window: int = 63,
) -> Report:
    """从 BacktestResult 组装 Report。"""
    equity = result.equity_curve
    # 交易统计基于 round-trip 配对(胜率/盈亏比按回合,而非标的累计 realized_pnl)
    trade_stats = compute_trade_stats(result.trades)
    metrics = compute_metrics(
        equity,
        result.trades,
        rf=rf,
        market=market,
        trade_pnls=trade_stats.per_trade_pnl,
    )
    _, dd = compute_max_drawdown(equity)
    bench_stats: BenchmarkStats | None = None
    if result.benchmark_curve is not None:
        bench_stats = compare_benchmark(
            equity, result.benchmark_curve, market=market, rf=rf
        )
    return Report(
        metrics=metrics,
        trade_stats=trade_stats,
        benchmark_stats=bench_stats,
        equity_curve=equity,
        benchmark_curve=result.benchmark_curve,
        drawdown_curve=dd,
        monthly_returns=monthly_returns(equity),
        yearly_returns=yearly_returns(equity),
        rolling_sharpe=rolling_sharpe(equity, window=rolling_window, market=market),
        rolling_volatility=rolling_volatility(
            equity, window=rolling_window // 3 or 21, market=market
        ),
        trades=result.trades,
        rejections=result.rejections,
        positions=result.positions_curve,
        weights=result.weights_curve,
        prices=result.prices_curve,
        symbols=result.symbols,
        config={
            "initial_cash": float(result.config.initial_cash) if result.config else 0.0
        },
    )

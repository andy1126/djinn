"""交易统计与基准对比。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from djinn.utils.decimalmath import to_float


@dataclass
class TradeStats:
    """交易统计。"""

    n_trades: int = 0
    n_buys: int = 0
    n_sells: int = 0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0
    total_realized_pnl: float = 0.0
    per_trade_pnl: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_trades": self.n_trades,
            "n_buys": self.n_buys,
            "n_sells": self.n_sells,
            "win_rate": self.win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_holding_days": self.avg_holding_days,
            "total_realized_pnl": self.total_realized_pnl,
        }


def compute_trade_stats(
    fills: list[Any],
    positions: dict[str, object] | None = None,
    *,
    realized_pnls: dict[str, Decimal] | None = None,
) -> TradeStats:
    """从成交记录与持仓 realized_pnl 计算交易统计。

    Args:
        fills: engine.Fill 列表。
        realized_pnls: {symbol: Decimal} 各标的累计已实现盈亏(来自 Position.realized_pnl)。
    """
    n_buys = sum(1 for f in fills if getattr(f, "side", "") == "buy")
    n_sells = sum(1 for f in fills if getattr(f, "side", "") == "sell")
    pnls: list[float] = []
    total_realized = 0.0
    if realized_pnls:
        for _sym, pnl in realized_pnls.items():
            v = to_float(pnl)
            if v != 0.0:
                pnls.append(v)
                total_realized += v
    if not pnls:
        return TradeStats(n_trades=len(fills), n_buys=n_buys, n_sells=n_sells)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    pl_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0
    # 平均持仓周期(简化:总交易日 / 卖出次数)
    avg_hold = 0.0
    return TradeStats(
        n_trades=len(fills),
        n_buys=n_buys,
        n_sells=n_sells,
        win_rate=win_rate,
        profit_loss_ratio=pl_ratio,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_holding_days=avg_hold,
        total_realized_pnl=total_realized,
        per_trade_pnl=pnls,
    )


@dataclass
class BenchmarkStats:
    """基准对比指标。"""

    alpha: float = 0.0  # 超额收益(年化)
    beta: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    correlation: float = 0.0
    benchmark_return: float = 0.0
    strategy_return: float = 0.0
    excess_return: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio,
            "correlation": self.correlation,
            "benchmark_return": self.benchmark_return,
            "strategy_return": self.strategy_return,
            "excess_return": self.excess_return,
        }


def compare_benchmark(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    *,
    market: str | None = None,
    rf: float = 0.0,
) -> BenchmarkStats:
    """计算策略 vs 基准的对比指标。"""
    from djinn.utils.dates import trading_days_per_year

    af = trading_days_per_year(market or "DEFAULT")
    # 对齐索引
    idx = strategy_equity.index.intersection(benchmark_equity.index)
    s = strategy_equity.loc[idx].dropna()
    b = benchmark_equity.loc[idx].dropna()
    if len(s) < 2 or len(b) < 2:
        return BenchmarkStats()
    sr = s.pct_change().dropna()
    br = b.pct_change().dropna()
    common = sr.index.intersection(br.index)
    sr = sr.loc[common]
    br = br.loc[common]
    if len(sr) < 2:
        return BenchmarkStats()
    beta = (
        float(np.cov(sr, br)[0, 1] / np.var(br, ddof=1))
        if np.var(br, ddof=1) > 0
        else 0.0
    )
    corr = float(sr.corr(br))
    excess = sr - br
    te = float(excess.std(ddof=0) * np.sqrt(af))
    ir = (
        float(excess.mean() / excess.std(ddof=0) * np.sqrt(af))
        if excess.std(ddof=0) > 0
        else 0.0
    )
    strat_ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
    bench_ret = float(b.iloc[-1] / b.iloc[0] - 1.0)
    n_years = len(s) / af
    alpha = (
        float(((s.iloc[-1] / s.iloc[0]) / (b.iloc[-1] / b.iloc[0]) - 1) / n_years)
        if n_years > 0
        else 0.0
    )
    return BenchmarkStats(
        alpha=alpha,
        beta=beta,
        tracking_error=te,
        information_ratio=ir,
        correlation=corr,
        benchmark_return=bench_ret,
        strategy_return=strat_ret,
        excess_return=strat_ret - bench_ret,
    )

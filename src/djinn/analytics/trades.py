"""交易统计与基准对比。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from djinn.analytics.roundtrip import pair_round_trips


@dataclass
class TradeStats:
    """交易统计。"""

    n_trades: int = 0
    n_buys: int = 0
    n_sells: int = 0
    n_round_trips: int = 0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0
    total_realized_pnl: float = 0.0
    per_trade_pnl: list[float] = field(default_factory=list)  # 每回合 pnl

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_trades": self.n_trades,
            "n_buys": self.n_buys,
            "n_sells": self.n_sells,
            "n_round_trips": self.n_round_trips,
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
    """从成交记录重建 round-trip 回合,计算交易统计。

    胜率 / 盈亏比的数据源为 :func:`pair_round_trips` 的**每回合净盈亏**(含双边
    佣金摊销),而非各标的累计 ``realized_pnl``(后者把同标的多次买卖压成一笔,
    胜率实为"盈利标的占比")。

    Args:
        fills: engine.Fill 列表(统计层用 float 口径)。
        realized_pnls: 已弃用,保留兼容(不再读取;回合自给自足)。
    """
    n_buys = sum(1 for f in fills if getattr(f, "side", "") == "buy")
    n_sells = sum(1 for f in fills if getattr(f, "side", "") == "sell")
    rts = pair_round_trips(fills)
    pnls = [rt.pnl for rt in rts]
    total_realized = float(sum(pnls))
    if not rts:
        return TradeStats(n_trades=len(fills), n_buys=n_buys, n_sells=n_sells)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    pl_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0
    avg_hold = float(np.mean([rt.holding_days for rt in rts])) if rts else 0.0
    return TradeStats(
        n_trades=len(fills),
        n_buys=n_buys,
        n_sells=n_sells,
        n_round_trips=len(rts),
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

    alpha: float = 0.0  # Jensen alpha(年化,见 compare_benchmark)
    beta: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    correlation: float = 0.0
    benchmark_return: float = 0.0
    strategy_return: float = 0.0
    excess_return: float = 0.0
    downside_capture: float = 0.0  # 下行捕获(基准下跌期的策略/基准收益比)
    upside_capture: float = 0.0  # 上行捕获(基准上涨期的策略/基准收益比)

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
            "downside_capture": self.downside_capture,
            "upside_capture": self.upside_capture,
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
    if b.isna().all():  # 基准全 NaN(防御:理论上 bfill 后已消除前导 NaN)
        return BenchmarkStats()
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
    # B6:Jensen alpha —— α = (R_s − rf) − β(R_b − rf),年化
    rf_daily = rf / af
    alpha = float((sr.mean() - rf_daily) - beta * (br.mean() - rf_daily)) * af
    down_mask = br < 0
    down_capture = (
        float(sr[down_mask].sum() / br[down_mask].sum())
        if down_mask.any() and br[down_mask].sum() != 0
        else 0.0
    )
    up_mask = br > 0
    up_capture = (
        float(sr[up_mask].sum() / br[up_mask].sum())
        if up_mask.any() and br[up_mask].sum() != 0
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
        downside_capture=down_capture,
        upside_capture=up_capture,
    )

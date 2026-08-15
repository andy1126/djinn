"""绩效指标:累计 / 年化收益 / 波动 / 夏普 / 索提诺 / 最大回撤 / Calmar / 胜率 / 换手。

净值曲线与收益率用 ``float64``;年化按市场交易日数(A股≈242,美股≈252)。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from djinn.utils.dates import trading_days_per_year


@dataclass
class Metrics:
    """回测绩效指标集合。"""

    total_return: float = 0.0  # 累计收益率
    annual_return: float = 0.0  # 年化收益率
    annual_volatility: float = 0.0  # 年化波动率
    sharpe: float = 0.0  # 夏普比率
    sortino: float = 0.0  # 索提诺比率
    max_drawdown: float = 0.0  # 最大回撤
    calmar: float = 0.0  # Calmar
    win_rate: float = 0.0  # 胜率(按交易)
    profit_loss_ratio: float = 0.0  # 盈亏比
    turnover: float = 0.0  # 换手率(双边、区间合计、未年化)
    turnover_annual: float = 0.0  # 单边年化换手(B8)
    n_trades: int = 0
    n_days: int = 0
    cagr: float = 0.0
    volatility: float = 0.0  # 日波动率
    var_95: float = 0.0  # 95% VaR(日损失,正)
    cvar_95: float = 0.0  # 95% CVaR(日期望损失,正)
    max_drawdown_duration: int = 0  # 最长水下期(天)
    max_losing_streak: int = 0  # 最长连续亏损(天)
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int]:
        d: dict[str, float | int] = {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "win_rate": self.win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "turnover": self.turnover,
            "turnover_annual": self.turnover_annual,
            "n_trades": self.n_trades,
            "n_days": self.n_days,
            "cagr": self.cagr,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "max_drawdown_duration": self.max_drawdown_duration,
            "max_losing_streak": self.max_losing_streak,
        }
        d.update(self.extra)
        return d


def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def _annual_factor(market: str | None) -> int:
    return trading_days_per_year(market or "DEFAULT")


def compute_max_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
    """最大回撤(峰值法)。返回 (mdd, drawdown_series)。"""
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    mdd = float(dd.min())
    return mdd, dd


def _max_underwater_days(dd: pd.Series) -> int:
    """最长连续回撤 < 0 的天数(水下期)。"""
    underwater = (dd < 0).to_numpy()
    if not underwater.any():
        return 0
    best = cur = 0
    for v in underwater:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _max_losing_streak(rets: pd.Series) -> int:
    """最长连续负收益天数。"""
    neg = (rets < 0).to_numpy()
    if not neg.any():
        return 0
    best = cur = 0
    for v in neg:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def compute_metrics(
    equity: pd.Series,
    trades: Iterable[Any] | None = None,
    *,
    rf: float = 0.0,
    market: str | None = None,
    trade_pnls: Iterable[float] | None = None,
) -> Metrics:
    """从净值曲线计算核心指标。

    Args:
        equity: 净值曲线(index=交易日)。
        trades: 成交列表(用于换手率;若提供 trade_pnls 则胜率用之)。
        rf: 无风险利率(年化,小数,如 0.02)。
        market: 市场代码(CN/HK/US),用于年化因子。
        trade_pnls: 每笔已实现盈亏列表(用于胜率/盈亏比);若 None 则从 trades 推导。
    """
    # 物化 trades(fills 列表):Iterable 可能已被消费,后续多处使用。
    trades = list(trades or [])
    if len(equity) < 2:
        return Metrics(n_days=len(equity), n_trades=len(trades))
    af = _annual_factor(market)
    rets = _daily_returns(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_years = len(equity) / af
    cagr = (
        float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1.0)
        if n_years > 0
        else 0.0
    )
    ann_return = cagr
    daily_vol = float(rets.std(ddof=0))
    ann_vol = daily_vol * np.sqrt(af)
    rf_daily = rf / af
    excess = rets - rf_daily
    sharpe = float(excess.mean() / daily_vol * np.sqrt(af)) if daily_vol > 0 else 0.0
    # B3:索提诺标准口径 —— 下行阈值用 MAR(rf 日化),下行偏差用全样本下半方差
    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(af))
    sortino = float(excess.mean() * af / downside_dev) if downside_dev > 0 else 0.0
    mdd, dd = compute_max_drawdown(equity)
    # B4:零回撤 → Calmar 未定义(NaN),而非 0(避免 sweep 排序把最优组合排最后)
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else float("nan")

    # 尾部风险 / 回撤时长 / 连亏(B8:VaR/CVaR 非负,历史法日度)
    var_95 = max(0.0, -float(rets.quantile(0.05)))
    tail = rets[rets <= rets.quantile(0.05)]
    cvar_95 = max(0.0, -float(tail.mean())) if len(tail) > 0 else 0.0
    max_dd_duration = _max_underwater_days(dd)
    max_losing = _max_losing_streak(rets)

    # 胜率 / 盈亏比
    pnls = list(trade_pnls) if trade_pnls is not None else _pnls_from_trades(trades)
    win_rate, pl_ratio = _win_stats(pnls)

    # 换手率 = 期间成交额 / 平均净值
    turnover = _turnover(equity, trades)
    # B8:单边年化换手 = 双边区间合计 × af / n_days / 2
    turnover_annual = turnover * af / len(equity) / 2.0 if len(equity) > 1 else 0.0

    return Metrics(
        total_return=total_return,
        annual_return=ann_return,
        annual_volatility=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        win_rate=win_rate,
        profit_loss_ratio=pl_ratio,
        turnover=turnover,
        turnover_annual=turnover_annual,
        n_trades=len(trades),
        n_days=len(equity),
        cagr=cagr,
        volatility=daily_vol,
        var_95=var_95,
        cvar_95=cvar_95,
        max_drawdown_duration=max_dd_duration,
        max_losing_streak=max_losing,
    )


def _pnls_from_trades(trades: Iterable[Any]) -> list[float]:
    """从成交流推导每笔(卖出)已实现盈亏。

    简化:取 sell 成交,用成交价 - 持仓均价(若 Fill 携带);否则无法推导,返回空。
    本框架的 engine.Fill 不携带 avg_cost,故胜率应在更高层用 Account.position.realized_pnl。
    这里返回空,留 trade_pnls 参数显式传入。
    """
    return []


def _win_stats(pnls: list[float]) -> tuple[float, float]:
    if not pnls:
        return 0.0, 0.0
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls)
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    pl_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0
    return win_rate, pl_ratio


def _turnover(equity: pd.Series, trades: Iterable[Any]) -> float:
    """换手率 = 总成交额 / 平均净值。"""
    trades = list(trades)
    if not trades:
        return 0.0
    total_amount = 0.0
    for t in trades:
        qty = getattr(t, "qty", 0) or 0
        price = getattr(t, "price", 0) or 0
        total_amount += abs(float(qty) * float(price))
    avg_equity = float(equity.mean()) if len(equity) else 0.0
    return total_amount / avg_equity if avg_equity > 0 else 0.0


def rolling_sharpe(
    equity: pd.Series, window: int = 63, market: str | None = None
) -> pd.Series:
    """滚动夏普。"""
    af = _annual_factor(market)
    rets = _daily_returns(equity)
    mean = rets.rolling(window).mean()
    std = rets.rolling(window).std(ddof=0)
    return pd.Series((mean / std * np.sqrt(af)).dropna())


def rolling_volatility(
    equity: pd.Series, window: int = 21, market: str | None = None
) -> pd.Series:
    af = _annual_factor(market)
    rets = _daily_returns(equity)
    return pd.Series((rets.rolling(window).std(ddof=0) * np.sqrt(af)).dropna())


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """月度收益矩阵(行=年,列=月),供热力图。"""
    monthly = equity.resample("ME").last().ffill()
    mret = monthly.pct_change()
    # B5:补首月收益(首月末/期初 − 1),否则 pct_change 首行为 NaN 被 dropna 丢掉
    if len(mret) > 0:
        mret.iloc[0] = float(monthly.iloc[0] / equity.iloc[0] - 1.0)
    mret = mret.dropna()
    idx = pd.DatetimeIndex(mret.index)
    df = pd.DataFrame({"year": idx.year, "month": idx.month, "ret": mret.values})
    pivot = df.pivot_table(index="year", columns="month", values="ret")
    pivot.columns = [f"{m:02d}月" for m in pivot.columns]
    return pivot


def yearly_returns(equity: pd.Series) -> pd.Series:
    """年度收益。"""
    yearly = equity.resample("YE").last().ffill()
    return yearly.pct_change().dropna()

"""组合型 / 状态型策略单测:横截面动量、双动量、海龟、网格、配对。

用一个极简 fake ctx 逐 bar 驱动 ``on_bar``,校验调仓订单(不跑完整引擎)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from djinn.strategy import (
    BuyAndHold,
    CrossSectionalMomentum,
    DualMomentum,
    Grid,
    PairsSpread,
    TurtleATR,
    VolTarget,
    get_strategy_class,
    param_schema,
)


class _FakePos:
    def __init__(self, qty: float = 1.0) -> None:
        self.qty = qty


class _FakePortfolio:
    def __init__(self, positions: dict[str, _FakePos] | None = None) -> None:
        self.positions = positions or {}
        self.equity = 100000.0


class _FakeData:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames
        self.symbols = list(frames)

    def __getitem__(self, s: str) -> pd.DataFrame:
        return self._frames[s]

    def __contains__(self, s: object) -> bool:
        return s in self._frames


class _FakeCtx:
    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        positions: dict[str, _FakePos] | None = None,
    ) -> None:
        self.data = _FakeData(frames)
        self.portfolio = _FakePortfolio(positions)
        self.orders: list[tuple[str, float]] = []
        self.now = pd.Timestamp("2024-01-02")

    def order_target_percent(self, symbol: str, pct: float) -> None:
        self.orders.append((symbol, pct))


def _frame(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=len(closes))
    df = pd.DataFrame({"close": closes}, index=idx)
    return df.assign(open=df.close, high=df.close + 1, low=df.close - 1, volume=1000)


def _run_bars(strategy, frames: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
    """逐 bar 调用 on_bar(每次喂到当前 bar 的切片),汇总全部调仓订单。"""
    n = min(len(f) for f in frames.values())
    orders: list[tuple[str, float]] = []
    for i in range(1, n + 1):
        sliced = {s: f.iloc[:i] for s, f in frames.items()}
        ctx = _FakeCtx(sliced)
        strategy.on_bar(ctx)
        orders.extend(ctx.orders)
    return orders


def test_new_strategies_in_registry():
    for name, cls in (
        ("BuyAndHold", BuyAndHold),
        ("CrossSectionalMomentum", CrossSectionalMomentum),
        ("DualMomentum", DualMomentum),
        ("TurtleATR", TurtleATR),
        ("Grid", Grid),
        ("PairsSpread", PairsSpread),
        ("VolTarget", VolTarget),
    ):
        assert get_strategy_class(name) is cls


def test_param_schemas():
    assert {p.name for p in param_schema(CrossSectionalMomentum)} == {
        "lookback",
        "n_stocks",
        "rebalance_freq",
    }
    assert {p.name for p in param_schema(TurtleATR)} == {
        "entry",
        "exit_",
        "atr_period",
        "risk_per_unit",
    }
    assert "symbol_a" in {p.name for p in param_schema(PairsSpread)}


def test_buy_and_hold_rebalances_equal_weight():
    frames = {"A": _frame(np.full(30, 100.0)), "B": _frame(np.full(30, 100.0))}
    orders = _run_bars(BuyAndHold(rebalance_freq=10), frames)
    assert ("A", 0.5) in orders
    assert ("B", 0.5) in orders


def test_cross_sectional_momentum_selects_top():
    a = np.linspace(100, 130, 20)  # 上涨
    b = np.full(20, 100.0)  # 平
    c = np.linspace(100, 70, 20)  # 下跌
    ctx = _FakeCtx({"A": _frame(a), "B": _frame(b), "C": _frame(c)})
    s = CrossSectionalMomentum(lookback=5, n_stocks=2, rebalance_freq=1)
    s.on_bar(ctx)
    targets = {sym for sym, pct in ctx.orders if pct > 0}
    assert targets == {"A", "B"}  # A 涨、B 平 > C 跌


def test_dual_momentum_cash_when_all_down():
    a = np.linspace(100, 70, 20)
    b = np.linspace(100, 80, 20)
    ctx = _FakeCtx({"A": _frame(a), "B": _frame(b)})
    s = DualMomentum(lookback=5, rebalance_freq=1)
    s.on_bar(ctx)
    assert ctx.orders == []  # 全部负收益 → 空仓


def test_turtle_atr_breakout_enters_long():
    close = np.concatenate([np.full(20, 100.0), [110.0]])
    orders = _run_bars(
        TurtleATR(entry=5, exit_=5, atr_period=5, risk_per_unit=0.01),
        {"A": _frame(close)},
    )
    assert any(sym == "A" and 0 < pct <= 1 for sym, pct in orders)


def test_turtle_atr_position_sizing():
    """TurtleATR 仓位 = risk_per_unit×price/ATR(经典海龟,旧式多除 atr_period)。"""
    from djinn.indicators import atr as atr_ind

    close = np.concatenate([np.full(20, 100.0), [110.0]])
    orders = _run_bars(
        TurtleATR(entry=5, exit_=5, atr_period=5, risk_per_unit=0.01),
        {"A": _frame(close)},
    )
    buys = [pct for sym, pct in orders if sym == "A" and pct > 0]
    assert buys
    df = _frame(close)
    a = float(atr_ind(df["high"], df["low"], df["close"], 5).iloc[-1])
    expected = min(1.0, 0.01 * 110.0 / a)  # 经典:不再除以 atr_period
    assert buys[0] == pytest.approx(expected, rel=0.1)
    # 旧式(多除 5)会得到 5 倍小的仓位,与经典公式显著不同
    assert buys[0] > 0.01 * 110.0 / (5 * a) * 2


def test_grid_accumulates_on_dip():
    close = np.concatenate([np.full(5, 100.0), [95.0, 90.0, 90.0]])
    orders = _run_bars(
        Grid(step=0.05, num_levels=5, unit_weight=0.1), {"A": _frame(close)}
    )
    # 首根基准 100,末价 90 → 下跌 10% → 2 档 → 目标权重 0.2
    assert ("A", 0.2) in orders


def test_vol_target_equal_weight_when_no_vol():
    frames = {"A": _frame(np.full(30, 100.0)), "B": _frame(np.full(30, 100.0))}
    orders = _run_bars(VolTarget(rebalance_freq=10), frames)
    # 净值恒定 → 已实现波动率为 0 → 满仓等权
    assert ("A", 0.5) in orders
    assert ("B", 0.5) in orders


def test_pairs_spread_orders_one_side():
    a = np.full(40, 100.0)
    b = np.concatenate([np.full(20, 100.0), np.linspace(100, 50, 20)])
    orders = _run_bars(
        PairsSpread(symbol_a="A", symbol_b="B", lookback=10, entry_z=1.0, exit_z=0.5),
        {"A": _frame(a), "B": _frame(b)},
    )
    # 价差走阔,至少产生一次做多便宜侧(B)的订单
    assert any(sym == "B" and pct > 0 for sym, pct in orders)

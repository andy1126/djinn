"""选股引擎测试:条件筛选 / 多因子打分 / 动态股票池 / union 日历回测。

不依赖网络:行情用合成 MarketData,基本面用合成 PIT 面板。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from djinn.data.market_data import MarketData
from djinn.data.schema import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_VOLUME,
    Market,
)
from djinn.engine import EngineConfig, EventDrivenEngine
from djinn.factor.library.momentum import MomentumFactor
from djinn.factor.library.quality import ROEFactor
from djinn.factor.library.value import EPFactor
from djinn.screen import (
    DynamicUniverse,
    FactorScore,
    ScreenCondition,
    Screener,
    score_cross_section,
    score_universe,
    top_n,
)
from djinn.strategy.base import SCOPE_PORTFOLIO, Context, Strategy
from djinn.strategy.library.factor_portfolio import FactorPortfolioStrategy


def _md(symbol: str, closes: dict[pd.Timestamp, float]) -> MarketData:
    """由 {日期: 收盘价} 构造合成美股 MarketData(无 T+1 / 涨跌停,lot=1)。"""
    keys = sorted(closes)
    df = pd.DataFrame(
        {
            COL_OPEN: [closes[k] for k in keys],
            COL_HIGH: [closes[k] for k in keys],
            COL_LOW: [closes[k] for k in keys],
            COL_CLOSE: [closes[k] for k in keys],
            COL_VOLUME: [10000.0] * len(keys),
        },
        index=pd.DatetimeIndex(keys),
    )
    return MarketData(symbol=symbol, market=Market.US, df=df)


def _buyhold(symbols_weights: dict[str, float]) -> Strategy:
    """首bar 一次性买入目标权重并持有。"""

    class _BuyHold(Strategy):
        scope = SCOPE_PORTFOLIO

        def __init__(self) -> None:
            super().__init__()
            self._done = False

        def on_bar(self, ctx: Context) -> None:
            if not self._done:
                for s, w in symbols_weights.items():
                    ctx.order_target_percent(s, w)
                self._done = True

    return _BuyHold()


# ── 条件筛选 ─────────────────────────────────────────────
def test_screener_conditions() -> None:
    df = pd.DataFrame(
        {
            "pe": [8.0, 15.0, 25.0, 40.0],
            "roe": [0.20, 0.12, 0.08, 0.03],
            "industry": ["tech", "fin", "tech", "fin"],
        },
        index=["A", "B", "C", "D"],
    )
    assert Screener.apply([ScreenCondition(field="pe", op="lt", value=20)], df) == [
        "A",
        "B",
    ]
    assert Screener.apply(
        [ScreenCondition(field="roe", op="between", value=[0.05, 0.15])], df
    ) == ["B", "C"]
    assert Screener.apply(
        [ScreenCondition(field="industry", op="in", value=["fin"])], df
    ) == ["B", "D"]
    conds = [
        ScreenCondition(field="pe", op="lt", value=20),
        ScreenCondition(field="roe", op="ge", value=0.10),
    ]
    assert Screener.apply(conds, df) == ["A", "B"]
    # 缺字段 → 全不通过
    assert Screener.apply([ScreenCondition(field="pb", op="lt", value=5)], df) == []


def test_screener_nan_excluded() -> None:
    df = pd.DataFrame({"pe": [5.0, float("nan"), 10.0]}, index=["A", "B", "C"])
    assert Screener.apply([ScreenCondition(field="pe", op="lt", value=20)], df) == [
        "A",
        "C",
    ]


def test_screener_between_validation() -> None:
    with pytest.raises(ValueError):
        ScreenCondition(field="pe", op="between", value=[10, 5])  # 下界>上界
    with pytest.raises(ValueError):
        ScreenCondition(field="pe", op="between", value=[10])  # 非两元素


# ── 打分 ─────────────────────────────────────────────────
def test_score_cross_section_ordering() -> None:
    cross = pd.DataFrame(
        {"ep": [0.10, 0.05, 0.02], "vol": [0.20, 0.30, 0.40]},
        index=["A", "B", "C"],
    )
    scores = [
        FactorScore(factor="ep", weight=1.0),  # 越高越好
        FactorScore(factor="vol", weight=-1.0),  # 负权重 = 越低越好
    ]
    s = score_cross_section(cross, scores, preprocess=True)
    assert s.idxmax() == "A"
    assert s["A"] > s["B"] > s["C"]


def test_score_universe_and_top_n() -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    panel = {
        "ep": pd.DataFrame(
            [[0.10, 0.05, 0.02]] * 3, index=dates, columns=["A", "B", "C"]
        )
    }
    sdf = score_universe(panel, [FactorScore(factor="ep", weight=1.0)])
    assert set(sdf.columns) == {"A", "B", "C"}
    assert top_n(sdf, dates[0], 2) == ["A", "B"]
    assert top_n(sdf, dates[1], 1) == ["A"]
    # 无数据日期 → 取之前最近一日
    assert top_n(sdf, date(2023, 12, 30), 2) == []


# ── 动态股票池 ────────────────────────────────────────────
def test_dynamic_universe_membership_change() -> None:
    sdf = pd.DataFrame(
        {"A": [3.0, 1.0], "B": [2.0, 2.0], "C": [1.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    uni = DynamicUniverse.from_score_history(sdf, 2)
    d0, d1 = sdf.index[0].date(), sdf.index[1].date()
    assert set(uni.symbols_on(d0)) == {"A", "B"}  # 3,2
    assert set(uni.symbols_on(d1)) == {"C", "B"}  # 3,2(A 落选、C 新进)
    assert set(uni.all_symbols) == {"A", "B", "C"}


# ── union 日历回测 ────────────────────────────────────────
def test_union_calendar_forward_fill_valuation() -> None:
    """持仓标的缺当日行情时,以前向填充价估值(市值不归零)。"""
    days = pd.date_range("2024-01-01", periods=10)
    a = dict.fromkeys(days, 100.0)
    b_days = list(days[:5]) + list(days[7:])  # B 缺 day5/6(停牌)
    b = dict.fromkeys(b_days, 50.0)
    data = {"A": _md("A", a), "B": _md("B", b)}
    strat = _buyhold({"A": 0.4, "B": 0.4})
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    # B 在缺行情日仍被估值 → 权重 > 0
    assert res.weights_curve["B"].iloc[5] > 0
    assert res.weights_curve["B"].iloc[6] > 0
    # 权益 > 现金(持仓被正确估值,未归零)
    assert res.equity_curve.iloc[5] > res.cash_curve.iloc[5]
    # 缺行情日持仓数量不变
    assert res.positions_curve["B"].iloc[5] == res.positions_curve["B"].iloc[4]


def test_union_calendar_pre_ipo_no_position() -> None:
    """union 日历下,未上市票在上市前无持仓(挂单等行情)。"""
    days = pd.date_range("2024-01-01", periods=10)
    a = dict.fromkeys(days, 100.0)
    new = dict.fromkeys(list(days[5:]), 20.0)  # NEW 第 5 日才上市
    data = {"A": _md("A", a), "NEW": _md("NEW", new)}
    strat = _buyhold({"A": 0.4, "NEW": 0.4})
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    # 上市前(day0-4)NEW 无持仓
    assert (res.positions_curve["NEW"].iloc[:5] == 0).all()
    # 上市首日 bar 出现后挂单成交 → 有持仓
    assert res.positions_curve["NEW"].iloc[5] > 0


def test_factor_portfolio_dynamic_universe_no_crash() -> None:
    """成分中途增删(中期 IPO)下,因子组合回测不崩溃且资金守恒。"""
    days = pd.date_range("2024-01-01", periods=30)
    data = {
        "A": _md("A", {d: 100.0 + i for i, d in enumerate(days)}),
        "B": _md("B", {d: 50.0 + 0.5 * i for i, d in enumerate(days)}),
        "NEW": _md("NEW", {d: 20.0 + i for i, d in enumerate(list(days[15:]))}),
    }
    strat = FactorPortfolioStrategy(
        factors=[MomentumFactor(period=5)],
        scores=[FactorScore(factor="momentum", weight=1.0)],
        n_stocks=2,
        rebalance_freq=5,
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    assert len(res.equity_curve) == 30
    # NEW 上市前无持仓
    assert (res.positions_curve["NEW"].iloc[:15] == 0).all()
    # 资金守恒(末日)
    final_prices = {s: float(md.df[COL_CLOSE].iloc[-1]) for s, md in data.items()}
    res.account.check_invariant(final_prices)


# ── 端到端:固定池 EP+ROE 双因子 TopN ─────────────────────
def test_end_to_end_ep_roe_top_n() -> None:
    days = pd.date_range("2024-01-01", periods=40)
    symbols = ["S0", "S1", "S2", "S3", "S4", "S5"]
    rng = np.random.default_rng(0)
    data: dict[str, MarketData] = {}
    for s in symbols:
        rets = rng.normal(0.0005, 0.02, len(days))
        close = 100 * np.exp(np.cumsum(rets))
        data[s] = _md(s, {d: float(c) for d, c in zip(days, close, strict=True)})
    # 基本面 PIT 面板(常数):S0 最优(最低 PE + 最高 ROE),S5 最差
    pe = {"S0": 5.0, "S1": 10.0, "S2": 15.0, "S3": 20.0, "S4": 25.0, "S5": 30.0}
    roe = {"S0": 0.30, "S1": 0.25, "S2": 0.20, "S3": 0.15, "S4": 0.10, "S5": 0.05}
    pe_panel = pd.DataFrame({s: [v] * len(days) for s, v in pe.items()}, index=days)
    roe_panel = pd.DataFrame({s: [v] * len(days) for s, v in roe.items()}, index=days)
    strat = FactorPortfolioStrategy(
        factors=[EPFactor(), ROEFactor()],
        scores=[
            FactorScore(factor="ep", weight=1.0),
            FactorScore(factor="roe", weight=1.0),
        ],
        n_stocks=3,
        rebalance_freq=10,
        fundamentals={"pe": pe_panel, "roe": roe_panel},
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    assert len(res.equity_curve) == len(days)
    assert res.n_trades > 0
    # 最优 S0 被选中持有;持股数 ≤ TopN
    held = [s for s in symbols if res.positions_curve[s].iloc[-1] > 0]
    assert "S0" in held
    assert len(held) <= 3
    final_prices = {s: float(data[s].df[COL_CLOSE].iloc[-1]) for s in symbols}
    res.account.check_invariant(final_prices)

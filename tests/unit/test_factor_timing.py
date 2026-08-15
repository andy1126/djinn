"""G 计划:择时规则库 + 选股流水线增强测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from djinn.data.schema import COL_AMOUNT
from djinn.strategy.library.factor_portfolio import FactorPortfolioStrategy
from djinn.strategy.timing import (
    AboveSMAConfirm,
    ATRTrailingExit,
    MarketRegimeFilter,
    SMABreakExit,
)


def _bare_strat(**attrs) -> FactorPortfolioStrategy:
    """绕过 __init__(免 factors/scores 校验)构造裸实例,仅测纯方法。"""
    s = object.__new__(FactorPortfolioStrategy)
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def _md(symbol: str, closes: dict[pd.Timestamp, float]):
    """合成美股 MarketData(仅收盘价;价差/量用平值)。"""
    from djinn.data.market_data import MarketData
    from djinn.data.schema import Market

    keys = sorted(closes)
    df = pd.DataFrame(
        {
            "open": [closes[k] for k in keys],
            "high": [closes[k] + 1.0 for k in keys],
            "low": [closes[k] - 1.0 for k in keys],
            "close": [closes[k] for k in keys],
            "volume": [10000.0] * len(keys),
        },
        index=pd.DatetimeIndex(keys),
    )
    return MarketData(symbol=symbol, market=Market.US, df=df)


class _RankFactor:
    """末行截面按列序给分(首列最高);其余行为 0。用于 G0/G7 测试。"""

    name = "rank"
    max_lookback = 1

    def compute(self, prices, ohlcv, fundamentals):
        out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n = len(prices.columns)
        out.iloc[-1] = [float(n - j) for j in range(n)]
        return out


# ── G5:择时规则库 ─────────────────────────────────────


def test_regime_filter() -> None:
    f = MarketRegimeFilter(window=5, floor=0.3)
    for c in [10.0, 11.0, 12.0]:
        f.update(c)
        assert f.exposure_cap() == 1.0  # 暖机期放行
    f.update(13.0)
    f.update(14.0)
    assert f.exposure_cap() == 1.0  # 站上 SMA
    f.update(1.0)
    assert f.exposure_cap() == 0.3  # 跌破 → floor


def test_regime_filter_large_window() -> None:
    """MarketRegimeFilter window>210 时闸门仍生效(旧 deque maxlen=210 恒放行)。"""
    f = MarketRegimeFilter(window=250, floor=0.3)
    for _ in range(300):
        f.update(100.0)  # 低于 SMA250 → 闸门应关
    assert f.exposure_cap() == 0.3
    f2 = MarketRegimeFilter(window=250, floor=0.3)
    for i in range(300):
        f2.update(100.0 + i * 0.1)  # 上行站上 SMA → 放行
    assert f2.exposure_cap() == 1.0


def test_sma_break_exit() -> None:
    e = SMABreakExit(window=3)
    for c in [10.0, 10.0, 10.0]:
        e.update("S", 10, 10, 10, c)
    assert e.should_exit("S") is False  # 均线持平
    e.update("S", 10, 10, 10, 5.0)
    assert e.should_exit("S") is True  # 跌破


def test_atr_trailing() -> None:
    e = ATRTrailingExit(mult=3.0, window=2)
    assert e.should_exit("S") is False  # 未 arm
    e.arm("S", 100.0)
    assert e._peak["S"] == 100.0
    e.update("S", 100, 105, 95, 100)  # 峰值只升不降
    assert e._peak["S"] == 105.0
    e.disarm("S")
    assert e.should_exit("S") is False  # disarm 后不再判定


def test_above_sma_confirm() -> None:
    c = AboveSMAConfirm(window=3)
    assert c.entry_ok(pd.Series([10.0, 10.0, 10.0])) is False  # 持平不站上
    assert c.entry_ok(pd.Series([10.0, 10.0, 11.0])) is True
    assert c.entry_ok(pd.Series([10.0])) is True  # 数据不足不拦截


# ── G2/G3/G4:选股流水线 ───────────────────────────────


def test_pick_neutral_basic() -> None:
    score = pd.Series(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2],
        index=[f"S{i}" for i in range(12)],
    )
    industry = {f"S{i}": ("A" if i < 4 else "B" if i < 8 else "C") for i in range(12)}
    picked = FactorPortfolioStrategy._pick_neutral(score, industry, 6)
    assert len(picked) == 6
    by_ind: dict[str, list[str]] = {}
    for s in picked:
        by_ind.setdefault(industry[s], []).append(s)
    assert all(len(v) <= 2 for v in by_ind.values())  # k = ceil(6/3) = 2


def test_sector_cap_scales() -> None:
    strat = _bare_strat(
        max_sector_weight=0.3,
        industry_map={"A": "X", "B": "X", "C": "Y"},
    )
    out = strat._apply_sector_cap({"A": 0.2, "B": 0.2, "C": 0.2})
    assert out["A"] == pytest.approx(0.15)
    assert out["B"] == pytest.approx(0.15)
    assert out["C"] == pytest.approx(0.2)


# ── G1:G2:G4 纯方法测试 ────────────────────────────────


def _tradable_panels(
    n_days: int = 30,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """构造 prices + ohlcv(amount) 面板供 _tradable 测试。"""
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    cols = ["A", "B"]
    prices = pd.DataFrame(100.0, index=idx, columns=cols)
    amount = pd.DataFrame({"A": [1.0e8] * n_days, "B": [1.0e6] * n_days}, index=idx)
    return prices, {COL_AMOUNT: amount}


def test_tradable_amount() -> None:
    """G1:低成交额标的(近 20 日均值 < min_amount)被滤掉。"""
    prices, ohlcv = _tradable_panels()
    strat = _bare_strat(
        min_amount=5.0e7, min_list_days=None, exclude_st=False, names=None
    )
    out = strat._tradable(None, ["A", "B"], prices, ohlcv)  # type: ignore[arg-type]
    assert out == ["A"]  # A 成交额 1e8 保留,B 1e6 过滤


def test_tradable_list_days() -> None:
    """G1:数据行数不足 min_list_days(次新近似)→ 被滤。"""
    idx = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    # B 只有 10 天数据(其余 NaN)
    prices.loc[idx[10:], "B"] = float("nan")
    strat = _bare_strat(min_amount=None, min_list_days=20, exclude_st=False, names=None)
    out = strat._tradable(None, ["A", "B"], prices, {})  # type: ignore[arg-type]
    assert out == ["A"]


def test_tradable_st() -> None:
    """G1:exclude_st + names 含 ST → 被滤。"""
    prices, ohlcv = _tradable_panels()
    strat = _bare_strat(
        min_amount=None, min_list_days=None, exclude_st=True, names={"B": "ST 某公司"}
    )
    out = strat._tradable(None, ["A", "B"], prices, ohlcv)  # type: ignore[arg-type]
    assert out == ["A"]


def test_tradable_suspended() -> None:
    """G1:union 日历下当日无行情(末值 < 末交易日)→ 被滤。"""
    idx = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    prices.loc[idx[-1], "B"] = float("nan")  # B 当日停牌
    strat = _bare_strat(
        min_amount=None, min_list_days=None, exclude_st=False, names=None
    )
    out = strat._tradable(None, ["A", "B"], prices, {})  # type: ignore[arg-type]
    assert out == ["A"]


def test_pick_neutral_topup() -> None:
    """G2:行业配额不足时从全局剩余按分补足,且绝对最高分不被挤掉。"""
    score = pd.Series(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        index=[f"S{i}" for i in range(9)],
    )
    industry = {
        **{f"S{i}": "A" for i in range(4)},
        **{f"S{i}": "B" for i in range(4, 8)},
        "S8": "C",
    }
    picked = FactorPortfolioStrategy._pick_neutral(score, industry, 6)
    assert len(picked) == 6
    assert "S0" in picked  # 最高分不被行业配额挤掉
    assert "S8" in picked  # C 行业唯一票保留
    # 欠额补位从全局剩余按分补足(topup 不再重限行业配额)
    assert "S2" in picked  # 补位取全局剩余最高分


def test_pick_neutral_trim() -> None:
    """G2:超额时按全局分(已入选者)砍尾到 n。"""
    score = pd.Series(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], index=[f"S{i}" for i in range(8)]
    )
    industry = {f"S{i}": ("A" if i < 4 else "B") for i in range(8)}
    picked = FactorPortfolioStrategy._pick_neutral(score, industry, 3)
    assert len(picked) == 3
    # k=2 取 A 前2(S0,S1)+B 前2(S4,S5)=4 只,砍尾到 3 → 全局前 3 已入选者 [S0,S1,S4]
    assert set(picked) == {"S0", "S1", "S4"}


def test_pick_neutral_missing_map() -> None:
    """G2:industry_map 缺某票 → 归"未知"组参与。"""
    score = pd.Series([0.9, 0.5, 0.1], index=["A", "B", "C"])
    industry = {"A": "tech", "B": "tech"}  # C 缺映射
    picked = FactorPortfolioStrategy._pick_neutral(score, industry, 3)
    assert "C" in picked  # 未知组仍参与


def test_turnover_penalty_swaps_when_big_gap() -> None:
    """G4:新票得分优势 ≥ min_score_diff → 换入。"""
    strat = _bare_strat(min_score_diff=0.5)
    score = pd.Series({"X": 0.8, "Y": 1.5})

    class _Pos:
        def __init__(self) -> None:
            self.qty = 1.0

    class _Portfolio:
        def __init__(self) -> None:
            self.positions = {"X": _Pos()}

    class _Ctx:
        def __init__(self) -> None:
            self.portfolio = _Portfolio()

    out = strat._apply_turnover_penalty(_Ctx(), ["Y"], score)
    # Y(1.5) − X(0.8) = 0.7 ≥ 0.5 → 换入 Y
    assert out == ["Y"]


def test_turnover_penalty_keeps_old() -> None:
    strat = _bare_strat(min_score_diff=0.5)
    score = pd.Series({"X": 0.8, "Y": 1.0})

    class _Pos:
        def __init__(self) -> None:
            self.qty = 1.0

    class _Portfolio:
        def __init__(self) -> None:
            self.positions = {"X": _Pos()}

    class _Ctx:
        def __init__(self) -> None:
            self.portfolio = _Portfolio()

    out = strat._apply_turnover_penalty(_Ctx(), ["Y"], score)
    # Y(1.0) 相对 X(0.8) 优势 0.2 < 0.5 → 保留老票 X,拦下 Y
    assert out == ["X"]


# ── G7:注册与继承 ─────────────────────────────────────


def test_factor_timing_registered() -> None:
    """FactorTiming 注册进 STRATEGY_REGISTRY 且继承 FactorPortfolioStrategy。"""
    from djinn.strategy.library import STRATEGY_REGISTRY

    assert "FactorTiming" in STRATEGY_REGISTRY
    assert issubclass(STRATEGY_REGISTRY["FactorTiming"], FactorPortfolioStrategy)


# ── G0/G7:引擎级等价性安全网 ──────────────────────────


def test_select_pool_equiv() -> None:
    """G0:_select_pool 的选池 = 逐因子打分 nlargest(默认参数重构等价性)。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.screen import FactorScore

    days = pd.bdate_range("2024-01-01", periods=30)
    data = {s: _md(s, dict.fromkeys(days, 100.0)) for s in ["A", "B", "C"]}
    strat = FactorPortfolioStrategy(
        factors=[_RankFactor()],
        scores=[FactorScore(factor="rank", weight=1.0)],
        n_stocks=2,
        rebalance_freq=5,
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    held = [s for s in ["A", "B", "C"] if res.positions_curve[s].iloc[-1] > 0]
    # rank 末行给分 A=3/B=2/C=1 → Top2 = A、B;C 不入选
    assert "A" in held and "B" in held and "C" not in held
    # 权重和为 1(等权 0.5 + 0.5;撮合价差带来 ~5e-6 的微小超额,容差放宽)
    assert abs(sum(res.weights_curve[held].iloc[-1]) - 1.0) < 1e-4


def test_selection_log_recorded() -> None:
    """G9:调仓快照被记录(日期/名单/得分),并随报告 meta 序列化透出。"""
    from djinn.analytics.report import build_report
    from djinn.api.report_store import serialize_report
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.screen import FactorScore

    days = pd.bdate_range("2024-01-01", periods=30)
    data = {s: _md(s, dict.fromkeys(days, 100.0)) for s in ["A", "B", "C"]}
    strat = FactorPortfolioStrategy(
        factors=[_RankFactor()],
        scores=[FactorScore(factor="rank", weight=1.0)],
        n_stocks=2,
        rebalance_freq=5,
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    assert strat.selection_log, "调仓快照应被记录"
    entry = strat.selection_log[0]
    assert "date" in entry and "selected" in entry and "scores" in entry
    assert set(entry["selected"]) == {"A", "B"}
    # 序列化链:runner 写 report.meta → serialize_report 透出(供前端调仓快照 Tab)
    report = build_report(res)
    report.meta["selection_log"] = strat.selection_log
    payload = serialize_report(report)
    assert payload["meta"]["selection_log"][0]["selected"] == ["A", "B"]


class _FlipFactor:
    """A 前 20 日高分、后 10 日 C 高分(按日期翻转,供出池即卖测试)。"""

    name = "flip"
    max_lookback = 1

    def compute(self, prices, ohlcv, fundamentals):
        out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        flip_ts = prices.index[0] + pd.Timedelta(days=18)
        before = prices.index < flip_ts
        out.loc[before] = [3.0, 2.0, 1.0]  # A > B > C
        out.loc[~before] = [1.0, 2.0, 3.0]  # C > B > A
        return out


def test_beta_benchmark_in_strategy_path() -> None:
    """C6:FactorPortfolioStrategy 内 BetaFactor 注入真实基准(替代等权代理)。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.factor import make_factor
    from djinn.screen import FactorScore

    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=100)
    base = np.cumsum(rng.normal(0, 0.01, 100))
    b_close = 100 * np.exp(base)  # 与基准强相关
    a_close = 100 * np.exp(np.random.default_rng(1).normal(0, 0.02, 100))  # 无关
    bench = 100 * np.exp(base + np.random.default_rng(2).normal(0, 0.001, 100))
    data = {
        "A": _md("A", dict(zip(days, a_close, strict=True))),
        "B": _md("B", dict(zip(days, b_close, strict=True))),
    }
    bench_md = _md("BENCH", dict(zip(days, bench, strict=True)))
    strat = FactorPortfolioStrategy(
        factors=[make_factor("beta", period=20, benchmark="BENCH")],
        scores=[FactorScore(factor="beta", weight=1.0)],
        n_stocks=1,
        rebalance_freq=10,
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data, benchmark=bench_md)
    held = [s for s in ["A", "B"] if res.positions_curve[s].iloc[-1] > 0]
    # B 与基准高相关(beta≈1)被选,A 与基准无关(beta≈0)不选
    assert held == ["B"]


def test_out_of_pool_immediate() -> None:
    """G7:掉出池的票在调仓日立即清零(因子判决优先,不择时)。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.screen import FactorScore
    from djinn.strategy.library.factor_timing import FactorTimingStrategy

    days = pd.bdate_range("2024-01-01", periods=30)
    data = {s: _md(s, dict.fromkeys(days, 100.0)) for s in ["A", "B", "C"]}
    strat = FactorTimingStrategy(
        factors=[_FlipFactor()],
        scores=[FactorScore(factor="flip", weight=1.0)],
        n_stocks=2,
        rebalance_freq=10,
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    # 前段池=A,B(已持有);翻转后池=C,B → A 掉出池,调仓日立即清零
    assert res.positions_curve["A"].max() > 0  # A 曾被持有
    assert res.positions_curve["A"].iloc[-1] == 0.0
    assert res.positions_curve["C"].iloc[-1] > 0.0
    # A 的清仓单 tag=rebalance:out(因子判决,非择时 exit)
    outs = [f for f in res.trades if f.symbol == "A" and f.side == "sell"]
    assert outs and any("rebalance:out" in f.tag for f in outs)


def test_exit_and_cooldown() -> None:
    """G7:跌破 SMA20 出场 + 冷却期内不买 + 冷却结束站上 SMA20 再入场。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.screen import FactorScore
    from djinn.strategy.library.factor_timing import FactorTimingStrategy
    from djinn.strategy.timing import SMABreakExit

    days = pd.bdate_range("2024-01-01", periods=60)
    a_close = dict.fromkeys(days, 100.0)
    c_close = dict.fromkeys(days, 100.0)
    b_close: dict[pd.Timestamp, float] = {}
    for i, d in enumerate(days):
        if i < 40:
            b_close[d] = 100.0  # 入池后需 ≥20 个收盘给出场缓冲暖机
        elif i < 45:
            b_close[d] = 85.0  # 跌破 SMA20 → 出场
        else:
            b_close[d] = 115.0  # 站回 SMA20 之上 → 冷却后入场
    data = {
        "A": _md("A", a_close),
        "B": _md("B", b_close),
        "C": _md("C", c_close),
    }
    strat = FactorTimingStrategy(
        factors=[_RankFactor()],
        scores=[FactorScore(factor="rank", weight=1.0)],
        n_stocks=3,
        rebalance_freq=20,
        exit_rule=SMABreakExit(window=20),
        cooldown_days=5,
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    # 出场:B 跌破 SMA20 → tag=exit:SMABreakExit
    exits = [
        f
        for f in res.trades
        if f.symbol == "B" and f.side == "sell" and f.tag.startswith("exit:")
    ]
    assert exits, "应有 B 的择时出场单"
    # 再入场:冷却结束 + 站上 SMA20 → entry 单
    entries = [f for f in res.trades if f.symbol == "B" and f.side == "buy"]
    assert entries, "B 应在冷却结束后再入场"
    # B 出场与再入场之间有持仓为 0 的段(冷却),最终恢复持仓
    pos = res.positions_curve["B"]
    assert float(pos.min()) == 0.0  # 有过空仓段
    assert pos.iloc[-1] > 0.0


def test_cash_left_for_blocked() -> None:
    """G7:入场确认拦截某票 → 其份额留现金,总目标权重和 < 1(不摊给其他票)。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.screen import FactorScore
    from djinn.strategy.library.factor_timing import FactorTimingStrategy
    from djinn.strategy.timing import AboveSMAConfirm

    days = pd.bdate_range("2024-01-01", periods=40)
    # A 上行(站上 SMA20 可买);B 下行(跌破 SMA20 被拦截)
    a_close = {d: float(100.0 + 0.5 * i) for i, d in enumerate(days)}
    b_close = {d: float(110.0 - 0.5 * i) for i, d in enumerate(days)}
    data = {"A": _md("A", a_close), "B": _md("B", b_close)}
    strat = FactorTimingStrategy(
        factors=[_RankFactor()],
        scores=[FactorScore(factor="rank", weight=1.0)],
        n_stocks=2,
        rebalance_freq=20,
        entry_confirm=AboveSMAConfirm(window=20),
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    # A 被买入,B 被拦截 → 总仓位 ≈ 0.5 < 1(差额留现金)
    assert res.positions_curve["A"].iloc[-1] > 0.0
    assert res.positions_curve["B"].iloc[-1] == 0.0
    assert res.weights_curve.sum(axis=1).iloc[-1] < 1.0


def test_regime_scales_weights() -> None:
    """G7:市场闸门开启(基准跌破 SMA)→ 新入场权重 = base_w × cap,留现金。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.screen import FactorScore
    from djinn.strategy.library.factor_timing import FactorTimingStrategy
    from djinn.strategy.timing import MarketRegimeFilter

    days = pd.bdate_range("2024-01-01", periods=30)
    data = {s: _md(s, dict.fromkeys(days, 100.0)) for s in ["A", "B", "C"]}
    # 基准持续下跌:第 6 日起 close < SMA(5) → 闸门 cap=0.3
    bench_close = np.linspace(100.0, 95.0, 30)
    bench = _md("BENCH", {d: float(v) for d, v in zip(days, bench_close, strict=True)})
    strat = FactorTimingStrategy(
        factors=[_RankFactor()],
        scores=[FactorScore(factor="rank", weight=1.0)],
        n_stocks=2,
        rebalance_freq=20,
        regime=MarketRegimeFilter(window=5, floor=0.3),
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data, benchmark=bench)
    # 末段闸门已关(cap=0.3):A/B 权重被缩到 ~0.15 而非 0.5,总仓位 < 1(留现金)
    w = res.weights_curve[["A", "B"]].iloc[-1].sum()
    assert w > 0.1, f"应有持仓,实际总权重 {w}"
    assert w < 0.6, f"闸门未缩放权重(总仓位 {w},应为 ~0.3)"


def test_icir_no_window_starvation() -> None:
    """C9:icir 加权的调仓面板覆盖滚动 ICIR 窗口(旧 D3 截断按 max_lookback 会饿死→选不出)。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.factor import make_factor
    from djinn.screen import FactorScore

    days = pd.bdate_range("2024-01-01", periods=300)
    # 不同斜率 → 动量截面有信号 → ICIR 应产出非零权重
    data = {
        s: _md(
            s, {d: 100.0 + j * (0.1 + i * 0.05) for j, d in enumerate(days)}
        )
        for i, s in enumerate(["A", "B", "C"])
    }
    strat = FactorPortfolioStrategy(
        factors=[make_factor("momentum", period=10)],
        scores=[FactorScore(factor="momentum", weight=1.0)],
        n_stocks=2,
        rebalance_freq=20,
        weighting="icir",
        icir_window=60,
        icir_min_periods=20,
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    assert strat.selection_log, "icir 应能选出股票(ICIR 窗口未被截断饿死)"
    assert any(res.positions_curve[s].iloc[-1] > 0 for s in ["A", "B", "C"])


def test_benchmark_in_ctx() -> None:
    """G6:引擎注入 benchmark 后,策略 ctx.benchmark_close() = 基准当日收盘。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.strategy.base import Context, Strategy

    days = pd.bdate_range("2024-01-01", periods=10)
    closes = {d: 100.0 + i for i, d in enumerate(days)}
    data = {"S": _md("S", closes)}
    bench = _md("BENCH", closes)

    seen: list[float | None] = []

    class _Capture(Strategy):
        def on_bar(self, ctx: Context) -> None:
            seen.append(ctx.benchmark_close())

    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0))
    eng.run(_Capture(), data, benchmark=bench)
    assert seen == [100.0 + i for i in range(10)]  # 逐日与基准收盘一致
    # 无 benchmark → None 不抛错
    seen2: list[float | None] = []

    class _Capture2(Strategy):
        def on_bar(self, ctx: Context) -> None:
            seen2.append(ctx.benchmark_close())

    eng.run(_Capture2(), data)
    assert all(v is None for v in seen2)


def test_tag_in_trades() -> None:
    """G9:成交 tag 归因 —— 买入单含 rebalance:in(稳定池场景)。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.screen import FactorScore

    days = pd.bdate_range("2024-01-01", periods=30)
    data = {s: _md(s, dict.fromkeys(days, 100.0)) for s in ["A", "B", "C"]}
    strat = FactorPortfolioStrategy(
        factors=[_RankFactor()],
        scores=[FactorScore(factor="rank", weight=1.0)],
        n_stocks=2,
        rebalance_freq=5,
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    buys = [f for f in res.trades if f.side == "buy"]
    assert buys and all("rebalance:in" in f.tag for f in buys)


def _synthetic_panel(n_bull: int = 40, n_bear: int = 20, n_rec: int = 60):
    """构造 10 票合成面板:tech(前 5)动量更高、熊市段(day40~60)全跌。

    Returns:
        (days, data, industries, bench_md) —— tech 高分 → 基线全选 tech(高 HHI)。
    """
    days = pd.bdate_range("2024-01-01", periods=n_bull + n_bear + n_rec)
    rng = np.random.default_rng(7)
    industries = {f"S{i}": ("tech" if i < 5 else "fin") for i in range(10)}
    data: dict = {}
    closes_all: list[list[float]] = []
    for i in range(10):
        slope = 0.005 if i < 5 else 0.002  # tech 涨得更快
        closes = [100.0]
        for d in range(1, len(days)):
            if n_bull <= d < n_bull + n_bear:
                ret = -0.008  # 熊市段全跌
            else:
                ret = slope + rng.normal(0, 0.0005)
            closes.append(closes[-1] * (1 + ret))
        closes_all.append(closes)
        data[f"S{i}"] = _md(f"S{i}", dict(zip(days, closes, strict=True)))
    # 基准 = 等权市场(熊市段跌破 SMA20 → 闸门触发)
    avg_close = [sum(cs[j] for cs in closes_all) / 10 for j in range(len(days))]
    bench_md = _md("BENCH", dict(zip(days, avg_close, strict=True)))
    return days, data, industries, bench_md


def _hhi(weights: dict[str, float], industries: dict[str, str]) -> float:
    """行业权重 HHI(∑ 行业权重²)。"""
    by_ind: dict[str, float] = {}
    for s, w in weights.items():
        by_ind[industries.get(s, "未知")] = (
            by_ind.get(industries.get(s, "未知"), 0.0) + w
        )
    return sum(v * v for v in by_ind.values())


def test_end_to_end_synthetic() -> None:
    """G9:基线 vs +selection vs +timing 三档(行业集中度降/换手降/熊市回撤浅)。"""
    from djinn.analytics.report import build_report
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.factor import make_factor
    from djinn.screen import FactorScore
    from djinn.strategy.library.factor_timing import FactorTimingStrategy
    from djinn.strategy.timing import MarketRegimeFilter, SMABreakExit

    days, data, industries, bench_md = _synthetic_panel()
    momentum = make_factor("momentum", period=10)
    scores = [FactorScore(factor="momentum", weight=1.0)]
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))

    def run(strat, use_bench=False):
        res = eng.run(strat, data, benchmark=bench_md if use_bench else None)
        rep = build_report(res)
        last_w = {s: float(w) for s, w in res.weights_curve.iloc[-1].items() if w > 0}
        return {
            "hhi": _hhi(last_w, industries),
            "turnover": rep.metrics.turnover,
            "max_drawdown": rep.metrics.max_drawdown,
        }

    base = run(
        FactorPortfolioStrategy(
            factors=[momentum], scores=scores, n_stocks=6, rebalance_freq=20
        )
    )
    sel = run(
        FactorPortfolioStrategy(
            factors=[momentum],
            scores=scores,
            n_stocks=6,
            rebalance_freq=20,
            industry_neutral=True,
            industry_map=industries,
            min_score_diff=0.3,
        )
    )
    timing = run(
        FactorTimingStrategy(
            factors=[momentum],
            scores=scores,
            n_stocks=6,
            rebalance_freq=20,
            regime=MarketRegimeFilter(window=20, floor=0.3),
            exit_rule=SMABreakExit(window=10),
        ),
        use_bench=True,
    )

    # ① 行业中性 → 行业集中度下降
    assert sel["hhi"] < base["hhi"], f"行业集中度应降: {base['hhi']} → {sel['hhi']}"
    # ② 换手惩罚 → 换手下降
    assert (
        sel["turnover"] < base["turnover"]
    ), f"换手应降: {base['turnover']} → {sel['turnover']}"
    # ③ 两层择时(闸门+出场)→ 熊市段最大回撤变浅(更接近 0)
    assert (
        timing["max_drawdown"] > base["max_drawdown"]
    ), f"熊市回撤应变浅: {base['max_drawdown']} → {timing['max_drawdown']}"


def test_factor_timing_equiv_when_no_timing() -> None:
    """G7:regime/exit/confirm 全 None 时,FactorTiming 与父类行为逐一致。"""
    from djinn.engine import EngineConfig, EventDrivenEngine
    from djinn.screen import FactorScore
    from djinn.strategy.library.factor_timing import FactorTimingStrategy

    days = pd.bdate_range("2024-01-01", periods=30)
    data = {s: _md(s, dict.fromkeys(days, 100.0)) for s in ["A", "B", "C"]}
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))

    def held(strat) -> dict[str, float]:
        res = eng.run(strat, data)
        return {s: float(res.positions_curve[s].iloc[-1]) for s in ["A", "B", "C"]}

    base = FactorPortfolioStrategy(
        factors=[_RankFactor()],
        scores=[FactorScore(factor="rank", weight=1.0)],
        n_stocks=2,
        rebalance_freq=5,
    )
    timing = FactorTimingStrategy(
        factors=[_RankFactor()],
        scores=[FactorScore(factor="rank", weight=1.0)],
        n_stocks=2,
        rebalance_freq=5,
        regime=None,
        exit_rule=None,
        entry_confirm=None,
    )
    h_base = held(base)
    h_timing = held(timing)
    for s in ["A", "B", "C"]:
        assert (h_base[s] > 0) == (h_timing[s] > 0), f"标的 {s} 持仓不一致"
    # 无择时 → 权重不缩放(无闸门),调仓日买入 tag 为 rebalance:in
    res_t = eng.run(timing, data)
    buys = [f for f in res_t.trades if f.side == "buy"]
    assert buys and all("rebalance:in" in f.tag for f in buys)

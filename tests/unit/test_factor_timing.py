"""G 计划:择时规则库 + 选股流水线增强测试。"""

from __future__ import annotations

import pandas as pd
import pytest

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

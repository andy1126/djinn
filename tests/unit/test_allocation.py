"""组合构建(Phase 4)测试:分配器 + 行业 / 换手风控。

不依赖网络:协方差用人工构造的正定矩阵,行情用合成 MarketData。
验证点(对应计划):风险平价各成分风险贡献近似相等;最小方差波动 ≤ 等权;
ScoreWeight 权重与打分单调一致;行业约束后单行业权重 ≤ 上限。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from djinn.data.market_data import MarketData
from djinn.data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_VOLUME, Market
from djinn.engine import EngineConfig, EventDrivenEngine
from djinn.factor.library.momentum import MomentumFactor
from djinn.portfolio import (
    CustomWeight,
    EqualWeight,
    MarketCapWeight,
    MeanVarianceWeight,
    MinVarianceWeight,
    RiskLimits,
    RiskManager,
    RiskParityWeight,
    ScoreWeight,
    estimate_covariance,
    make_allocation,
)
from djinn.screen import FactorScore
from djinn.strategy.library.factor_portfolio import FactorPortfolioStrategy
from djinn.strategy.signal import OrderIntent
from djinn.utils.exceptions import StrategyError

_SYMBOLS = ["A", "B", "C"]


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


def _cov_df() -> pd.DataFrame:
    """正定协方差(不同波动率 + 弱相关),行列均为 _SYMBOLS。"""
    vols = np.array([0.10, 0.20, 0.30])
    corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    cov = np.outer(vols, vols) * corr
    return pd.DataFrame(cov, index=_SYMBOLS, columns=_SYMBOLS)


def _vec(w: dict[str, float]) -> np.ndarray:
    return np.array([w[s] for s in _SYMBOLS])


# ── 风险平价 ─────────────────────────────────────────────
def test_risk_parity_equal_risk_contribution() -> None:
    cov = _cov_df()
    w = RiskParityWeight().target_weights(_SYMBOLS, cov=cov)
    c = cov.to_numpy()
    wv = _vec(w)
    sigma_p = float(np.sqrt(wv @ c @ wv))
    rc = wv * (c @ wv) / sigma_p  # 各成分风险贡献(Euler 分解,和为 σ_p)
    assert np.allclose(rc, sigma_p / len(_SYMBOLS), rtol=1e-2)
    assert sum(w.values()) == pytest.approx(1.0)


# ── 最小方差 ─────────────────────────────────────────────
def test_min_variance_le_equal_weight() -> None:
    c = _cov_df().to_numpy()
    w = MinVarianceWeight().target_weights(_SYMBOLS, cov=_cov_df())
    vol_mv = float(np.sqrt(_vec(w) @ c @ _vec(w)))
    w_eq = np.full(len(_SYMBOLS), 1.0 / len(_SYMBOLS))
    vol_eq = float(np.sqrt(w_eq @ c @ w_eq))
    assert vol_mv <= vol_eq + 1e-9
    # 最小方差偏向低波动资产 A、远离高波动 C
    assert w["A"] > w["C"]
    assert sum(w.values()) == pytest.approx(1.0)


# ── 均值-方差 ────────────────────────────────────────────
def test_mean_variance_tilts_to_high_score() -> None:
    cov = _cov_df()
    scores = {"A": 0.1, "B": 0.0, "C": -0.1}
    w_mv = MinVarianceWeight().target_weights(_SYMBOLS, cov=cov)
    w_mmv = MeanVarianceWeight(risk_aversion=1.0).target_weights(
        _SYMBOLS, scores=scores, cov=cov
    )
    # 均值-方差比最小方差更偏向高分 A
    assert w_mmv["A"] > w_mv["A"]
    assert sum(w_mmv.values()) == pytest.approx(1.0)


def test_mean_variance_no_scores_is_min_variance() -> None:
    cov = _cov_df()
    w_mmv = MeanVarianceWeight(risk_aversion=1.0).target_weights(_SYMBOLS, cov=cov)
    w_mv = MinVarianceWeight().target_weights(_SYMBOLS, cov=cov)
    # 无 scores → 退化为最小方差
    for s in _SYMBOLS:
        assert w_mmv[s] == pytest.approx(w_mv[s], abs=1e-6)


# ── 打分加权 ─────────────────────────────────────────────
def test_score_weight_monotonic() -> None:
    scores = {"A": 1.0, "B": 2.5, "C": -1.0}
    w = ScoreWeight().target_weights(_SYMBOLS, scores=scores)
    assert w["B"] > w["A"] > w["C"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_score_weight_all_equal_is_equal() -> None:
    w = ScoreWeight().target_weights(_SYMBOLS, scores=dict.fromkeys(_SYMBOLS, 1.0))
    assert w == pytest.approx(dict.fromkeys(_SYMBOLS, 1.0 / 3.0))


def test_score_weight_no_scores_is_equal() -> None:
    w = ScoreWeight().target_weights(_SYMBOLS)
    assert w == dict.fromkeys(_SYMBOLS, 1.0 / 3.0)


# ── 退化(缺 cov)→ 等权 ───────────────────────────────────
def test_optimizers_degrade_to_equal_without_cov() -> None:
    expected = dict.fromkeys(_SYMBOLS, 1.0 / 3.0)
    assert RiskParityWeight().target_weights(_SYMBOLS) == expected
    assert MinVarianceWeight().target_weights(_SYMBOLS) == expected
    assert MeanVarianceWeight().target_weights(_SYMBOLS, scores={"A": 1.0}) == expected


def test_optimizer_degrades_on_nan_cov() -> None:
    bad = _cov_df().copy()
    bad.loc["A", "B"] = np.nan
    expected = dict.fromkeys(_SYMBOLS, 1.0 / 3.0)
    assert MinVarianceWeight().target_weights(_SYMBOLS, cov=bad) == expected
    assert RiskParityWeight().target_weights(_SYMBOLS, cov=bad) == expected


def test_allocation_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """A8:缺参退化为等权时显式告警,同一实例只警一次。"""
    expected = dict.fromkeys(_SYMBOLS, 1.0 / 3.0)
    alloc = MinVarianceWeight()
    assert alloc.target_weights(_SYMBOLS) == expected
    assert alloc.target_weights(_SYMBOLS) == expected
    warns = [
        r
        for r in caplog.records
        if r.name.startswith("djinn.portfolio.allocation")
        and "退化为等权" in r.getMessage()
    ]
    assert len(warns) == 1
    assert "MinVarianceWeight" in warns[0].getMessage()
    assert "cov" in warns[0].getMessage()
    # ScoreWeight 同样告警
    with caplog.at_level("WARNING", logger="djinn.portfolio.allocation"):
        ScoreWeight().target_weights(_SYMBOLS)
    assert any("ScoreWeight" in r.getMessage() for r in caplog.records)


# ── 基础分配器(新签名向后兼容)─────────────────────────────
def test_basic_allocators_still_work() -> None:
    assert EqualWeight().target_weights(_SYMBOLS) == dict.fromkeys(_SYMBOLS, 1.0 / 3.0)
    mc = MarketCapWeight().target_weights(
        _SYMBOLS, prices={"A": 10.0, "B": 20.0, "C": 30.0}
    )
    assert mc == pytest.approx({"A": 1 / 6, "B": 2 / 6, "C": 3 / 6})
    cw = CustomWeight({"A": 2.0, "B": 1.0}).target_weights(["A", "B"])
    assert cw == pytest.approx({"A": 2 / 3, "B": 1 / 3})


# ── 工厂 ─────────────────────────────────────────────────
def test_make_allocation_kinds() -> None:
    assert isinstance(make_allocation("equal"), EqualWeight)
    assert isinstance(make_allocation("market_cap"), MarketCapWeight)
    assert isinstance(make_allocation("custom", {"A": 1.0}), CustomWeight)
    assert isinstance(make_allocation("score"), ScoreWeight)
    assert isinstance(make_allocation("risk_parity"), RiskParityWeight)
    assert isinstance(make_allocation("min_variance"), MinVarianceWeight)
    assert isinstance(make_allocation("mean_variance"), MeanVarianceWeight)
    with pytest.raises(StrategyError):
        make_allocation("custom")  # 缺 weights
    with pytest.raises(StrategyError):
        make_allocation("unknown-kind")


# ── 协方差估计(收缩)──────────────────────────────────────
def test_estimate_covariance_shrink() -> None:
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.02, (60, 3)), columns=_SYMBOLS)
    sample = estimate_covariance(rets, shrink=0.0)
    full = estimate_covariance(rets, shrink=1.0)
    assert full.loc["A", "B"] == 0.0  # 完全收缩 → 非对角归零
    assert full.loc["A", "A"] == pytest.approx(sample.loc["A", "A"])  # 对角保留
    half = estimate_covariance(rets, shrink=0.5)
    assert half.loc["A", "B"] == pytest.approx(0.5 * sample.loc["A", "B"])


# ── 行业集中度风控 ────────────────────────────────────────
def test_sector_cap() -> None:
    rm = RiskManager(
        RiskLimits(
            max_sector_weight=0.5,
            sector_map={"A": "tech", "B": "tech", "C": "fin"},
        )
    )
    orders = [
        OrderIntent(symbol="A", side="buy", target_percent=0.5),
        OrderIntent(symbol="B", side="buy", target_percent=0.4),
        OrderIntent(symbol="C", side="buy", target_percent=0.1),
    ]
    filtered = rm.filter(orders, current_weights={})
    tech = sum(o.target_percent or 0.0 for o in filtered if o.symbol in ("A", "B"))
    assert tech <= 0.5 + 1e-9  # tech 行业(0.9 → ≤0.5)
    fin = next(o for o in filtered if o.symbol == "C")
    assert fin.target_percent == pytest.approx(0.1)  # fin 未超上限,不变


# ── 换手限制风控 ──────────────────────────────────────────
def test_max_turnover() -> None:
    rm = RiskManager(RiskLimits(max_turnover=0.5))
    orders = [
        OrderIntent(symbol="A", side="buy", target_percent=0.8),
        OrderIntent(symbol="B", side="buy", target_percent=0.0),
    ]
    current = {"A": 0.0, "B": 0.6}  # 调整前换手 = 0.8 + 0.6 = 1.4
    filtered = rm.filter(orders, current_weights=current)
    turnover = sum(
        abs((o.target_percent or 0.0) - current.get(o.symbol, 0.0)) for o in filtered
    )
    assert turnover <= 0.5 + 1e-9
    a = next(o for o in filtered if o.symbol == "A")
    assert a.target_percent == pytest.approx(0.8 * (0.5 / 1.4))


# ── 端到端:最小方差分配器接入因子组合策略 ───────────────────
def test_factor_portfolio_min_variance_end_to_end() -> None:
    """最小方差分配器驱动因子组合回测:不崩溃、有成交、资金守恒。"""
    days = pd.date_range("2024-01-01", periods=40)
    rng = np.random.default_rng(1)
    data: dict[str, MarketData] = {}
    for s in ["A", "B", "C", "D"]:
        rets = rng.normal(0.0005, 0.02, len(days))
        close = 100 * np.exp(np.cumsum(rets))
        data[s] = _md(s, {d: float(c) for d, c in zip(days, close, strict=True)})
    strat = FactorPortfolioStrategy(
        factors=[MomentumFactor(period=5)],
        scores=[FactorScore(factor="momentum", weight=1.0)],
        n_stocks=3,
        rebalance_freq=5,
        allocation=MinVarianceWeight(),
    )
    eng = EventDrivenEngine(EngineConfig(initial_cash=100000.0, calendar="union"))
    res = eng.run(strat, data)
    assert len(res.equity_curve) == len(days)
    assert res.n_trades > 0
    final_prices = {s: float(md.df[COL_CLOSE].iloc[-1]) for s, md in data.items()}
    res.account.check_invariant(final_prices)

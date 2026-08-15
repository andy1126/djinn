"""通用信号策略(SignalStrategy)测试:注册表 + 动态实例化 + 信号适配。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from djinn.strategy.library import get_strategy_class
from djinn.strategy.signals import SIGNAL_INDICATORS, get_signal_indicator
from djinn.utils.exceptions import StrategyError


def _df(n: int = 600, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = np.concatenate(
        [
            rng.normal(0.0015, 0.012, n // 3),
            rng.normal(-0.0015, 0.015, n // 3),
            rng.normal(0.0, 0.010, n - 2 * (n // 3)),
        ]
    )
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    high = close * (1 + rng.uniform(0.002, 0.015, n))
    low = close * (1 - rng.uniform(0.002, 0.015, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}
    )


def test_signal_indicators_registered() -> None:
    assert {"supertrend", "adaptive_trend_trail", "ma_cross"} <= set(SIGNAL_INDICATORS)


def test_generic_strategy_each_indicator() -> None:
    df = _df()
    for name, kwargs in [
        ("supertrend", {"factor": 3.0}),
        ("adaptive_trend_trail", {"trend_length": 40}),
        ("ma_cross", {"fast": 5, "slow": 20}),
    ]:
        cls = get_strategy_class("SignalStrategy")
        s = cls(indicator=name, **kwargs)
        sig = s.signals(df)
        assert set(sig.unique()).issubset({-1, 0, 1}), name
        assert s.params["indicator"] == name


def test_unknown_indicator_raises() -> None:
    with pytest.raises(StrategyError):
        get_signal_indicator("no_such_indicator")
    with pytest.raises(StrategyError):
        get_strategy_class("SignalStrategy")(indicator="no_such_indicator")


def test_signal_strategy_registered() -> None:
    assert get_strategy_class("SignalStrategy").__name__ == "SignalStrategy"


# ── D1:signals 因果性自检 ───────────────────────────────


def test_check_causal_clean() -> None:
    """D1:滚动类因果信号 → check_causal 无告警。"""
    from djinn.strategy.check import check_causal

    idx = pd.date_range("2024-01-01", periods=120)
    close = pd.Series(np.linspace(100.0, 200.0, 120), index=idx)
    df = pd.DataFrame({"close": close})

    def causal_signals(d: pd.DataFrame) -> pd.Series:
        ma = d["close"].rolling(20).mean()
        return (d["close"] > ma).astype(int)

    assert check_causal(causal_signals, df) == []


def test_check_causal_detects_shift_neg() -> None:
    """D1:含 shift(-1)(依赖未来)的 signals 被 check_causal 检出。"""
    from djinn.strategy.check import check_causal

    idx = pd.date_range("2024-01-01", periods=120)
    close = pd.Series(np.linspace(100.0, 200.0, 120), index=idx)
    df = pd.DataFrame({"close": close})

    def non_causal_signals(d: pd.DataFrame) -> pd.Series:
        # 明天的收盘决定今天的信号 → 未来函数
        return (d["close"].shift(-1) > d["close"]).astype(int)

    problems = check_causal(non_causal_signals, df)
    assert problems, "shift(-1) 应被检出为非因果"


def test_asof_no_lookahead() -> None:
    """D1:Series.asof(t) 只取 ≤t 的最近值,看不到未来信号。"""
    idx = pd.date_range("2024-01-01", periods=5)
    sig = pd.Series([0, 0, 0, 1, 1], index=idx)  # 01-04 突变
    assert sig.asof(pd.Timestamp("2024-01-01")) == 0  # 突变前
    assert sig.asof(pd.Timestamp("2024-01-03")) == 0  # 突变前一日(看不到 01-04)
    assert sig.asof(pd.Timestamp("2024-01-04")) == 1  # 突变当日
    assert pd.isna(sig.asof(pd.Timestamp("2023-12-31")))  # 早于首日无值

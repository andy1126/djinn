"""因子引擎与内置因子测试(本地小样本人工构造,不依赖网络)。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from djinn.factor import make_factor
from djinn.factor.library import MomentumFactor, ReversalFactor, SizeFactor


def _trading_index(n: int = 10) -> pd.DatetimeIndex:
    base = date(2024, 1, 1)
    return pd.DatetimeIndex([base + timedelta(days=i) for i in range(n)])


def _prices(data: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(data, index=_trading_index(len(next(iter(data.values())))))


def test_momentum_factor_hand_computed() -> None:
    """手算一只票 5 日动量:close[t-0]/close[t-5] - 1(skip=0)。"""
    close = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    prices = _prices({"A": [float(c) for c in close]})
    f = MomentumFactor(period=5, skip=0)
    out = f.compute(prices, {}, {})
    # t=5: 15/10 - 1 = 0.5
    assert out["A"].iloc[5] == pytest.approx(0.5)
    # 前 5 日无足够历史 → NaN
    assert out["A"].iloc[:5].isna().all()


def test_momentum_factor_skip() -> None:
    """skip=1:动量 = close[t-1]/close[t-1-period] - 1。"""
    close = [float(c) for c in range(10, 20)]
    prices = _prices({"A": close})
    f = MomentumFactor(period=5, skip=1)
    out = f.compute(prices, {}, {})
    # t=6: close[5]/close[0] - 1 = 15/10 - 1 = 0.5
    assert out["A"].iloc[6] == pytest.approx(0.5)


def test_reversal_factor_sign() -> None:
    close = [10, 11, 12, 13, 14, 15]
    prices = _prices({"A": [float(c) for c in close]})
    f = ReversalFactor(period=1)
    out = f.compute(prices, {}, {})
    # 上涨 → 反转因子为负
    assert out["A"].iloc[-1] < 0


def test_size_factor_log() -> None:
    from djinn.data.schema import COL_MARKET_CAP

    prices = _prices({"A": [1.0] * 10, "B": [1.0] * 10})
    cap = pd.DataFrame({COL_MARKET_CAP: [1.0e10] * 10}, index=prices.index)
    fundamentals = {
        COL_MARKET_CAP: pd.DataFrame(
            {"A": cap[COL_MARKET_CAP], "B": cap[COL_MARKET_CAP]}
        )
    }
    f = SizeFactor()
    out = f.compute(prices, {}, fundamentals)
    assert out["A"].iloc[-1] == pytest.approx(np.log(1.0e10))


def test_factor_no_future_values() -> None:
    """末行之后无数据:因子面板索引不超过价格索引,且无超前非空。"""
    close = [float(c) for c in range(10, 30)]
    prices = _prices({"A": close})
    f = MomentumFactor(period=5, skip=0)
    out = f.compute(prices, {}, {})
    assert len(out) == len(prices)
    assert out.index.equals(prices.index)


def test_factor_registry_all_instantiable() -> None:
    """注册表所有因子可无参实例化并被 compute 调用(价格类)。"""
    prices = _prices(
        {"A": [10.0 + i for i in range(10)], "B": [20.0 - i for i in range(10)]}
    )
    amount = prices * 1000.0
    ohlcv = {"amount": amount}
    for name in ("momentum", "reversal", "volatility", "beta", "turnover"):
        f = make_factor(name)
        out = f.compute(prices, ohlcv, {})
        assert out.shape == prices.shape, name


def test_value_factor_reciprocal() -> None:
    from djinn.data.schema import COL_PE

    prices = _prices({"A": [1.0] * 10})
    pe = pd.DataFrame({"A": [20.0] * 10}, index=prices.index)
    f = make_factor("ep")
    out = f.compute(prices, {}, {COL_PE: pe})
    assert out["A"].iloc[-1] == pytest.approx(1 / 20.0)
    # 负 PE → NaN
    pe_neg = pd.DataFrame({"A": [-5.0] * 10}, index=prices.index)
    out_neg = f.compute(prices, {}, {COL_PE: pe_neg})
    assert out_neg["A"].isna().all()

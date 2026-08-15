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


def test_beta_factor_real_benchmark() -> None:
    """C6:BetaFactor 用真实基准(ohlcv["__benchmark__"])时 beta 精确为 ±1。"""
    from djinn.factor.library import BetaFactor

    n = 80
    idx = _trading_index(n)
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    mkt_vals = mkt.to_numpy()
    # close = cumprod(1+r) → pct_change 精确等于 r / -r
    prices = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1.0 + mkt_vals),
            "B": 100 * np.cumprod(1.0 - mkt_vals),
        },
        index=idx,
    )
    f = BetaFactor(period=20)
    out = f.compute(prices, {"__benchmark__": mkt}, {})
    assert out["A"].iloc[-1] == pytest.approx(1.0, abs=1e-6)
    assert out["B"].iloc[-1] == pytest.approx(-1.0, abs=1e-6)


def test_beta_factor_benchmark_degrade() -> None:
    """C6:无基准注入 → 退化为等权代理,不抛错。"""
    from djinn.factor.library import BetaFactor

    prices = _prices(
        {"A": [10.0 + i for i in range(60)], "B": [20.0 - i for i in range(60)]}
    )
    out = BetaFactor(period=20).compute(prices, {}, {})
    assert not out["A"].iloc[-1:].isna().all()


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


def test_high_52w_factor() -> None:
    """52 周高点距离:末日创窗口新高 → 因子值为 0。"""
    close = [float(c) for c in range(10, 30)]
    prices = _prices({"A": close})
    f = make_factor("high_52w", window=20)
    out = f.compute(prices, {}, {})
    # 末日为窗口内最高 → close/max = 1 → 因子 0
    assert out["A"].iloc[-1] == pytest.approx(0.0)
    assert out["A"].iloc[:19].isna().all()  # 前 19 日窗口不足 → NaN


def test_new_factors_registered() -> None:
    """C7 新增因子均注册进 FACTOR_REGISTRY 且可实例化。"""
    from djinn.factor.library import FACTOR_REGISTRY

    new_names = [
        "high_52w",
        "max_lottery",
        "idio_vol",
        "turnover_chg",
        "accruals",
        "asset_growth",
        "cfp",
        "div_yield",
    ]
    for name in new_names:
        assert name in FACTOR_REGISTRY, name
        f = make_factor(name)
        assert f.name == name


def test_lookback_truncation_equal() -> None:
    """D3:全历史面板末行 vs 截断面板末行逐值一致(滚动只依赖最近 lb 日)。"""
    n = 300
    prices = _prices({"A": [float(10 + i) for i in range(n)]})
    f = make_factor("momentum", period=20, skip=0)
    full = f.compute(prices, {}, {})
    cutoff = prices.index[-1] - pd.Timedelta(days=int(20 * 1.6) + 30)
    prices_win = prices.loc[prices.index >= cutoff]
    trunc = f.compute(prices_win, {}, {})
    assert full.iloc[-1]["A"] == pytest.approx(trunc.iloc[-1]["A"], rel=1e-12)


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


def test_div_yield_factor() -> None:
    """C7 股息率 = 近 12 个月每股现金分红(TTM)/ 收盘价。"""
    from djinn.data.schema import COL_DIVIDEND

    prices = _prices({"A": [10.0] * 10})
    # 第 5 日派息 0.5 元/股,其余日 0
    div = pd.DataFrame({"A": [0.0] * 10}, index=prices.index)
    div.iloc[5, 0] = 0.5
    f = make_factor("div_yield")
    out = f.compute(prices, {}, {COL_DIVIDEND: div})
    # 派息日前 TTM=0;派息日及之后 TTM=0.5 → 股息率 0.05
    assert out["A"].iloc[:5].eq(0.0).all()
    assert out["A"].iloc[5] == pytest.approx(0.5 / 10.0)
    assert out["A"].iloc[-1] == pytest.approx(0.5 / 10.0)


def test_div_yield_no_dividend_is_zero() -> None:
    """无分红记录 → 股息率为 0(而非 NaN)。"""
    from djinn.data.schema import COL_DIVIDEND

    prices = _prices({"A": [10.0] * 10})
    div = pd.DataFrame({"A": [0.0] * 10}, index=prices.index)
    out = make_factor("div_yield").compute(prices, {}, {COL_DIVIDEND: div})
    assert out["A"].eq(0.0).all()


def test_net_profit_margin_nonzero() -> None:
    """净利率因子 = net_profit / revenue,非 NaN(修复 DEFAULT_FUNDAMENTAL_FIELDS 缺字段)。"""
    from djinn.data.schema import COL_NET_PROFIT, COL_REVENUE

    prices = _prices({"A": [1.0] * 10})
    net_profit = pd.DataFrame({"A": [100.0] * 10}, index=prices.index)
    revenue = pd.DataFrame({"A": [1000.0] * 10}, index=prices.index)
    f = make_factor("net_profit_margin")
    out = f.compute(prices, {}, {COL_NET_PROFIT: net_profit, COL_REVENUE: revenue})
    assert out["A"].iloc[-1] == pytest.approx(0.1)
    assert not out["A"].isna().all()


def test_factor_missing_fields_raises() -> None:
    """声明 required_fundamentals 的因子在缺字段时,FactorEngine 抛 FactorError。"""
    from datetime import date

    from djinn.data.market_data import MarketData
    from djinn.data.provider import DataProvider, ProviderRegistry
    from djinn.data.schema import (
        COL_CLOSE,
        COL_HIGH,
        COL_LOW,
        COL_OPEN,
        COL_VOLUME,
        Adjust,
        Market,
    )
    from djinn.factor import Factor, FactorEngine
    from djinn.factor.base import Panel, PanelDict
    from djinn.utils.exceptions import FactorError

    class _Stub(DataProvider):
        name = "stub"
        market = Market.CN

        def supports(self, symbol: str, market: Market | None = None) -> bool:
            return True

        def get_ohlcv(
            self, symbol: str, start: date, end: date, adjust: Adjust = Adjust.BACKWARD
        ) -> MarketData:
            idx = pd.bdate_range(start, end)
            n = len(idx)
            return MarketData(
                symbol=symbol,
                market=Market.CN,
                df=pd.DataFrame(
                    {
                        COL_OPEN: [10.0] * n,
                        COL_HIGH: [10.0] * n,
                        COL_LOW: [10.0] * n,
                        COL_CLOSE: [10.0] * n,
                        COL_VOLUME: [1.0e6] * n,
                    },
                    index=idx,
                ),
                adjust=adjust,
            )

    class MissingFieldFactor(Factor):
        name = "missing_field"
        required_fundamentals = ("nonexistent",)

        def compute(
            self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
        ) -> Panel:
            return prices

    eng = FactorEngine()
    with pytest.raises(FactorError):
        eng.compute(
            [MissingFieldFactor()],
            ["600000.SH"],
            date(2024, 1, 1),
            date(2024, 1, 10),
            ProviderRegistry([_Stub()]),
        )


def test_factor_required_declarations_complete() -> None:
    """元测试:compute() 引用的基本面/行情 COL 常量必须已声明 required_*。

    遍历 FACTOR_REGISTRY,凡 compute 源码中出现基本面粉本字段常量
    (fund_panel / fundamentals.get 的参数),其字段值必须在 required_fundamentals;
    行情字段(COL_AMOUNT 等)必须在 required_ohlcv。防止新增因子漏声明而静默 NaN。
    """
    import inspect
    import re

    import djinn.data.schema as schema
    from djinn.factor.library import FACTOR_REGISTRY

    fundamental_cols = {
        "COL_PE",
        "COL_PB",
        "COL_PS",
        "COL_ROE",
        "COL_GROSS_MARGIN",
        "COL_NET_PROFIT",
        "COL_REVENUE",
        "COL_OCF",
        "COL_TOTAL_ASSETS",
        "COL_REVENUE_YOY",
        "COL_PROFIT_YOY",
        "COL_MARKET_CAP",
        "COL_FLOAT_CAP",
        "COL_DIVIDEND",
    }
    ohlcv_cols = {"COL_AMOUNT", "COL_OPEN", "COL_HIGH", "COL_LOW", "COL_VOLUME"}
    col_values = {n: getattr(schema, n) for n in dir(schema) if n.startswith("COL_")}

    for name, cls in FACTOR_REGISTRY.items():
        src = inspect.getsource(cls.compute)
        referenced = set(re.findall(r"\bCOL_[A-Z_]+\b", src))
        declared_f = set(cls.required_fundamentals)
        declared_o = set(cls.required_ohlcv)
        for const in referenced & fundamental_cols:
            assert (
                col_values[const] in declared_f
            ), f"因子 {name} 引用 {const} 但未在 required_fundamentals 声明"
        for const in referenced & ohlcv_cols:
            assert (
                col_values[const] in declared_o
            ), f"因子 {name} 引用 {const} 但未在 required_ohlcv 声明"

"""YahooProvider 基本面 / 扩展档案纯单测(无网络)。

用假 info dict / 假 income_stmt / balance_sheet DataFrame 测:
- ``_fnum`` / ``_pct`` / ``_float_cap`` 数值归一(比例 ×100、单位约定);
- ``_normalize_fin_history`` 历史财务规范化(roe/gross_margin/同比 + 公告日近似);
- ``get_profile`` 扩展档案映射(经 monkeypatch 的假 yfinance)。
"""

from __future__ import annotations

import math
import sys
import types

import pandas as pd
import pytest

from djinn.data.providers.yahoo import (
    YahooProvider,
    _float_cap,
    _fnum,
    _pct,
)
from djinn.data.schema import (
    COL_ANNOUNCE_DATE,
    COL_GROSS_MARGIN,
    COL_PROFIT_YOY,
    COL_REPORT_DATE,
    COL_REVENUE_YOY,
    COL_ROE,
    Market,
)


def test_fnum_none_and_invalid_are_nan() -> None:
    assert math.isnan(_fnum(None))
    assert math.isnan(_fnum("abc"))
    assert _fnum("3.5") == pytest.approx(3.5)
    assert _fnum(42) == pytest.approx(42.0)


def test_pct_scales_fraction_to_percent() -> None:
    assert _pct(0.15) == pytest.approx(15.0)
    assert _pct(1.0) == pytest.approx(100.0)
    assert math.isnan(_pct(None))
    assert math.isnan(_pct("x"))


def test_float_cap_multiplies_shares_by_price() -> None:
    assert _float_cap(
        {"floatShares": 1_000_000, "currentPrice": 10.0}
    ) == pytest.approx(10_000_000.0)
    assert math.isnan(_float_cap({}))
    assert math.isnan(_float_cap({"floatShares": 100.0}))  # 缺价格


def _fin_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 3 期 income_stmt / balance_sheet(yfinance 列=期末日,行=报表行)。"""
    periods = [
        pd.Timestamp("2023-12-31"),
        pd.Timestamp("2022-12-31"),
        pd.Timestamp("2021-12-31"),
    ]
    ist = pd.DataFrame(
        {
            periods[0]: [1000.0, 200.0, 400.0],  # revenue / net income / gross profit
            periods[1]: [800.0, 160.0, 320.0],
            periods[2]: [700.0, 140.0, 280.0],
        },
        index=["Total Revenue", "Net Income", "Gross Profit"],
    )
    bs = pd.DataFrame(
        {
            periods[0]: [5000.0],
            periods[1]: [4000.0],
            periods[2]: [3500.0],
        },
        index=["Stockholders Equity"],
    )
    return ist, bs


def test_normalize_fin_history_computes_ratios_and_yoy() -> None:
    ist, bs = _fin_frames()
    out = YahooProvider._normalize_fin_history(ist, bs)
    # 结果按日期升序(2021 → 2023)
    assert list(out.index) == [
        pd.Timestamp("2021-12-31"),
        pd.Timestamp("2022-12-31"),
        pd.Timestamp("2023-12-31"),
    ]
    # ROE = 200/5000, 160/4000, 140/3500 → 均 4%
    assert out.loc[pd.Timestamp("2023-12-31"), COL_ROE] == pytest.approx(4.0)
    assert out.loc[pd.Timestamp("2021-12-31"), COL_ROE] == pytest.approx(4.0)
    # 毛利率 = 400/1000 → 40%
    assert out.loc[pd.Timestamp("2023-12-31"), COL_GROSS_MARGIN] == pytest.approx(40.0)
    # 营收同比:2022 = (800/700-1)*100 ≈ 14.29,2023 = 25,2021 首期为 NaN
    assert math.isnan(out.loc[pd.Timestamp("2021-12-31"), COL_REVENUE_YOY])
    assert out.loc[pd.Timestamp("2022-12-31"), COL_REVENUE_YOY] == pytest.approx(
        14.2857, rel=1e-3
    )
    assert out.loc[pd.Timestamp("2023-12-31"), COL_REVENUE_YOY] == pytest.approx(25.0)
    # 净利同比同增速
    assert out.loc[pd.Timestamp("2023-12-31"), COL_PROFIT_YOY] == pytest.approx(25.0)
    # 报告日 / 公告日(report + 45 天)
    assert out.loc[pd.Timestamp("2022-12-31"), COL_REPORT_DATE] == pd.Timestamp(
        "2022-12-31"
    )
    assert out.loc[pd.Timestamp("2022-12-31"), COL_ANNOUNCE_DATE] == pd.Timestamp(
        "2023-02-14"
    )


def test_normalize_fin_history_empty_income() -> None:
    out = YahooProvider._normalize_fin_history(
        pd.DataFrame(index=["Net Income"]), pd.DataFrame(index=["Stockholders Equity"])
    )
    assert len(out) == 0


FAKE_INFO = {
    # 估值扩展
    "forwardPE": 20.5,
    "trailingEps": 6.1,
    "forwardEps": 7.0,
    "pegRatio": 1.8,
    "bookValue": 3.4,
    "enterpriseValue": 2_500_000_000_000.0,
    "enterpriseToEbitda": 12.0,
    "beta": 1.2,
    # 财务健康(债务权益比 yfinance 已是百分数,不乘 100)
    "currentRatio": 1.5,
    "quickRatio": 1.3,
    "debtToEquity": 55.3,
    "totalCash": 1_000_000_000.0,
    "totalDebt": 500_000_000.0,
    "freeCashflow": 80_000_000.0,
    # 行情动量
    "fiftyTwoWeekHigh": 200.0,
    "fiftyTwoWeekLow": 120.0,
    "fiftyDayAverage": 180.0,
    "twoHundredDayAverage": 175.0,
    # 分析师
    "targetMeanPrice": 210.0,
    "targetHighPrice": 240.0,
    "targetLowPrice": 180.0,
    "numberOfAnalystOpinions": 41,
    # 分红(比率 → 百分数)
    "dividendRate": 0.96,
    "trailingAnnualDividendYield": 0.025,
    # 盈利质量(比率 → 百分数)
    "operatingMargins": 0.15,
    "profitMargins": 0.10,
    "returnOnAssets": 0.05,
    # 公司概况
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "recommendationKey": "buy",
    "website": "https://apple.com",
    "longBusinessSummary": "Apple Inc. designs consumer electronics.",
}


class _FakeTicker:
    def __init__(self, symbol: str) -> None:
        self.info = FAKE_INFO


def test_get_profile_maps_info_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(Ticker=_FakeTicker)
    )
    profile = YahooProvider().get_profile("AAPL", Market.US)
    # 数值字段直接透传
    assert profile["forward_pe"] == pytest.approx(20.5)
    assert profile["beta"] == pytest.approx(1.2)
    assert profile["number_of_analysts"] == pytest.approx(41.0)
    # 比率字段 ×100
    assert profile["operating_margin"] == pytest.approx(15.0)
    assert profile["profit_margin"] == pytest.approx(10.0)
    assert profile["return_on_assets"] == pytest.approx(5.0)
    assert profile["dividend_yield"] == pytest.approx(2.5)
    # 债务权益比保持百分数原始值
    assert profile["debt_to_equity"] == pytest.approx(55.3)
    # 字符串字段
    assert profile["sector"] == "Technology"
    assert profile["recommendation"] == "buy"
    assert profile["website"] == "https://apple.com"
    assert profile["summary"] == "Apple Inc. designs consumer electronics."


def test_get_profile_rejects_cn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(Ticker=_FakeTicker)
    )
    with pytest.raises(NotImplementedError):
        YahooProvider().get_profile("600519.SH", Market.CN)


def test_get_profile_missing_fields_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺失的数值字段返回 None(非 NaN),保证 JSON 可序列化。"""

    class _EmptyTicker:
        def __init__(self, symbol: str) -> None:
            self.info: dict = {}

    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(Ticker=_EmptyTicker)
    )
    profile = YahooProvider().get_profile("AAPL", Market.US)
    assert profile["forward_pe"] is None
    assert profile["beta"] is None
    assert profile["operating_margin"] is None
    assert profile["sector"] is None
    assert profile["summary"] is None


def test_info_cache_reuses_ticker_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """D11:详情端点多次 info 需求合并为一次 Ticker.info 网络拉取(TTL 内)。"""
    calls = {"n": 0}
    info_with_price = {**FAKE_INFO, "currentPrice": 150.0}

    class _CountingTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def info(self) -> dict:
            calls["n"] += 1
            return info_with_price

    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(Ticker=_CountingTicker)
    )
    p = YahooProvider()
    p.get_profile("AAPL", Market.US)
    p.get_stock_name("AAPL", Market.US)
    assert p.get_stock_price("AAPL", Market.US) == pytest.approx(150.0)
    assert calls["n"] == 1  # 三次 info 需求仅一次网络拉取

    # 强制过期(timestamp 置 0)→ 再取重新拉取
    p._info_cache["AAPL"] = (0.0, info_with_price)
    p.get_profile("AAPL", Market.US)
    assert calls["n"] == 2

"""基本面数据层测试:asof point-in-time、FundamentalsRouter 路由、缓存命中。

以本地 mock provider 为主(不依赖网络);真实 akshare 拉取标 ``@pytest.mark.network``。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from djinn.data.fundamentals import asof_snapshot
from djinn.data.providers.fundamentals_router import FundamentalsRouter
from djinn.data.schema import (
    COL_ANNOUNCE_DATE,
    COL_MARKET_CAP,
    COL_PB,
    COL_PE,
    COL_REPORT_DATE,
    COL_ROE,
    Market,
)

# ── mock provider ──────────────────────────────────────


class _MockProvider:
    """记录调用次数的假 provider(模拟 AkShare 基本面接口)。"""

    name = "mock"
    market = Market.CN

    def __init__(self) -> None:
        self.snapshot_calls = 0
        self.history_calls = 0
        self._history = self._make_history()

    @staticmethod
    def _make_history() -> pd.DataFrame:
        # 两期财报:一期公告日已过,一期公告日在未来
        rep = pd.to_datetime(["2023-03-31", "2024-03-31"])
        ann = pd.to_datetime(["2023-04-28", "2024-04-30"])
        return pd.DataFrame(
            {
                COL_ROE: [12.0, 15.0],
                COL_REPORT_DATE: rep,
                COL_ANNOUNCE_DATE: ann,
            },
            index=pd.DatetimeIndex(rep, name="date"),
        )

    def supports(self, symbol: str, market: Market | None = None) -> bool:
        return True

    def get_fundamentals(self, symbols: list[str], when: date) -> pd.DataFrame:
        self.snapshot_calls += 1
        return pd.DataFrame(
            {
                COL_MARKET_CAP: [1.0e11] * len(symbols),
                COL_PE: [20.0] * len(symbols),
                COL_PB: [2.5] * len(symbols),
            },
            index=symbols,
        )

    def get_fundamentals_history(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        self.history_calls += 1
        return self._history


# ── asof point-in-time ─────────────────────────────────


def test_asof_snapshot_excludes_future_announcement() -> None:
    hist = _MockProvider._make_history()
    # 2024-04-01:仅能看到 2023 年报(2024 年报公告日 2024-04-30 尚未到)
    snap = asof_snapshot(hist, date(2024, 4, 1))
    assert snap is not None
    assert float(snap[COL_ROE]) == 12.0


def test_asof_snapshot_includes_published_report() -> None:
    hist = _MockProvider._make_history()
    # 2024-05-01:2024 年报已公告,应取到 ROE=15
    snap = asof_snapshot(hist, date(2024, 5, 1))
    assert snap is not None
    assert float(snap[COL_ROE]) == 15.0


def test_asof_snapshot_no_visible_returns_none() -> None:
    hist = _MockProvider._make_history()
    assert asof_snapshot(hist, date(2020, 1, 1)) is None
    assert asof_snapshot(pd.DataFrame(), date(2024, 1, 1)) is None


# ── FundamentalsRouter ─────────────────────────────────


def test_router_snapshot_point_in_time() -> None:
    """截面快照:公告日在未来的财报不应出现在早于公告日的快照里。"""
    provider = _MockProvider()
    router = FundamentalsRouter([provider])  # type: ignore[list-item]
    syms = ["000001.SZ"]
    early = router.get_snapshot(syms, date(2024, 4, 1))
    late = router.get_snapshot(syms, date(2024, 5, 1))
    # 估值字段(快照口径)
    assert early.loc["000001.SZ", COL_MARKET_CAP] > 0
    assert early.loc["000001.SZ", COL_PE] > 0
    # 财务字段(PIT):early 取 2023 年报,late 取 2024 年报
    assert early.loc["000001.SZ", COL_ROE] == 12.0
    assert late.loc["000001.SZ", COL_ROE] == 15.0


def test_router_snapshot_fills_all_value_columns() -> None:
    provider = _MockProvider()
    router = FundamentalsRouter([provider])  # type: ignore[list-item]
    out = router.get_snapshot(["000001.SZ"], date(2024, 5, 1))
    from djinn.data.schema import FUNDAMENTAL_VALUE_COLUMNS

    for col in FUNDAMENTAL_VALUE_COLUMNS:
        assert col in out.columns


# ── 真实 akshare(需网络)─────────────────────────────────


@pytest.mark.network
def test_akshare_fundamentals_snapshot_real() -> None:
    pytest.importorskip("akshare")
    from djinn.data.providers.akshare import AkShareProvider

    p = AkShareProvider()
    if not p.supports("000001.SZ", Market.CN):
        pytest.skip("akshare 不可用")
    snap = p.get_fundamentals(["000001.SZ", "600519.SH"], date.today())
    assert len(snap) >= 1
    assert snap[COL_MARKET_CAP].dropna().gt(0).any()

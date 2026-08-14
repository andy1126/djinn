"""round-trip 交易配对单元测试(B1)。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from djinn.analytics.roundtrip import pair_round_trips
from djinn.analytics.trades import compute_trade_stats
from djinn.engine.events import Fill

_BASE = date(2024, 1, 1)


def _fill(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    day: int,
    *,
    commission: float = 0.0,
) -> Fill:
    return Fill(
        order_id=0,
        timestamp=_BASE + timedelta(days=day),
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        commission=commission,
    )


def test_fifo_pairing_basic() -> None:
    """买 100@10(佣 5)→ 卖 100@12(佣 5)→ 1 回合,pnl=200−10=190。"""
    fills = [
        _fill("A", "buy", 100, 10, 0, commission=5.0),
        _fill("A", "sell", 100, 12, 3, commission=5.0),
    ]
    trips = pair_round_trips(fills)
    assert len(trips) == 1
    t = trips[0]
    assert t.pnl == pytest.approx(200.0 - 10.0)
    assert t.holding_days == 3


def test_fifo_partial_close() -> None:
    """买 100@10 → 买 100@11 → 卖 150@12 → 2 回合,剩余 50 股不生成回合。"""
    fills = [
        _fill("A", "buy", 100, 10, 0),
        _fill("A", "buy", 100, 11, 1),
        _fill("A", "sell", 150, 12, 2),
    ]
    trips = pair_round_trips(fills)
    assert len(trips) == 2
    assert trips[0].qty == 100 and trips[0].open_price == 10.0
    assert trips[1].qty == 50 and trips[1].open_price == 11.0
    assert trips[0].pnl == pytest.approx(200.0)
    assert trips[1].pnl == pytest.approx(50.0)


def test_win_rate_multi_trades_same_symbol() -> None:
    """同标的 3 次完整买卖(2 盈 1 亏)→ win_rate == 2/3。"""
    fills = [
        _fill("A", "buy", 100, 10, 0),
        _fill("A", "sell", 100, 12, 1),  # 盈 +200
        _fill("A", "buy", 100, 10, 2),
        _fill("A", "sell", 100, 9, 3),  # 亏 -100
        _fill("A", "buy", 100, 10, 4),
        _fill("A", "sell", 100, 11, 5),  # 盈 +100
    ]
    stats = compute_trade_stats(fills)
    assert stats.n_round_trips == 3
    assert stats.win_rate == pytest.approx(2 / 3)


def test_avg_holding_days() -> None:
    """两回合 holding_days 3 与 7 → avg == 5.0。"""
    fills = [
        _fill("A", "buy", 100, 10, 0),
        _fill("A", "sell", 100, 10, 3),
        _fill("A", "buy", 100, 10, 4),
        _fill("A", "sell", 100, 10, 11),
    ]
    stats = compute_trade_stats(fills)
    assert stats.avg_holding_days == pytest.approx(5.0)


def test_open_position_excluded() -> None:
    """末段未平仓 → 不计回合。"""
    fills = [
        _fill("A", "buy", 100, 10, 0),
        _fill("A", "sell", 100, 12, 1),
        _fill("A", "buy", 100, 13, 2),  # 未平仓
    ]
    trips = pair_round_trips(fills)
    assert len(trips) == 1
    stats = compute_trade_stats(fills)
    assert stats.n_round_trips == 1


def test_commission_apportioned() -> None:
    """部分平仓时佣金按股数摊派。"""
    fills = [
        _fill("A", "buy", 100, 10, 0, commission=10.0),
        _fill("A", "sell", 50, 12, 1, commission=5.0),
    ]
    trips = pair_round_trips(fills)
    assert len(trips) == 1
    # 开仓佣金摊派 10 * 50/100 = 5;平仓佣金 5 * 50/50 = 5
    # pnl = (12-10)*50 - 5 - 5 = 90
    assert trips[0].pnl == pytest.approx(90.0)

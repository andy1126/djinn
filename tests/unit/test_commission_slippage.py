"""佣金 / 滑点单元测试。"""

from __future__ import annotations

from decimal import Decimal

import pytest

from djinn.data.schema import Bar, Market
from djinn.engine.commission import (
    ChinaCommissionModel,
    ConservativeCommissionModel,
    HKCommissionModel,
    USCommissionModel,
    make_commission,
)
from djinn.engine.slippage import (
    FixedBpsSlippage,
    VolumeShareSlippage,
    ZeroSlippage,
    make_slippage,
)


def _bar(symbol: str = "AAPL", market: Market = Market.US) -> Bar:
    return Bar(
        timestamp=__import__("datetime").date(2024, 1, 2),
        symbol=symbol,
        market=market,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=10000.0,
    )


# ── 佣金 ──────────────────────────────────────────────
def test_commission_min_applied():
    """小单:佣金 = max(amount*rate, min)。"""
    m = ConservativeCommissionModel(rate=0.0003, min_commission=5.0)
    # 100 股 * 10 元 = 1000 元 * 万三 = 0.3 元 < 5 元 → 取 5
    assert m.cost("buy", 10, 100) == Decimal("5.00")


def test_commission_rate_applied():
    """大单:佣金 = amount * rate。"""
    m = ConservativeCommissionModel(rate=0.0003, min_commission=5.0)
    # 10000 股 * 100 = 1,000,000 * 0.0003 = 300
    assert m.cost("buy", 100, 10000) == Decimal("300.00")


def test_china_stamp_duty_on_sell_only():
    """A 股印花税仅卖出收取(过户费双边)。"""
    m = ChinaCommissionModel(
        rate=0.0003, min_commission=5.0, stamp_duty_rate=0.001, transfer_fee_rate=0.0
    )
    # 100000 * 0.0003 = 30(佣金,无过户费)
    buy_cost = m.cost("buy", 100, 1000)
    # 30 + 100000*0.001 = 130(佣金 + 印花税,无过户费)
    sell_cost = m.cost("sell", 100, 1000)
    assert buy_cost == Decimal("30.00")
    assert sell_cost == Decimal("130.00")
    assert sell_cost > buy_cost


def test_us_no_stamp_duty():
    m = USCommissionModel(rate=0.0005, min_commission=1.0)
    assert m.cost("buy", 100, 1000) == m.cost("sell", 100, 1000)


def test_hk_stamp_duty_both_sides():
    """港股印花税双边征收:买入卖出均含印花税,税额相等。"""
    m = HKCommissionModel(rate=0.0005, min_commission=30.0, stamp_duty_rate=0.001)
    # amount = 10000 * 10 = 100000;佣金 = max(50, 30) = 50;印花税双边 = 100
    buy = m.cost("buy", 10, 10000)
    sell = m.cost("sell", 10, 10000)
    assert buy == sell == Decimal("150.00")
    assert buy > 0


def test_cn_stamp_duty_sell_only():
    """A 股印花税仅卖出:买入无印花税。"""
    m = ChinaCommissionModel(
        rate=0.0003, min_commission=5.0, stamp_duty_rate=0.001, transfer_fee_rate=0.0
    )
    buy = m.cost("buy", 100, 1000)  # 佣金 30,无印花税
    sell = m.cost("sell", 100, 1000)  # 佣金 30 + 印花税 100
    assert buy == Decimal("30.00")
    assert sell == Decimal("130.00")


def test_cn_transfer_fee_sh_only():
    """A 股过户费仅沪市(60/68 开头)收取,深市不收(含后缀剥离)。"""
    m = ChinaCommissionModel(
        rate=0.0003, min_commission=5.0, stamp_duty_rate=0.0, transfer_fee_rate=0.00001
    )
    # amount = 100000;佣金 = 30;过户费 = 1(仅沪市)
    assert m.cost("buy", 100, 1000, symbol="600519") == Decimal("31.00")
    assert m.cost("buy", 100, 1000, symbol="000001") == Decimal("30.00")
    assert m.cost("buy", 100, 1000, symbol="300750.SZ") == Decimal("30.00")
    assert m.cost("buy", 100, 1000) == Decimal("30.00")  # 无 symbol 默认不收


def test_make_commission_by_market():
    assert isinstance(make_commission(Market.CN), ChinaCommissionModel)
    assert isinstance(make_commission(Market.US), USCommissionModel)
    assert isinstance(make_commission(Market.HK), HKCommissionModel)


# ── 滑点 ──────────────────────────────────────────────
def test_zero_slippage():
    assert ZeroSlippage().fill_price("buy", 100.0, _bar()) == 100.0


def test_fixed_bps_slippage_direction():
    """买入加价、卖出降价。"""
    s = FixedBpsSlippage(bps=10.0)  # 10 bps = 0.1%
    buy = s.fill_price("buy", 100.0, _bar())
    sell = s.fill_price("sell", 100.0, _bar())
    assert buy > 100.0
    assert sell < 100.0
    assert buy == pytest.approx(100.1, abs=1e-9)  # 100 * (1 + 0.001)
    assert sell == pytest.approx(99.9, abs=1e-9)


def test_make_slippage_factory():
    assert isinstance(make_slippage("zero"), ZeroSlippage)
    assert isinstance(make_slippage("fixed_bps", bps=5), FixedBpsSlippage)
    assert isinstance(make_slippage("volume_share"), VolumeShareSlippage)

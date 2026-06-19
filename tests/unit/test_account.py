"""Account 资金守恒与会计测试(含 hypothesis 属性测试)。"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from djinn.portfolio.account import Account
from djinn.utils.exceptions import AccountError


def test_buy_updates_cash_and_position():
    a = Account(initial_cash=Decimal("100000"))
    a.buy("AAPL", Decimal(100), Decimal("50"), Decimal("5"))
    assert a.cash == Decimal("94995.00")  # 100000 - 5000 - 5
    pos = a.positions["AAPL"]
    assert pos.qty == Decimal("100.0000")
    assert pos.avg_cost == Decimal("50.00")
    assert pos.available == Decimal("100.0000")


def test_money_conservation_buy():
    """买入后:cash + 持仓市值(按成交价) = 初始 - 佣金。"""
    a = Account(initial_cash=Decimal("100000"))
    a.buy("AAPL", Decimal(100), Decimal("50"), Decimal("5"))
    a.check_invariant({"AAPL": 50.0})
    # 按成交价,权益 = 初始 - 佣金
    assert a.equity({"AAPL": 50.0}) == Decimal("99995.00")


def test_sell_realized_pnl():
    a = Account(initial_cash=Decimal("100000"))
    a.buy("AAPL", Decimal(100), Decimal("50"), Decimal("5"))
    a.sell("AAPL", Decimal(60), Decimal("60"), Decimal("5"))
    pos = a.positions["AAPL"]
    assert pos.qty == Decimal("40.0000")
    # 已实现盈亏 = (60 - 50) * 60 = 600(不含费用摊销)
    assert pos.realized_pnl == Decimal("600.00")
    a.check_invariant({"AAPL": 60.0})


def test_sell_more_than_available_fails():
    a = Account(initial_cash=Decimal("100000"))
    a.buy("AAPL", Decimal(100), Decimal("50"), Decimal("5"))
    with pytest.raises(AccountError):
        a.sell("AAPL", Decimal(101), Decimal("60"), Decimal("5"))


def test_buy_insufficient_cash_fails():
    a = Account(initial_cash=Decimal("1000"))
    with pytest.raises(AccountError):
        a.buy("AAPL", Decimal(100), Decimal("50"), Decimal("5"))


def test_t_plus_1_freeze_and_unfreeze():
    a = Account(initial_cash=Decimal("100000"), t_plus_1=True)
    a.buy("XYZ", Decimal(100), Decimal("10"), Decimal("5"))
    pos = a.positions["XYZ"]
    assert pos.available == Decimal("0.0000")
    assert pos.frozen == Decimal("100.0000")
    # T+1:当日不可卖
    with pytest.raises(AccountError):
        a.sell("XYZ", Decimal(1), Decimal("10"), Decimal("1"))
    # 次日解冻
    a.unfreeze_all()
    assert pos.available == Decimal("100.0000")
    assert pos.frozen == Decimal("0")
    a.sell("XYZ", Decimal(50), Decimal("10"), Decimal("5"))
    assert pos.qty == Decimal("50.0000")


def test_avg_cost_weighted():
    """分批买入:均价加权。"""
    a = Account(initial_cash=Decimal("100000"))
    a.buy("AAPL", Decimal(100), Decimal("50"), Decimal("0"))  # 100 @ 50
    a.buy("AAPL", Decimal(100), Decimal("60"), Decimal("0"))  # 100 @ 60
    pos = a.positions["AAPL"]
    assert pos.qty == Decimal("200.0000")
    assert pos.avg_cost == Decimal("55.00")  # (5000+6000)/200


# ── 属性测试:资金守恒 ──────────────────────────────────
@given(
    n_buys=st.integers(min_value=1, max_value=10),
    price_mult=st.floats(
        min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=30, deadline=None)
def test_money_conservation_property(n_buys: int, price_mult: float):
    """任意买入序列后,cash + 持仓市值 == equity(守恒)。"""
    a = Account(initial_cash=Decimal("1000000"))
    price = Decimal("100")
    qty_per = Decimal("100")
    comm = Decimal("5")
    for _ in range(n_buys):
        try:
            a.buy("AAPL", qty_per, price, comm)
        except AccountError:
            break
    # 在任意价格下守恒
    for p in [50.0, 100.0, 150.0, float(price) * price_mult]:
        a.check_invariant({"AAPL": p})


@given(
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=20, deadline=None)
def test_round_trip_no_free_money(seed: int):
    """买后卖(往返)后:现金变化 = 卖出净额 - 买入成本,无凭空增减。"""
    import random

    rng = random.Random(seed)
    a = Account(initial_cash=Decimal("100000"))
    buy_price = Decimal(str(rng.uniform(50, 100)))
    qty = Decimal(rng.randint(10, 500))
    buy_comm = Decimal("5")
    try:
        a.buy("X", qty, buy_price, buy_comm)
    except AccountError:
        return
    sell_price = Decimal(str(rng.uniform(50, 100)))
    sell_comm = Decimal("5")
    a.sell("X", qty, sell_price, sell_comm)
    # 现金净变化 = 卖出净 - 买入总(允许累计舍入误差 ≤ 1 分)
    expected_delta = (qty * sell_price - sell_comm) - (qty * buy_price + buy_comm)
    actual_delta = a.cash - Decimal("100000")
    assert abs(actual_delta - expected_delta) <= Decimal("0.02")

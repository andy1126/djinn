"""D2:PortfolioView weight/equity 缓存一致性测试。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from djinn.portfolio.account import Account
from djinn.strategy.base import PortfolioView


def _view() -> PortfolioView:
    acct = Account(initial_cash=Decimal("100000"))
    acct.buy("A", Decimal("100"), Decimal("50"), Decimal("0"))
    acct.buy("B", Decimal("200"), Decimal("25"), Decimal("0"))
    return PortfolioView(acct, {"A": 50.0, "B": 25.0}, date(2024, 1, 1))


def test_weights_consistency() -> None:
    """weights() 与逐 symbol weight() 结果一致。"""
    pv = _view()
    w = pv.weights()
    for s in ("A", "B"):
        assert w[s] == pytest.approx(pv.weight(s))
    # 无持仓标的:weight 返回 0,weights 不含
    assert pv.weight("C") == 0.0
    assert "C" not in w


def test_equity_cached() -> None:
    """equity 惰性缓存:多次访问返回同值,且已缓存。"""
    pv = _view()
    e1 = pv.equity
    e2 = pv.equity
    assert e1 == pytest.approx(e2)
    assert pv._equity_cache is not None
    assert e1 == pytest.approx(100000.0)  # 现金 92500 + 5000 + 2500


def test_weights_empty_equity() -> None:
    """无持仓:weights 返回空,equity = 现金。"""
    acct = Account(initial_cash=Decimal("50000"))
    pv = PortfolioView(acct, {}, date(2024, 1, 1))
    assert pv.weights() == {}
    assert pv.equity == pytest.approx(50000.0)

"""交易约束单元测试:涨跌停 / 停牌 / 最小手 / T+1。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from djinn.data.schema import Bar, Market
from djinn.engine.constraints import TradeConstraints, check_constraints, limit_prices


def _bar(
    symbol: str,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: float = 10000,
    suspended: bool = False,
    market: Market = Market.CN,
) -> Bar:
    return Bar(
        timestamp=date(2024, 1, 5),
        symbol=symbol,
        market=market,
        open=open_ if open_ is not None else close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close,
        volume=volume,
        is_suspended=suspended,
    )


def test_suspension_blocks_order():
    con = TradeConstraints(market=Market.CN, enforce_suspension=True)
    bar = _bar("600000", 10.0, suspended=True)
    r = check_constraints("buy", Decimal(100), bar, 9.0, Decimal("100000"), 10.0, con)
    assert not r.ok
    assert "停牌" in r.reason


def test_lot_size_rounding_cn():
    """A 股 100 股最小手:155 股向下取整到 100。"""
    con = TradeConstraints(market=Market.CN, enforce_lot=True)
    bar = _bar("600000", 10.0)
    r = check_constraints("buy", Decimal(155), bar, 9.0, Decimal("100000"), 10.0, con)
    assert r.ok
    assert r.adjusted_qty == Decimal("100")


def test_lot_size_insufficient_cn():
    """不足 100 股:拒单。"""
    con = TradeConstraints(market=Market.CN, enforce_lot=True)
    bar = _bar("600000", 10.0)
    r = check_constraints("buy", Decimal(50), bar, 9.0, Decimal("100000"), 10.0, con)
    assert not r.ok
    assert "最小手" in r.reason


def test_us_no_lot_rounding():
    """美股 lot=1:不取整。"""
    con = TradeConstraints(market=Market.US, enforce_lot=True)
    bar = _bar("AAPL", 100.0, market=Market.US)
    r = check_constraints(
        "buy", Decimal("47.5"), bar, 99.0, Decimal("100000"), 100.0, con
    )
    assert r.ok
    assert r.adjusted_qty == Decimal("47.5000")


def test_sell_odd_lot_allowed():
    """A 股卖出豁免整手取整:零股(37 股)可卖。"""
    con = TradeConstraints(market=Market.CN, enforce_lot=True)
    bar = _bar("600000", 10.0)
    r = check_constraints("sell", Decimal(37), bar, 9.0, Decimal("100000"), 10.0, con)
    assert r.ok
    assert r.adjusted_qty == Decimal("37")


def test_sell_clamped_to_available():
    """卖出夹到可用股数(不取整到 100)。"""
    con = TradeConstraints(market=Market.CN, enforce_lot=True)
    bar = _bar("600000", 10.0)
    r = check_constraints(
        "sell",
        Decimal(150),
        bar,
        9.0,
        Decimal("100000"),
        10.0,
        con,
        available_qty=Decimal("120"),
    )
    assert r.ok
    assert r.adjusted_qty == Decimal("120")


def test_buy_still_floored():
    """买入仍整手取整(回归):155 → 100。"""
    con = TradeConstraints(market=Market.CN, enforce_lot=True)
    bar = _bar("600000", 10.0)
    r = check_constraints("buy", Decimal(155), bar, 9.0, Decimal("100000"), 10.0, con)
    assert r.ok
    assert r.adjusted_qty == Decimal("100")


def test_price_limit_sealed_up():
    """涨停封板(全天 high=low=涨停价):买单拒单。"""
    con = TradeConstraints(market=Market.CN, enforce_price_limit=True)
    prev = 10.0
    up, down = limit_prices(prev, "600000", con)
    # 全天封涨停:high=low=close=up
    bar = _bar("600000", up, high=up, low=up, open_=up)
    r = check_constraints("buy", Decimal(100), bar, prev, Decimal("100000"), up, con)
    assert not r.ok
    assert "涨停" in r.reason


def test_price_limit_sealed_down():
    """跌停封板:卖单拒单。"""
    con = TradeConstraints(market=Market.CN, enforce_price_limit=True)
    prev = 10.0
    up, down = limit_prices(prev, "600000", con)
    bar = _bar("600000", down, high=down, low=down, open_=down)
    r = check_constraints("sell", Decimal(100), bar, prev, Decimal("100000"), down, con)
    assert not r.ok
    assert "跌停" in r.reason


def test_price_limit_not_sealed_allows_trade():
    """未封板(日内有波动):订单放行。"""
    con = TradeConstraints(market=Market.CN, enforce_price_limit=True)
    prev = 10.0
    up, down = limit_prices(prev, "600000", con)
    # 当日触及涨停但未封板(low < up)
    bar = _bar("600000", up, high=up, low=prev, open_=prev)
    r = check_constraints("buy", Decimal(100), bar, prev, Decimal("100000"), prev, con)
    assert r.ok


def test_us_no_price_limit():
    """美股无涨跌停。"""
    con = TradeConstraints(market=Market.US, enforce_price_limit=True)
    lim = limit_prices(100.0, "AAPL", con)
    assert lim is None


def test_insufficient_cash_shrinks_cn():
    """资金不足:A 股按可用资金缩减到最小手整数倍。"""
    con = TradeConstraints(market=Market.CN, enforce_lot=True)
    bar = _bar("600000", 100.0)
    # 现金 5000,价格 100 → 最多 50 股 < 100 最小手 → 拒单
    r = check_constraints("buy", Decimal(100), bar, 99.0, Decimal("5000"), 100.0, con)
    assert not r.ok
    # 现金 15000 → 最多 150 股 → 取整 100
    r2 = check_constraints("buy", Decimal(200), bar, 99.0, Decimal("15000"), 100.0, con)
    assert r2.ok
    assert r2.adjusted_qty == Decimal("100")


def test_volume_cap_shrinks_order():
    """成交量上限(A11):买 50 万股、volume=1e6、max_share=0.1 → 缩减到 10 万。"""
    con = TradeConstraints(market=Market.US, enforce_lot=True, max_volume_share=0.1)
    bar = _bar("AAPL", 100.0, volume=1_000_000.0)
    r = check_constraints(
        "buy", Decimal(500_000), bar, 99.0, Decimal("100000000"), 100.0, con
    )
    assert r.ok
    assert r.adjusted_qty == Decimal("100000")


def test_volume_cap_zero_disabled():
    """默认 max_volume_share=0 不生效(回归)。"""
    con = TradeConstraints(market=Market.US, enforce_lot=True)
    bar = _bar("AAPL", 100.0, volume=1_000_000.0)
    r = check_constraints(
        "buy", Decimal(500_000), bar, 99.0, Decimal("100000000"), 100.0, con
    )
    assert r.ok
    assert r.adjusted_qty == Decimal("500000")

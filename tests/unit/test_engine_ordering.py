"""引擎撮合顺序 / 成交参考价 / 限价单 / 停牌续挂测试(A3/A4/A5/A6)。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from djinn.data.market_data import MarketData
from djinn.data.schema import COL_DIVIDEND, Adjust, Bar, Market
from djinn.engine.broker import Broker
from djinn.engine.commission import USCommissionModel
from djinn.engine.constraints import TradeConstraints
from djinn.engine.event_engine import EngineConfig, EventDrivenEngine
from djinn.engine.events import Fill, Order, Rejection
from djinn.engine.slippage import ZeroSlippage
from djinn.portfolio.account import Account
from djinn.portfolio.allocation import MinVarianceWeight
from djinn.portfolio.rebalance import RebalanceConfig, Rebalancer
from djinn.strategy.base import SCOPE_PORTFOLIO, Strategy


def _bar(
    *,
    open_: float = 100.0,
    close: float = 100.0,
    volume: float = 10000.0,
    amount: float = 0.0,
    suspended: bool = False,
) -> Bar:
    return Bar(
        timestamp=date(2024, 1, 5),
        symbol="S",
        market=Market.US,
        open=open_,
        high=max(open_, close) * 1.01,
        low=min(open_, close) * 0.99,
        close=close,
        volume=volume,
        amount=amount,
        is_suspended=suspended,
    )


def _broker(acct: Account, fill_ref: str = "open") -> Broker:
    return Broker(
        account=acct,
        commission=USCommissionModel(rate=0.0, min_commission=0.0),
        slippage=ZeroSlippage(),
        constraints=TradeConstraints(market=Market.US),
        fill_ref=fill_ref,
    )


def _md(symbol: str, prices: list[float]) -> MarketData:
    idx = pd.bdate_range("2024-01-01", periods=len(prices))
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1.0e6] * len(prices),
        },
        index=idx,
    )
    return MarketData(symbol=symbol, market=Market.US, df=df, adjust=Adjust.BACKWARD)


def _engine(initial_cash: float) -> EngineConfig:
    return EngineConfig(
        initial_cash=initial_cash,
        commission=USCommissionModel(rate=0.0, min_commission=0.0),
        slippage=ZeroSlippage(),
    )


# ── A5:fill_ref ────────────────────────────────────────


def test_fill_ref_close() -> None:
    """fill_ref=close:成交价 = 当日 close(而非 open)。"""
    acct = Account(initial_cash=Decimal("100000"))
    broker = _broker(acct, fill_ref="close")
    order = Order(id=1, symbol="S", side="buy", created_ts=date(2024, 1, 5), qty=100)
    result = broker.execute(order, _bar(open_=110.0, close=120.0), 100.0, 100000.0)
    assert isinstance(result, Fill)
    assert result.price == 120.0


def test_fill_ref_vwap() -> None:
    """fill_ref=vwap:成交价 = amount / volume。"""
    acct = Account(initial_cash=Decimal("100000"))
    broker = _broker(acct, fill_ref="vwap")
    order = Order(id=1, symbol="S", side="buy", created_ts=date(2024, 1, 5), qty=100)
    result = broker.execute(
        order, _bar(open_=110.0, close=120.0, amount=2_000_000.0), 100.0, 100000.0
    )
    assert isinstance(result, Fill)
    assert result.price == pytest.approx(200.0)  # 2_000_000 / 10_000


# ── A5:limit_price ─────────────────────────────────────


def test_limit_buy_not_filled() -> None:
    """买限价低于开盘价 → 拒单(续挂,retryable=True)。"""
    acct = Account(initial_cash=Decimal("100000"))
    broker = _broker(acct)
    order = Order(
        id=1,
        symbol="S",
        side="buy",
        created_ts=date(2024, 1, 5),
        qty=100,
        limit_price=100.0,
    )
    result = broker.execute(order, _bar(open_=110.0, close=110.0), 100.0, 100000.0)
    assert isinstance(result, Rejection)
    assert result.retryable is True
    assert "限价" in result.reason


def test_limit_sell_not_filled() -> None:
    """卖限价高于开盘价 → 拒单(续挂)。"""
    acct = Account(initial_cash=Decimal("100000"))
    broker = _broker(acct)
    order = Order(
        id=1,
        symbol="S",
        side="sell",
        created_ts=date(2024, 1, 5),
        qty=100,
        limit_price=120.0,
    )
    result = broker.execute(order, _bar(open_=110.0, close=110.0), 100.0, 100000.0)
    assert isinstance(result, Rejection)
    assert result.retryable is True


# ── A5:限价单引擎级 e2e(次日落入限价成交)─────────────────


def _md_with_opens(symbol: str, opens: list[float]) -> MarketData:
    """构造指定逐日开盘价的美股 MarketData(其余列跟随)。"""
    idx = pd.bdate_range("2024-01-01", periods=len(opens))
    return MarketData(
        symbol=symbol,
        market=Market.US,
        df=pd.DataFrame(
            {
                "open": opens,
                "high": [o + 1.0 for o in opens],
                "low": [o - 1.0 for o in opens],
                "close": opens,
                "volume": [1.0e6] * len(opens),
            },
            index=idx,
        ),
        adjust=Adjust.BACKWARD,
    )


def test_limit_buy_fills_next_day() -> None:
    """A5:限价买单首日未达 → 续挂,次日价格落入限价 → 以开盘价成交。"""
    a_md = _md_with_opens("A", [100.0, 95.0, 90.0, 90.0])  # day2 open 落入限价 90

    class _Strat(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.day = 0

        def on_bar(self, ctx) -> None:  # type: ignore[override]
            self.day += 1
            if self.day == 1:
                ctx.buy("A", size=100, limit=90.0)

    res = EventDrivenEngine(_engine(100000.0)).run(_Strat(), {"A": a_md})
    # 首日(open 95 > 90)未达 → 续挂;次日价格落入限价 → 以开盘价 90 成交
    # (而非按首日 95 成交,证明订单一直挂到限价满足)
    buys = [f for f in res.trades if f.side == "buy"]
    assert len(buys) == 1
    assert buys[0].price == pytest.approx(90.0)
    assert buys[0].timestamp == date(2024, 1, 3)  # day2 成交(day1 未达续挂)
    assert res.positions_curve["A"].iloc[-1] > 0


def test_limit_sell_fills_next_day() -> None:
    """A5:限价卖单首日未达 → 续挂,次日价格升到限价 → 成交。"""
    a_md = _md_with_opens("A", [100.0, 105.0, 110.0, 110.0])  # day2 open 升到限价 110

    class _Strat(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.day = 0

        def on_bar(self, ctx) -> None:  # type: ignore[override]
            self.day += 1
            if self.day == 1:
                ctx.buy("A", size=100)  # 先建仓
            elif self.day == 2:
                ctx.sell("A", size=100, limit=110.0)

    res = EventDrivenEngine(_engine(100000.0)).run(_Strat(), {"A": a_md})
    sells = [f for f in res.trades if f.side == "sell"]
    assert len(sells) == 1
    assert sells[0].price == pytest.approx(110.0)
    assert res.positions_curve["A"].iloc[-1] == 0.0  # 全部卖出


# ── A6:停牌续挂结构化 ───────────────────────────────────


def test_suspension_retryable_flag() -> None:
    """停牌拒单 retryable=True;资金不足拒单 retryable=False。"""
    acct = Account(initial_cash=Decimal("100000"))
    broker = _broker(acct)
    order = Order(id=1, symbol="S", side="buy", created_ts=date(2024, 1, 5), qty=100)
    susp = broker.execute(order, _bar(suspended=True), 100.0, 100000.0)
    assert isinstance(susp, Rejection)
    assert susp.retryable is True

    # 非停牌拒单(CN 市场不足最小手):retryable=False
    cn_broker = Broker(
        account=Account(initial_cash=Decimal("100000")),
        commission=USCommissionModel(rate=0.0, min_commission=0.0),
        slippage=ZeroSlippage(),
        constraints=TradeConstraints(market=Market.CN),
        fill_ref="open",
    )
    order50 = Order(
        id=2, symbol="600000", side="buy", created_ts=date(2024, 1, 5), qty=50
    )
    result = cn_broker.execute(order50, _bar(open_=10.0, close=10.0), 10.0, 100000.0)
    assert isinstance(result, Rejection)
    assert result.retryable is False


# ── A3:先卖后买 ────────────────────────────────────────


def test_sell_before_buy_same_batch() -> None:
    """同一批订单先卖后买:卖单回笼资金后买单可全额成交(不受列表顺序影响)。"""

    class _Strat(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.day = 0

        def on_bar(self, ctx) -> None:  # type: ignore[override]
            self.day += 1
            if self.day == 1:
                ctx.buy("A", size=100)
            elif self.day == 2:
                ctx.buy("B", size=100)  # 先买(FIFO 会因资金不足缩减)
                ctx.sell("A", size=100)  # 后卖(回笼资金)

    result = EventDrivenEngine(_engine(10000.0)).run(
        _Strat(), {"A": _md("A", [50.0] * 4), "B": _md("B", [80.0] * 4)}
    )
    b_buys = [f for f in result.trades if f.symbol == "B" and f.side == "buy"]
    assert b_buys, "B 应有买单成交"
    # 两阶段:先卖 A(回笼 5000)→ 现金 10000,足以全额买 100 股 B(8000)
    assert b_buys[0].qty == pytest.approx(100.0)


# ── A3:撮合顺序确定性 ──────────────────────────────────


def test_ordering_deterministic() -> None:
    """A3:打乱 symbols 字典顺序跑两次 → fills 序列逐笔一致(不依赖数据顺序)。"""
    pa = [50.0 + i for i in range(20)]
    pb = [80.0 - i for i in range(20)]

    def run(order: list[str]) -> list[tuple[str, str, float, float]]:
        data = {s: _md(s, pa if s == "A" else pb) for s in order}

        class _Strat(Strategy):
            def __init__(self) -> None:
                super().__init__()
                self.day = 0

            def on_bar(self, ctx) -> None:  # type: ignore[override]
                self.day += 1
                if self.day == 1:
                    ctx.buy("A", size=100)
                    ctx.buy("B", size=100)

        res = EventDrivenEngine(_engine(100000.0)).run(_Strat(), data)
        return [(f.symbol, f.side, f.qty, f.price) for f in res.trades]

    assert run(["A", "B"]) == run(["B", "A"])


# ── A4:target_percent 用开盘价口径 ──────────────────────


def test_target_percent_uses_open_price() -> None:
    """跳空(open != close)时,再平衡 target_percent 按开盘价口径计算股数。"""

    idx = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        {
            "open": [110.0] * 3,
            "high": [120.0] * 3,
            "low": [110.0] * 3,
            "close": [120.0] * 3,
            "volume": [1.0e6] * 3,
        },
        index=idx,
    )
    a_md = MarketData(symbol="A", market=Market.US, df=df, adjust=Adjust.BACKWARD)

    class _Strat(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.day = 0

        def on_bar(self, ctx) -> None:  # type: ignore[override]
            self.day += 1
            if self.day == 1:
                ctx.buy("A", size=100)
            elif self.day == 2:
                ctx.order_target_percent("A", 0.5)

    result = EventDrivenEngine(_engine(100000.0)).run(_Strat(), {"A": a_md})
    buys = [f for f in result.trades if f.side == "buy"]
    assert len(buys) == 2
    # 第二笔:equity_open = 100000,cur_mv = 100*110,delta = 39000 → qty = 39000/110
    assert buys[1].qty == pytest.approx(39000.0 / 110.0, rel=1e-6)


def test_engine_rejects_cov_allocation():
    """A8:引擎再平衡路径 + 需 cov 的分配器 → 启动抛 ValueError。"""

    class _Hold(Strategy):
        def on_bar(self, ctx) -> None:  # type: ignore[override]
            return

    cfg = EngineConfig(
        initial_cash=100000.0,
        commission=USCommissionModel(rate=0.0, min_commission=0.0),
        slippage=ZeroSlippage(),
        allocation=MinVarianceWeight(),
        rebalance=Rebalancer(RebalanceConfig(period="daily")),
    )
    with pytest.raises(ValueError, match="FactorPortfolioStrategy"):
        EventDrivenEngine(cfg).run(_Hold(), {"A": _md("A", [50.0] * 3)})


def test_prices_curve_recorded_and_serialized():
    """价格走势数据:引擎记录每日收盘价,报告序列化含 prices(供前端买卖点图)。"""

    class _Hold(Strategy):
        def on_bar(self, ctx) -> None:  # type: ignore[override]
            return

    prices = [50.0 + i for i in range(40)]
    result = EventDrivenEngine(_engine(100000.0)).run(_Hold(), {"S": _md("S", prices)})
    # index=交易日、columns=symbol、末值=最后收盘价
    assert result.prices_curve.shape == (40, 1)
    assert list(result.prices_curve.columns) == ["S"]
    assert result.prices_curve["S"].iloc[-1] == pytest.approx(89.0)

    from djinn.analytics.report import build_report
    from djinn.api.report_store import serialize_report

    payload = serialize_report(build_report(result))
    assert payload["prices"]["columns"] == ["S"]
    assert len(payload["prices"]["index"]) == 40


# ── A9:退市强制平仓 + A10:adjust=none 公司行为 ────────────


def test_delist_forced_liquidation() -> None:
    """A9:union 日历下持仓标的超 grace 天无行情 → 按最近价强制平仓(tag=delist)。"""
    n_a = 10
    n_b = len(pd.bdate_range("2024-01-01", "2024-04-30"))
    data = {"A": _md("A", [100.0] * n_a), "B": _md("B", [100.0] * n_b)}

    class _BuyBoth(Strategy):
        scope = SCOPE_PORTFOLIO

        def __init__(self) -> None:
            super().__init__()
            self._placed = False

        def on_bar(self, ctx) -> None:  # type: ignore[override]
            if not self._placed:
                self._placed = True
                ctx.order_target_percent("A", 0.5)
                ctx.order_target_percent("B", 0.5)

    cfg = EngineConfig(
        initial_cash=100000.0,
        commission=USCommissionModel(rate=0.0, min_commission=0.0),
        slippage=ZeroSlippage(),
        calendar="union",
    )
    res = EventDrivenEngine(cfg).run(_BuyBoth(), data)
    delist = [f for f in res.trades if f.tag == "delist"]
    assert delist, "应有 delist 强平单"
    assert all(f.symbol == "A" for f in delist)
    # A 曾建仓、强平后归零;B 始终有行情不受影响
    assert res.positions_curve["A"].max() > 0
    assert res.positions_curve["A"].iloc[-1] == 0.0
    assert res.positions_curve["B"].iloc[-1] > 0
    # 强平回收现金 ≈ 最后价 × 强平股数(0 佣金);强平日现金较前一日回笼
    sold = delist[0]
    assert sold.price == pytest.approx(100.0)
    dday = res.cash_curve.index.get_loc(pd.Timestamp(sold.timestamp))
    assert res.cash_curve.iloc[dday] > res.cash_curve.iloc[dday - 1]
    assert res.cash_curve.iloc[dday] == pytest.approx(50000.0)


def test_delist_pending_purged() -> None:
    """A9:被强平标的的未成交挂单在强平日清除(复牌后不再成交)。"""
    idx_a1 = pd.bdate_range("2024-01-01", periods=10)  # Jan 1-12 正常行情
    idx_a2 = pd.bdate_range("2024-03-18", periods=30)  # 复牌段(挂单未清会在这里成交)
    idx_all = idx_a1.append(idx_a2)
    df_a = pd.DataFrame(
        {
            "open": [100.0] * len(idx_all),
            "high": [101.0] * len(idx_all),
            "low": [99.0] * len(idx_all),
            "close": [100.0] * len(idx_all),
            "volume": [1.0e6] * len(idx_all),
        },
        index=idx_all,
    )
    n_b = len(pd.bdate_range("2024-01-01", "2024-04-30"))
    data = {
        "A": MarketData(symbol="A", market=Market.US, df=df_a, adjust=Adjust.BACKWARD),
        "B": _md("B", [100.0] * n_b),
    }

    class _HoldThenReorder(Strategy):
        scope = SCOPE_PORTFOLIO

        def __init__(self) -> None:
            super().__init__()
            self._bought = False
            self._reordered = False

        def on_bar(self, ctx) -> None:  # type: ignore[override]
            if not self._bought:
                self._bought = True
                ctx.order_target_percent("A", 0.5)  # 建仓
                return
            if not self._reordered and "A" not in ctx.data:
                # A 停牌首日:追加买入单 → 该单无行情挂起,等待复牌成交
                self._reordered = True
                ctx.order_target_percent("A", 0.9)

    cfg = EngineConfig(
        initial_cash=100000.0,
        commission=USCommissionModel(rate=0.0, min_commission=0.0),
        slippage=ZeroSlippage(),
        calendar="union",
    )
    res = EventDrivenEngine(cfg).run(_HoldThenReorder(), data)
    # A 曾被建仓并超期强平
    assert [f for f in res.trades if f.tag == "delist" and f.symbol == "A"]
    assert res.positions_curve["A"].max() > 0
    # 复牌段 A 恢复行情却未再买入 → 强平日已清掉挂单
    assert res.positions_curve["A"].iloc[-1] == 0.0


def test_dividend_cash_when_unadjusted() -> None:
    """A10:adjust=none 且 process_corporate_actions 时,分红日现金增加、无假回撤。"""
    idx = pd.bdate_range("2024-01-01", periods=10)
    n = len(idx)
    dividend = [0.0] * n
    dividend[4] = 0.5  # 第 5 个交易日每股派 0.5
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1.0e6] * n,
            COL_DIVIDEND: dividend,
        },
        index=idx,
    )
    data = {"A": MarketData(symbol="A", market=Market.US, df=df, adjust=Adjust.NONE)}

    class _Buy1000(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self._bought = False

        def on_bar(self, ctx) -> None:  # type: ignore[override]
            if not self._bought:
                self._bought = True
                ctx.buy("A", size=1000)

    cfg = EngineConfig(
        initial_cash=100000.0,
        commission=USCommissionModel(rate=0.0, min_commission=0.0),
        slippage=ZeroSlippage(),
        process_corporate_actions=True,
    )
    res = EventDrivenEngine(cfg).run(_Buy1000(), data)
    # 买入 1000 股 × 100 = 10 万 → 现金 0;分红日 0.5×1000 = 500 入账
    assert res.cash_curve.loc[idx[4]] == pytest.approx(500.0)
    # 除息日 equity 不跌反升(无假回撤)
    assert res.equity_curve.loc[idx[4]] > res.equity_curve.loc[idx[3]]

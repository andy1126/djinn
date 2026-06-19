"""集成测试:已知结果的回测场景。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from djinn.data import Adjust, CSVProvider, Market
from djinn.engine import EngineConfig, EventDrivenEngine
from djinn.engine.commission import USCommissionModel
from djinn.engine.slippage import ZeroSlippage
from djinn.portfolio import EqualWeight, RebalanceConfig, Rebalancer
from djinn.strategy import DCA, MACrossover
from djinn.strategy.base import Context, Strategy


# ── buy-and-hold ──────────────────────────────────────
def test_buy_and_hold_matches_asset_return(synthetic_aapl: Path):
    """买入持有:策略净值 ≈ 标的后复权收益(误差仅来自佣金)。

    构造一个金叉后一直持有的场景(MACrossover 在单调上涨数据上)。
    """
    prov = CSVProvider(synthetic_aapl.parent, default_market=Market.US)
    md = prov.get_ohlcv("AAPL", date(2024, 1, 2), date(2024, 2, 27), Adjust.BACKWARD)
    strat = MACrossover(fast=5, slow=10)  # 单调上涨,早期金叉后持有
    cfg = EngineConfig(
        initial_cash=100000, commission=USCommissionModel(), slippage=ZeroSlippage()
    )
    result = EventDrivenEngine(cfg).run(strat, {"AAPL": md})
    assert result.n_trades >= 1
    # 标的收益
    asset_close = md.df["close"]
    asset_ret = asset_close.iloc[-1] / asset_close.iloc[0]
    # 策略应持有大部分仓位,净值涨幅接近标的涨幅(建仓后)
    strat_ret = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0]
    # 建仓在金叉后(非首日),策略收益应 < 全程标的收益但同向为正
    assert strat_ret > 1.0
    assert strat_ret <= asset_ret + 0.01  # 不会超过(含佣金且建仓晚)


# ── DCA 份额累计 ──────────────────────────────────────
def test_dca_accumulates_shares(make_csv, tmp_csv_dir: Path):
    """定投:每期等额买入,份额累计正确。"""
    make_csv("AAPL", periods=60, drift=0.0, vol=0.0, seed=1)  # 平稳价
    prov = CSVProvider(tmp_csv_dir, default_market=Market.US)
    md = prov.get_ohlcv("AAPL", date(2024, 1, 2), date(2024, 3, 27), Adjust.BACKWARD)
    # 价格平稳(~100),定投 frequency=10, amount=1000 → 每期约 10 股
    strat = DCA(symbol="AAPL", frequency=10, amount=1000)
    cfg = EngineConfig(
        initial_cash=100000,
        commission=USCommissionModel(rate=0.0, min_commission=0.0),
        slippage=ZeroSlippage(),
    )
    result = EventDrivenEngine(cfg).run(strat, {"AAPL": md})
    # frequency=10:bar 10,20,30,40,50 触发(共 5 次),每次 ~1000/100=10 股
    n_periods = (len(md) - 1) // 10
    expected_shares = n_periods * 10
    final_pos = result.positions_curve["AAPL"].iloc[-1]
    assert final_pos == pytest.approx(expected_shares, abs=2)


# ── MACrossover 金叉死叉精确触发 ───────────────────────
def test_macrossover_golden_cross_trades(make_csv, tmp_csv_dir: Path):
    """合成金叉/死叉数据:MACrossover 在交叉处精确建仓/平仓。"""
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(
        np.concatenate(
            [
                np.linspace(100, 90, 20),
                np.linspace(90, 110, 20),
                np.linspace(110, 100, 20),
            ]
        ),
        index=idx,
    )
    df = pd.DataFrame(
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 10000,
        }
    )
    tmp_csv_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_csv_dir / "X.csv", index=False)
    prov = CSVProvider(tmp_csv_dir, default_market=Market.US)
    md = prov.get_ohlcv("X", date(2024, 1, 2), date(2024, 3, 27), Adjust.BACKWARD)
    strat = MACrossover(fast=5, slow=15)
    cfg = EngineConfig(
        initial_cash=100000, commission=USCommissionModel(), slippage=ZeroSlippage()
    )
    result = EventDrivenEngine(cfg).run(strat, {"X": md})
    # 应至少有 1 笔买入(金叉)和 1 笔卖出(死叉)
    buys = [f for f in result.trades if f.side == "buy"]
    sells = [f for f in result.trades if f.side == "sell"]
    assert len(buys) >= 1
    assert len(sells) >= 1
    # 买在金叉后(t+1 开盘),卖在死叉后
    assert sells[0].timestamp > buys[0].timestamp


# ── 等权组合季度再平衡 ─────────────────────────────────
def test_equal_weight_quarterly_rebalance(make_csv, tmp_csv_dir: Path):
    """等权组合 + 季度再平衡:再平衡后各成分权重回到目标 ±阈值内。"""
    for sym, seed in [("A", 1), ("B", 2), ("C", 3)]:
        make_csv(sym, periods=252, drift=0.0005, vol=0.015, seed=seed)
    prov = CSVProvider(tmp_csv_dir, default_market=Market.US)
    symbols = ["A", "B", "C"]
    data = {
        s: prov.get_ohlcv(s, date(2024, 1, 2), date(2024, 12, 30), Adjust.BACKWARD)
        for s in symbols
    }

    class HoldEqual(Strategy):
        def __init__(self):
            super().__init__()
            self._built = False

        def on_bar(self, ctx: Context) -> None:
            if not self._built:
                for s in ctx.data.symbols:
                    ctx.order_target_percent(s, 1 / 3)
                self._built = True

    cfg = EngineConfig(
        initial_cash=100000,
        commission=USCommissionModel(),
        slippage=ZeroSlippage(),
        allocation=EqualWeight(),
        rebalance=Rebalancer(RebalanceConfig(period="quarterly", threshold=0.1)),
    )
    result = EventDrivenEngine(cfg).run(HoldEqual(), data)
    # 末态各成分权重应接近 1/3
    final_weights = result.weights_curve.iloc[-1]
    for s in symbols:
        w = final_weights[s]
        # 现金 + 三标的,标的权重应各接近 1/3(允许再平衡后偏离)
        assert 0.2 < w < 0.45, f"{s} 权重 {w} 偏离目标过大"
    # 资金守恒
    prices = {s: float(data[s].df["close"].iloc[-1]) for s in symbols}
    result.account.check_invariant(prices)


# ── A 股涨跌停拒单 ────────────────────────────────────
def test_a_share_limit_rejection(make_csv, tmp_csv_dir: Path):
    """A 股涨停封板:买单应被拒。"""
    # 构造:前两日 close=10,第3日起封涨停 11(10*1.1=11,全天 high=low=close=11)
    n = 5
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = [10.0, 10.0, 11.0, 11.0, 11.0]  # 第3-5日封涨停(prev=10 → up=11)
    df = pd.DataFrame(
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000,
        }
    )
    tmp_csv_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_csv_dir / "600000.csv", index=False)
    from djinn.data import detect_market

    assert detect_market("600000") is Market.CN
    prov = CSVProvider(tmp_csv_dir, default_market=Market.CN)
    md = prov.get_ohlcv("600000", date(2024, 1, 2), date(2024, 1, 8), Adjust.BACKWARD)

    class BuyEveryDay(Strategy):
        def on_bar(self, ctx: Context) -> None:
            for s in ctx.data.symbols:
                ctx.buy(s, size=100)  # 每天买 100 股(固定数量,总有买入意图)

    from djinn.engine import TradeConstraints

    con = TradeConstraints(
        market=Market.CN,
        enforce_price_limit=True,
        enforce_t_plus_1=True,
        enforce_lot=True,
    )
    from djinn.engine.commission import ChinaCommissionModel

    cfg = EngineConfig(
        initial_cash=100000,
        commission=ChinaCommissionModel(),
        slippage=ZeroSlippage(),
        constraints=con,
    )
    result = EventDrivenEngine(cfg).run(BuyEveryDay(), {"600000": md})
    # 应有拒单(涨停封板日的买单被拒)
    real_rejections = [r for r in result.rejections if "noop" not in (r.tag or "")]
    assert len(real_rejections) > 0
    assert any("涨停" in r.reason for r in real_rejections)

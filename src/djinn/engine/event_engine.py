"""事件驱动回测引擎(Phase 1 核心)。

主循环按交易日推进,信号 ``t`` 生成、``t+1`` 开盘撮合,杜绝未来函数。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal

import pandas as pd

from djinn.data.market_data import MarketData
from djinn.data.schema import Market
from djinn.engine.broker import Broker, orders_from_intents
from djinn.engine.commission import CommissionModel, make_commission
from djinn.engine.constraints import TradeConstraints
from djinn.engine.events import Fill, Order, Rejection
from djinn.engine.slippage import SlippageModel, make_slippage
from djinn.portfolio.account import Account
from djinn.portfolio.allocation import Allocation, make_allocation
from djinn.portfolio.rebalance import Rebalancer
from djinn.portfolio.risk import RiskLimits, RiskManager
from djinn.strategy.base import Context, DataView, PortfolioView, Strategy
from djinn.utils.decimalmath import D, to_float
from djinn.utils.exceptions import BacktestCancelled
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class EngineConfig:
    """引擎运行配置。"""

    initial_cash: Decimal | float = 100000.0
    currency: str = "USD"
    # 费用 / 滑点 / 约束
    commission: CommissionModel | None = None
    slippage: SlippageModel | None = None
    constraints: TradeConstraints | None = None
    # 组合
    allocation: Allocation | None = None
    rebalance: Rebalancer | None = None
    risk: RiskManager | None = None
    # 成交执行参考价
    fill_ref: str = "open"  # "open" / "close" / "vwap"
    # 交易日对齐方式:intersection(交集,默认)/ union(并集,选股回测用)
    calendar: Literal["intersection", "union"] = "intersection"
    # 基准标的(union 模式下以其交易日历为主日历时提供)
    benchmark_symbol: str | None = None

    def resolve(
        self, default_market: Market
    ) -> tuple[CommissionModel, SlippageModel, TradeConstraints]:
        comm = self.commission or make_commission(default_market)
        slip = self.slippage or make_slippage("zero")
        con = self.constraints or TradeConstraints(market=default_market)
        return comm, slip, con


@dataclass
class BacktestResult:
    """回测结果。"""

    equity_curve: pd.Series  # index=交易日, value=净值(float)
    cash_curve: pd.Series
    positions_curve: pd.DataFrame  # index=交易日, columns=symbol, value=股数
    weights_curve: pd.DataFrame
    trades: list[Fill] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    account: Account | None = None
    config: EngineConfig | None = None
    symbols: list[str] = field(default_factory=list)
    benchmark_curve: pd.Series | None = None

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_rejections(self) -> int:
        return len(self.rejections)


class EventDrivenEngine:
    """事件驱动回测引擎。"""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def run(
        self,
        strategy: Strategy,
        data: dict[str, MarketData],
        *,
        benchmark: MarketData | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> BacktestResult:
        """运行回测。

        Args:
            strategy: 策略实例。
            data: {symbol: MarketData},多标的按交易日对齐。
            benchmark: 可选基准 MarketData,记录基准净值曲线。
            should_stop: 协作式取消回调,每日开盘前检查,返回 True 抛
                :class:`BacktestCancelled`(E4)。
        """
        if not data:
            raise ValueError("engine.run 需要至少一个标的数据")
        cfg = self.config
        symbols = list(data.keys())
        default_market = self._infer_market(data, symbols)
        comm, slip, con = cfg.resolve(default_market)

        account = Account(
            initial_cash=D(cfg.initial_cash),
            currency=cfg.currency,
            t_plus_1=con.enforce_t_plus_1,
        )
        broker = Broker(
            account=account,
            commission=comm,
            slippage=slip,
            constraints=con,
            fill_ref=cfg.fill_ref,
        )

        # 对齐所有标的的交易日索引(intersection 取交集;union 取并集/以基准日历为主)
        trading_index = self._aligned_index(data, benchmark)

        # D1:signals-only 策略预计算全量信号(无状态纯函数,滚动类指标 t 日值只依赖 ≤t)。
        # 预计算后主循环每日 O(1) asof 查表,替代 O(T) 全历史重算。
        presignals: dict[str, pd.Series] = {}
        if (
            getattr(strategy, "precompute_signals", False)
            and type(strategy).on_bar is Strategy.on_bar  # 未覆写 on_bar(走默认适配)
            and getattr(strategy, "scope", None) == "per_symbol"
        ):
            for sym, md in data.items():
                try:
                    presignals[sym] = strategy.signals(md.df)
                except Exception as e:  # 预计算失败 → 整体回退慢路径
                    _log.warning("预计算 %s 信号失败,回退逐日: %s", sym, e)
                    presignals.clear()
                    break
        strategy._presignals = presignals  # type: ignore[attr-defined]

        # 每标的的 prev_close 缓存(涨跌停需要昨收)
        prev_close: dict[str, float] = {}
        pending_orders: list[Order] = []
        order_counter = 1

        equity_hist: list[float] = []
        cash_hist: list[float] = []
        positions_hist: list[dict[str, float]] = []
        weights_hist: list[dict[str, float]] = []
        ts_hist: list[date] = []

        allocation = cfg.allocation or make_allocation("equal")
        rebalancer = cfg.rebalance
        risk = cfg.risk or RiskManager(RiskLimits())

        for i, ts in enumerate(trading_index):
            # E4:协作式取消检查点(每日开盘前)
            if should_stop is not None and should_stop():
                raise BacktestCancelled(f"回测已取消 @{ts.date()}")
            ts_date = ts.date()
            bars = self._bars_at(data, ts)
            prices = {s: b.close for s, b in bars.items() if b is not None}
            # 前向填充估值价:当日无行情的持仓标的用最近可得价(prev_close)估值,
            # 否则持仓市值会归零、破坏 Account 资金守恒不变式(union 日历下尤其重要)。
            prices_mtm = dict(prices)
            for s, pc in prev_close.items():
                prices_mtm.setdefault(s, pc)

            # 1. MARKET_OPEN:解冻 T+1
            if con.enforce_t_plus_1:
                account.unfreeze_all()

            # 2. PRICE:撮合昨日 pending 订单(用今日 bar)
            if pending_orders:
                # 撮合口径统一为开盘价:equity / 当前市值 / 成交价三处一致(A4)
                prices_open = {
                    s: b.open for s, b in bars.items() if b is not None and b.open > 0
                }
                for s, pc in prev_close.items():
                    prices_open.setdefault(s, pc)  # 当日无行情标的沿用昨收
                equity_now = account.equity_float(prices_open)
                # 两阶段撮合:先全部卖单(回笼现金),再全部买单,消除顺序依赖(A3)
                sells = [o for o in pending_orders if o.side == "sell"]
                buys = [o for o in pending_orders if o.side == "buy"]
                still_pending: list[Order] = []
                for order in sells + buys:
                    bar = bars.get(order.symbol)
                    if bar is None:
                        still_pending.append(order)
                        continue
                    result = broker.execute(
                        order, bar, prev_close.get(order.symbol), equity_now
                    )
                    if isinstance(result, Rejection) and result.retryable:
                        # 停牌 / 限价未达:订单继续挂起等下日(A6)
                        still_pending.append(order)
                pending_orders = still_pending

            # 3. SIGNAL:策略生成今日订单(进 pending,明日撮合)
            data_view = DataView(data, ts_date)
            portfolio_view = PortfolioView(account, prices_mtm, ts_date)
            ctx = Context(now=ts_date, data=data_view, portfolio=portfolio_view)
            try:
                strategy.on_bar(ctx)
            except Exception as e:
                _log.error("策略 on_bar 异常 @%s: %s", ts_date, e)
                raise

            new_orders = ctx.orders
            # 4. REBALANCE:再平衡注入调仓单
            if rebalancer is not None and i > 0:
                cur_weights = portfolio_view.weights()
                rb_orders_intents = rebalancer.maybe_rebalance(
                    ts_date, symbols, allocation, cur_weights, prices=prices_mtm
                )
                new_orders.extend(rb_orders_intents)

            # 5. 风控过滤
            cur_weights = portfolio_view.weights()
            filtered = risk.filter(new_orders, cur_weights)
            new_engine_orders = orders_from_intents(
                filtered, ts_date, counter_start=order_counter
            )
            order_counter += len(new_engine_orders)
            pending_orders.extend(new_engine_orders)

            # 6. MARKET_CLOSE:mark to market + 记录
            equity = account.mark_to_market(ts_date, prices_mtm)
            equity_hist.append(to_float(equity))
            cash_hist.append(to_float(account.cash))
            pos_snapshot: dict[str, float] = {}
            w_snapshot: dict[str, float] = {}
            for s in symbols:
                pos = account.positions.get(s)
                pos_snapshot[s] = to_float(pos.qty) if pos else 0.0
            w_all = portfolio_view.weights()  # 一次算 equity,避免逐 symbol 重算(D2)
            for s in symbols:
                w_snapshot[s] = w_all.get(s, 0.0)
            positions_hist.append(pos_snapshot)
            weights_hist.append(w_snapshot)
            ts_hist.append(ts_date)

            # 更新 prev_close
            for s, b in bars.items():
                if b is not None:
                    prev_close[s] = b.close

        # 组装结果
        idx = pd.DatetimeIndex(trading_index)
        equity_curve = pd.Series(equity_hist, index=idx, name="equity", dtype=float)
        cash_curve = pd.Series(cash_hist, index=idx, name="cash", dtype=float)
        positions_df = pd.DataFrame(positions_hist, index=idx).astype(float)
        weights_df = pd.DataFrame(weights_hist, index=idx).astype(float)

        benchmark_curve: pd.Series | None = None
        if benchmark is not None:
            # bfill 回填前导缺失:基准数据起点晚于策略首日时,bm.iloc[0] 不为 NaN,
            # 避免整条基准曲线因前导 NaN 传染为全 NaN。
            bm = benchmark.df["close"].reindex(idx).ffill().bfill()
            benchmark_curve = (bm / bm.iloc[0]) * equity_curve.iloc[0]

        return BacktestResult(
            equity_curve=equity_curve,
            cash_curve=cash_curve,
            positions_curve=positions_df,
            weights_curve=weights_df,
            trades=list(broker.fills),
            rejections=list(broker.rejections),
            account=account,
            config=cfg,
            symbols=symbols,
            benchmark_curve=benchmark_curve,
        )

    # ── 辅助 ────────────────────────────────────────────
    def _infer_market(self, data: dict[str, MarketData], symbols: list[str]) -> Market:
        markets = {data[s].market for s in symbols}
        if len(markets) == 1:
            return next(iter(markets))
        # 多市场混合:取第一个,约束按其配置(跨市场为 Phase 2)
        return data[symbols[0]].market

    def _aligned_index(
        self,
        data: dict[str, MarketData],
        benchmark: MarketData | None = None,
    ) -> pd.DatetimeIndex:
        """对齐各标的交易日索引。

        - ``intersection``(默认):所有标的交易日交集,保证每个 ts 都有全部标的的 bar。
        - ``union``(选股回测):交易日并集;若提供基准(或 ``benchmark_symbol`` 命中
          data)则以其交易日历为主日历(基准多为指数,交易日最全)。缺失 bar 的持仓
          由主循环前向填充估值。
        """
        if self.config.calendar == "union":
            if benchmark is not None:
                return pd.DatetimeIndex(benchmark.df.index).sort_values()
            bm_sym = self.config.benchmark_symbol
            if bm_sym is not None and bm_sym in data:
                return pd.DatetimeIndex(data[bm_sym].df.index).sort_values()
            idx_u: pd.DatetimeIndex | None = None
            for md in data.values():
                other = pd.DatetimeIndex(md.df.index)
                idx_u = other if idx_u is None else idx_u.union(other)
            if idx_u is None or len(idx_u) == 0:
                raise ValueError("标的交易日为空,无法对齐")
            return pd.DatetimeIndex(idx_u.sort_values())

        idx: pd.DatetimeIndex | None = None
        for md in data.values():
            other = pd.DatetimeIndex(md.df.index)
            if idx is None:
                idx = other
            else:
                idx = idx.intersection(other)
        if idx is None or len(idx) == 0:
            raise ValueError("标的交易日无交集,无法对齐")
        return pd.DatetimeIndex(idx.sort_values())

    def _bars_at(self, data: dict[str, MarketData], ts: pd.Timestamp) -> dict[str, Any]:
        """取各标的在 ts 的 Bar(无该日返回 None)。"""
        out: dict[str, Any] = {}
        for sym, md in data.items():
            if ts in md.df.index:
                out[sym] = md.bar_at(ts.date())
            else:
                out[sym] = None
        return out

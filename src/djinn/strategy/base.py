"""策略层:Strategy ABC + Context + DataView + PortfolioView + SignalAdapter。

事件驱动核心接口是 :meth:`Strategy.on_bar(ctx)`;简单单标的策略可只覆写
:meth:`Strategy.signals`,由 :class:`SignalAdapter` 自动转成 on_bar 下单逻辑。
"""

from __future__ import annotations

from abc import ABC
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from djinn.strategy.parameter import (
    _PARAM_ATTR,
    collect_params,
    get_params,
    param,
)
from djinn.strategy.signal import OrderIntent, Side, Signal
from djinn.utils.exceptions import StrategyError
from djinn.utils.logging import get_logger

if TYPE_CHECKING:
    from djinn.data.market_data import MarketData
    from djinn.portfolio.account import Account
    from djinn.portfolio.position import Position

_log = get_logger(__name__)

# 策略作用域。
SCOPE_PER_SYMBOL = "per_symbol"  # 每个成分独立跑信号
SCOPE_PORTFOLIO = "portfolio"  # 整体调仓,访问全标的数据


class DataView:
    """多标的行情只读视图(防未来函数)。

    通过 ``ctx.data["AAPL"]`` 取得单标的的 ``<= now`` 切片;切片上可直接访问
    ``.close``、``.volume`` 等 Series,或用 ``.iloc[-N:]`` 取最近 N 根。
    """

    def __init__(self, datas: dict[str, MarketData], now: date) -> None:
        self._datas = datas
        self._now = pd.Timestamp(now)
        self._cache: dict[str, pd.DataFrame] = {}

    def __getitem__(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._datas:
            raise KeyError(f"无标的数据: {symbol}")
        if symbol not in self._cache:
            df = self._datas[symbol].df
            self._cache[symbol] = df.loc[: self._now]
        return self._cache[symbol]

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._datas

    @property
    def symbols(self) -> list[str]:
        return list(self._datas)

    @property
    def now(self) -> pd.Timestamp:
        return self._now

    def latest(self, symbol: str, field: str = "close") -> float:
        """取 symbol 最近一日的字段值(无数据则抛错)。"""
        df = self[symbol]
        if field not in df.columns or len(df) == 0:
            raise StrategyError(f"无法取 {symbol}.{field}@{self._now.date()}")
        return float(df[field].iloc[-1])

    def history(self, symbol: str, field: str, n: int) -> pd.Series:
        """取最近 n 根的字段 Series。"""
        df = self[symbol]
        if field not in df.columns:
            raise StrategyError(f"无字段 {symbol}.{field}")
        return df[field].iloc[-n:]


class PortfolioView:
    """组合只读视图(供策略查询持仓 / 现金 / 净值,不可直接修改)。"""

    def __init__(self, account: Account, prices: dict[str, float], now: date) -> None:
        self._account = account
        self._prices = prices
        self._now = now
        self._equity_cache: float | None = None

    @property
    def cash(self) -> Any:
        from djinn.utils.decimalmath import to_float

        return to_float(self._account.cash)

    @property
    def equity(self) -> float:
        # 惰性缓存:PortfolioView 每交易日新建,缓存天然按日失效(D2)
        if self._equity_cache is None:
            self._equity_cache = float(self._account.equity(self._prices))
        return self._equity_cache

    @property
    def now(self) -> date:
        return self._now

    def position(self, symbol: str) -> Position | None:
        return self._account.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        pos = self._account.positions.get(symbol)
        return pos is not None and pos.qty > 0

    def weight(self, symbol: str) -> float:
        """该标的当前市值权重。"""
        pos = self._account.positions.get(symbol)
        if pos is None or pos.qty <= 0:
            return 0.0
        price = self._prices.get(symbol, 0.0)
        mv = float(pos.qty) * price
        eq = self.equity
        return mv / eq if eq > 0 else 0.0

    def weights(self) -> dict[str, float]:
        # 一次算 equity 再统一除,避免逐 symbol 重复算 equity(O(n²) → O(n))(D2)
        eq = self.equity
        if eq <= 0:
            return dict.fromkeys(self._account.positions, 0.0)
        return {
            s: float(p.qty) * self._prices.get(s, 0.0) / eq
            for s, p in self._account.positions.items()
            if p.qty > 0
        }

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._account.positions)


class Context:
    """策略执行上下文(每个交易日 on_bar 调用时构造)。

    策略通过 ``ctx.buy()`` / ``ctx.sell()`` / ``ctx.order_target_percent()`` 下单,
    订单存入 ``ctx.orders``,由引擎在 ``t+1`` 撮合。
    """

    def __init__(
        self,
        now: date,
        data: DataView,
        portfolio: PortfolioView,
    ) -> None:
        self.now = now
        self.data = data
        self.portfolio = portfolio
        self.orders: list[OrderIntent] = []

    # ── 下单 ────────────────────────────────────────────
    def buy(
        self,
        symbol: str,
        *,
        size: int | float | None = None,
        percent: float | None = None,
        limit: float | None = None,
    ) -> None:
        """买入。``size`` 指股数,``percent`` 指按当前净值的百分比买入。

        ``limit`` 为限价(买限价 ≥ 成交价才成交),未达则次日续挂。
        """
        if size is None and percent is None:
            raise StrategyError("buy 需指定 size 或 percent")
        if percent is not None:
            # 按净值百分比折算为"目标权重增加",引擎以 target_percent 处理
            cur = self.portfolio.weight(symbol)
            intent = OrderIntent(
                symbol=symbol,
                side="buy",
                target_percent=cur + percent,
                limit_price=limit,
                created_ts=self.now,
            )
        else:
            intent = OrderIntent(
                symbol=symbol,
                side="buy",
                size=size,
                limit_price=limit,
                created_ts=self.now,
            )
        self.orders.append(intent)

    def sell(
        self,
        symbol: str,
        *,
        size: int | float | None = None,
        percent: float | None = None,
        limit: float | None = None,
    ) -> None:
        """卖出。``limit`` 为限价(卖限价 ≤ 成交价才成交),未达则次日续挂。"""
        if size is None and percent is None:
            raise StrategyError("sell 需指定 size 或 percent")
        if percent is not None:
            cur = self.portfolio.weight(symbol)
            intent = OrderIntent(
                symbol=symbol,
                side="sell",
                target_percent=max(0.0, cur - percent),
                limit_price=limit,
                created_ts=self.now,
            )
        else:
            intent = OrderIntent(
                symbol=symbol,
                side="sell",
                size=size,
                limit_price=limit,
                created_ts=self.now,
            )
        self.orders.append(intent)

    def order_target_percent(self, symbol: str, pct: float) -> None:
        """将该标的调仓到目标市值权重 pct∈[0,1]。"""
        if not 0.0 <= pct <= 1.0:
            raise StrategyError(f"target_percent 必须在 [0,1],实际 {pct}")
        side: Side = "buy" if pct >= self.portfolio.weight(symbol) else "sell"
        self.orders.append(
            OrderIntent(
                symbol=symbol, side=side, target_percent=pct, created_ts=self.now
            )
        )

    def has_position(self, symbol: str) -> bool:
        return self.portfolio.has_position(symbol)

    # ── 便捷 ────────────────────────────────────────────
    def clear_orders(self) -> None:
        self.orders.clear()


class Strategy(ABC):  # noqa: B024
    """策略抽象基类。

    子类用 ``param()`` 声明参数,覆写 :meth:`on_bar`(事件驱动核心)或仅
    覆写 :meth:`signals`(单标的快捷路径,由默认 :meth:`on_bar` 适配)。

    抽象性通过 ``__init__`` guard 与 ``__init_subclass__`` 检查约束(无 abstractmethod)。
    """

    scope: str = SCOPE_PER_SYMBOL
    # D1:signals-only 策略默认启用预计算(引擎一次性算全量信号,主循环 O(1) 查表)。
    # 前提:signals 为因果运算(t 日输出仅依赖 ≤t 输入),rolling/shift 类指标满足。
    precompute_signals: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        params = collect_params(cls)
        setattr(cls, _PARAM_ATTR, params)
        # 子类必须实现 on_bar 或 signals 之一
        has_on_bar = (
            "on_bar" in cls.__dict__ and cls.__dict__["on_bar"] is not Strategy.on_bar
        )
        has_signals = (
            "signals" in cls.__dict__
            and cls.__dict__["signals"] is not Strategy.signals
        )
        if not has_on_bar and not has_signals:
            raise TypeError(f"策略 {cls.__name__} 必须实现 on_bar() 或 signals() 之一")

    def __init__(self, **params: Any) -> None:
        if type(self) is Strategy:
            raise TypeError("Strategy 是抽象基类,不能直接实例化")
        declared = get_params(type(self))
        for name, p in declared.items():
            setattr(self, name, p.default)
        for k, v in params.items():
            if k not in declared:
                raise StrategyError(f"策略 {type(self).__name__} 无参数 {k!r}")
            setattr(self, k, v)

    def on_bar(self, ctx: Context) -> None:
        """默认实现:转调 :meth:`signals`,在信号变化点下单。

        复杂 / 组合策略应覆写本方法以获得完整 Context 访问权。
        """
        if self.scope != SCOPE_PER_SYMBOL:
            raise StrategyError(
                f"{type(self).__name__} scope={self.scope} 必须覆写 on_bar"
            )
        if not hasattr(self, "_signal_state"):
            self._signal_state: dict[str, int] = {}
        # D1 快速路径:引擎预计算的信号序列,O(1) asof 查表(≤now 最近非 NaN 值)
        presignals = getattr(self, "_presignals", None)
        for symbol in ctx.data.symbols:
            if presignals is not None and symbol in presignals:
                ser = presignals[symbol]
                asof_val = ser.asof(pd.Timestamp(ctx.now))
                today_sig = int(asof_val) if pd.notna(asof_val) else 0
            else:
                df = ctx.data[symbol]
                if len(df) == 0:
                    continue
                sig_series = self.signals(df)
                today_sig = int(sig_series.iloc[-1]) if len(sig_series) else 0
            last = self._signal_state.get(symbol, 0)
            if today_sig != last:
                if today_sig == 1:
                    ctx.order_target_percent(symbol, 1.0)
                elif today_sig == -1:
                    ctx.order_target_percent(symbol, 0.0)
                self._signal_state[symbol] = today_sig

    def signals(self, data: pd.DataFrame) -> pd.Series:
        """简单策略快捷路径:输入标的历史切片(<= now),输出信号 Series {1,-1,0}。

        仅 scope=per_symbol 时由默认 :meth:`on_bar` 调用。
        """
        raise NotImplementedError(f"{type(self).__name__} 未实现 signals()")

    @property
    def params(self) -> dict[str, Any]:
        declared = get_params(type(self))
        return {k: getattr(self, k) for k in declared}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"


class SignalAdapter(Strategy):
    """把仅实现 ``signals()`` 的策略显式适配为可执行策略。

    多数情况下无需手动使用——:meth:`Strategy.on_bar` 的默认实现已对
    ``scope=per_symbol`` 的 signals-only 策略做了适配。本类用于需要把一个
    signals-only 策略当作 on_bar 策略注入到组合流程时显式包装。
    """

    def __init__(self, strategy: Strategy) -> None:
        if type(strategy) is Strategy:
            raise TypeError("不能包装 Strategy 基类")
        self._strategy = strategy
        self._last_signal: dict[str, int] = {}
        # 实例属性遮蔽类属性,转发被包装策略的作用域
        self.scope = strategy.scope

    @property
    def params(self) -> dict[str, Any]:
        return self._strategy.params

    def on_bar(self, ctx: Context) -> None:
        for symbol in ctx.data.symbols:
            df = ctx.data[symbol]
            if len(df) == 0:
                continue
            sig_series = self._strategy.signals(df)
            today_sig = int(sig_series.iloc[-1]) if len(sig_series) else 0
            last = self._last_signal.get(symbol, 0)
            if today_sig != last:
                if today_sig == 1:
                    ctx.order_target_percent(symbol, 1.0)
                elif today_sig == -1:
                    ctx.order_target_percent(symbol, 0.0)
                self._last_signal[symbol] = today_sig


# 重新导出 param 供 ``from djinn.strategy import param``
__all__ = [
    "SCOPE_PER_SYMBOL",
    "SCOPE_PORTFOLIO",
    "Context",
    "DataView",
    "OrderIntent",
    "PortfolioView",
    "Signal",
    "SignalAdapter",
    "Strategy",
    "param",
]

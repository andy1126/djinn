"""账户:Decimal 现金账本 + 持仓会计。

资金守恒不变式(任意成交序列后):
    cash + Σ(positions.market_value) == equity(prices)

买入:cash -= (qty*price + commission);qty 与 avg_cost 更新;A 股计入 frozen。
卖出:cash += (qty*price - commission);available 校验;realized_pnl 累计。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from djinn.portfolio.position import Position
from djinn.utils.decimalmath import D, q_money, q_shares, to_float
from djinn.utils.exceptions import AccountError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class LedgerEntry:
    """单笔成交(Decimal 口径)的账本分录。"""

    timestamp: date
    symbol: str
    side: str  # "buy" / "sell"
    qty: Decimal
    price: Decimal
    commission: Decimal
    tag: str = ""

    @property
    def gross(self) -> Decimal:
        return q_money(self.qty * self.price)

    @property
    def net_cash_delta(self) -> Decimal:
        """对现金的净影响(买入为负,卖出为正,已扣费用)。"""
        if self.side == "buy":
            return -(self.gross + self.commission)
        return self.gross - self.commission


@dataclass
class Account:
    """现金 + 持仓账户(全 Decimal)。

    Args:
        initial_cash: 初始现金。
        currency: 币种(仅记录,不参与换算)。
        t_plus_1: 是否启用 T+1(A 股当日买入冻结,次日解冻)。
    """

    initial_cash: Decimal
    currency: str = "USD"
    t_plus_1: bool = False
    cash: Decimal = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[LedgerEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.initial_cash = q_money(self.initial_cash)
        self.cash = self.initial_cash

    # ── 查询 ────────────────────────────────────────────
    def equity(self, prices: Mapping[str, float | Decimal]) -> Decimal:
        """总权益 = cash + Σ 持仓市值(按给定价格)。"""
        mv = D(0)
        for sym, pos in self.positions.items():
            if pos.qty <= 0:
                continue
            price = D(prices.get(sym, 0))
            mv += q_money(pos.qty * price)
        return q_money(self.cash + mv)

    def position_value(self, prices: Mapping[str, float | Decimal]) -> Decimal:
        mv = D(0)
        for sym, pos in self.positions.items():
            if pos.qty <= 0:
                continue
            price = D(prices.get(sym, 0))
            mv += q_money(pos.qty * price)
        return mv

    def get_position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol=symbol))

    # ── 成交 ────────────────────────────────────────────
    def buy(
        self,
        symbol: str,
        qty: Decimal,
        price: Decimal | float,
        commission: Decimal | float,
        *,
        freeze: bool | None = None,
        timestamp: date | None = None,
        tag: str = "",
    ) -> LedgerEntry:
        """买入成交:扣现金、更新持仓均价。

        Args:
            freeze: 是否冻结(A 股 T+1);None 则按 self.t_plus_1。
        """
        qty = q_shares(qty)
        if qty <= 0:
            raise AccountError(f"买入股数必须为正,实际 {qty}")
        price = D(price)
        commission = q_money(D(commission))
        gross = q_money(qty * price)
        cost = gross + commission
        if self.cash < cost:
            raise AccountError(f"现金不足:需 {cost},有 {self.cash}")
        self.cash = q_money(self.cash - cost)
        pos = self.get_position(symbol)
        new_qty = q_shares(pos.qty + qty)
        # 加权均价 = (旧持仓*旧均价 + 新买入金额) / 新总持仓
        if new_qty > 0:
            pos.avg_cost = q_money((pos.qty * pos.avg_cost + gross) / new_qty)
        pos.qty = new_qty
        pos._cum_buy_cost += cost
        if freeze is None:
            freeze = self.t_plus_1
        if freeze:
            pos.frozen = q_shares(pos.frozen + qty)
        else:
            pos.available = q_shares(pos.available + qty)
        fill = LedgerEntry(
            timestamp or date.min, symbol, "buy", qty, price, commission, tag
        )
        self.fills.append(fill)
        return fill

    def sell(
        self,
        symbol: str,
        qty: Decimal,
        price: Decimal | float,
        commission: Decimal | float,
        *,
        timestamp: date | None = None,
        tag: str = "",
    ) -> LedgerEntry:
        """卖出成交:加现金、校验 available、累计已实现盈亏。"""
        qty = q_shares(qty)
        if qty <= 0:
            raise AccountError(f"卖出股数必须为正,实际 {qty}")
        pos = self.positions.get(symbol)
        if pos is None or pos.available < qty:
            avail = pos.available if pos else D(0)
            raise AccountError(f"可卖股数不足:需 {qty},可用 {avail}({symbol})")
        price = D(price)
        commission = q_money(D(commission))
        gross = q_money(qty * price)
        net = gross - commission
        self.cash = q_money(self.cash + net)
        # 已实现盈亏 = (卖价 - 均价) * qty - 摊销费用(简化:不含费用摊销)
        realized = q_money((price - pos.avg_cost) * qty)
        pos.realized_pnl = q_money(pos.realized_pnl + realized)
        pos.available = q_shares(pos.available - qty)
        pos.qty = q_shares(pos.qty - qty)
        if pos.qty <= 0:
            pos.qty = D(0)
            pos.avg_cost = D(0)
            pos.frozen = D(0)
            pos.available = D(0)
        fill = LedgerEntry(
            timestamp or date.min, symbol, "sell", qty, price, commission, tag
        )
        self.fills.append(fill)
        return fill

    def unfreeze_all(self) -> None:
        """次日开盘前解冻所有持仓(T+1)。"""
        for pos in self.positions.values():
            if pos.frozen > 0:
                pos.unfreeze()

    # ── mark to market ─────────────────────────────────
    def mark_to_market(
        self, ts: date, prices: Mapping[str, float | Decimal]
    ) -> Decimal:
        """按当日价格标记持仓 last_price,返回当日权益(float 口径给净值序列)。"""
        equity = self.equity(prices)
        for sym, price in prices.items():
            pos = self.positions.get(sym)
            if pos is not None and pos.qty > 0:
                pos.last_price = D(price)
                pos.last_mark_date = ts
        return equity

    # ── 分红 ───────────────────────────────────────────
    def receive_dividend(
        self,
        symbol: str,
        per_share: Decimal | float,
        *,
        reinvest: bool = False,
        price: Decimal | float | None = None,
    ) -> None:
        """分红:现金入账或按 ``price`` 再投资(碎股)。"""
        pos = self.positions.get(symbol)
        if pos is None or pos.qty <= 0:
            return
        amt = q_money(pos.qty * D(per_share))
        if not reinvest:
            self.cash = q_money(self.cash + amt)
            return
        if price is None or D(price) <= 0:
            self.cash = q_money(self.cash + amt)
            return
        # 再投资:用分红金额按 price 买入碎股(免费用,不扣现金)
        extra = q_shares(amt / D(price))
        # 零成本增量摊薄均价(总成本不变、股数增加),否则后续 realized_pnl 高估
        pos.avg_cost = q_money(pos.avg_cost * pos.qty / (pos.qty + extra))
        pos.qty = q_shares(pos.qty + extra)

    def apply_split(self, symbol: str, ratio: Decimal | float) -> None:
        """拆股:股数 × ratio、均价 ÷ ratio(市值与总成本不变)。"""
        pos = self.positions.get(symbol)
        if pos is None or pos.qty <= 0:
            return
        r = D(ratio)
        if r <= 0:
            return
        pos.qty = q_shares(pos.qty * r)
        pos.available = q_shares(pos.available * r)
        pos.avg_cost = q_money(pos.avg_cost / r)

    # ── 诊断 ───────────────────────────────────────────
    def check_invariant(self, prices: Mapping[str, float | Decimal]) -> None:
        """校验资金守恒(测试用):cash + 持仓市值 == equity。"""
        eq = self.equity(prices)
        pv = self.position_value(prices)
        assert eq == q_money(self.cash + pv), f"资金不守恒: {eq} != {self.cash}+{pv}"

    def equity_float(self, prices: Mapping[str, float | Decimal]) -> float:
        return to_float(self.equity(prices))

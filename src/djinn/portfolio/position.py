"""持仓:股数 / 均价 / 可用(T+1 冻结)用 Decimal 记账。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from djinn.utils.decimalmath import D, q_shares


@dataclass
class Position:
    """单标的持仓。

    Attributes:
        symbol: 标的代码。
        qty: 总持仓股数(Decimal,含当日买入未解冻部分)。
        avg_cost: 持仓均价(Decimal,含费用摊销口径由 Account 决定)。
        available: 可卖股数(A 股 T+1:当日买入不计入 available,次日解冻)。
        frozen: 当日买入冻结股数(qty = available + frozen)。
        last_price: 最近 mark 价(用于市值计算)。
        last_mark_date: 最近 mark 日期。
    """

    symbol: str
    qty: Decimal = D(0)
    avg_cost: Decimal = D(0)
    available: Decimal = D(0)
    frozen: Decimal = D(0)
    last_price: Decimal = D(0)
    last_mark_date: date | None = None
    # 累计已实现盈亏(Decimal,供分析)
    realized_pnl: Decimal = D(0)
    # 成交记录辅助:累计买入金额(含费)、累计卖出金额(含费)
    _cum_buy_cost: Decimal = field(default_factory=lambda: D(0), repr=False)

    @property
    def market_value(self) -> Decimal:
        """按 last_price 计算的市值。"""
        return q_shares(self.qty) * self.last_price

    def can_sell(self, size: Decimal) -> bool:
        """是否可卖出 size 股(校验 available)。"""
        return self.available >= size and size > 0

    def unfreeze(self) -> None:
        """次日开盘前:把冻结股数转入 available(T+1 解冻)。"""
        self.available = q_shares(self.available + self.frozen)
        self.frozen = D(0)

    def is_empty(self) -> bool:
        return self.qty <= 0

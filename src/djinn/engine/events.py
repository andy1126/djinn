"""引擎执行回报:订单 / 成交 / 拒单(事件驱动引擎)。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class Order:
    """已提交待撮合的订单(由 OrderIntent 转换,带 id)。"""

    id: int
    symbol: str
    side: str  # "buy" / "sell"
    created_ts: date
    qty: int | float | None = None
    target_percent: float | None = None
    limit_price: float | None = None
    tag: str = ""

    def is_target(self) -> bool:
        return self.target_percent is not None


@dataclass(slots=True)
class Fill:
    """成交回执(引擎口径,float 价格 + Decimal 金额由 Account 记)。"""

    order_id: int
    timestamp: date
    symbol: str
    side: str
    qty: float
    price: float
    commission: float
    tag: str = ""


@dataclass(slots=True)
class Rejection:
    """订单拒绝回执。"""

    order_id: int
    timestamp: date
    symbol: str
    side: str
    reason: str
    requested_qty: float = 0.0
    tag: str = ""
    retryable: bool = False  # 停牌 / 限价未达等"次日续挂"类拒单

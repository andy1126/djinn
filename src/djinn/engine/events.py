"""事件类型与优先级(事件驱动引擎)。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from typing import Any


class EventPriority(IntEnum):
    """事件处理优先级(同时间戳下,数值小者先处理)。"""

    MARKET_OPEN = 1  # 开盘:解冻 T+1、挂单入场
    PRICE = 2  # 行情:撮合昨日挂单(t+1 开盘价)
    FILL = 3  # 成交回执
    SIGNAL = 4  # 策略信号生成(产生 t+1 订单)
    REBALANCE = 5  # 再平衡注入订单
    MARKET_CLOSE = 6  # 收盘:mark to market


@dataclass(order=True, slots=True)
class Event:
    """事件(按 (timestamp, priority, seq) 排序)。"""

    timestamp: date
    priority: EventPriority
    seq: int = field(default=0)
    payload: Any = field(default=None, compare=False)
    kind: str = field(default="", compare=False)


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

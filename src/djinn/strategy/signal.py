"""信号与订单意图类型。

- :class:`Signal`:策略输出的离散信号(+1 买 / -1 卖 / 0 持平)。
- :class:`OrderIntent`:策略经 :class:`Context` 下达的订单意图(数量 / 目标权重),
  由引擎在 ``t+1`` 撮合。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from typing import Literal

Side = Literal["buy", "sell"]


class SignalValue(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass(frozen=True, slots=True)
class Signal:
    """单标的单日信号。"""

    timestamp: date
    symbol: str
    value: int  # +1 / -1 / 0

    def __post_init__(self) -> None:
        if self.value not in (1, -1, 0):
            raise ValueError(f"信号值必须为 1/-1/0,实际 {self.value}")


@dataclass(slots=True)
class OrderIntent:
    """策略下达的订单意图(待引擎 t+1 撮合)。

    两种模式(二选一):
    - 数量模式:``size`` 为正整数(股数,buy)/ 负或正(sell)。
    - 目标权重模式:``target_percent`` 为该标的目标市值占比 [0,1]。
    """

    symbol: str
    side: Side
    size: int | float | None = None
    target_percent: float | None = None
    limit_price: float | None = None
    created_ts: date | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        if self.size is None and self.target_percent is None:
            raise ValueError("OrderIntent 必须指定 size 或 target_percent")
        if self.size is not None and self.size <= 0 and self.side == "buy":
            raise ValueError("买单 size 必须为正")

    def is_target(self) -> bool:
        return self.target_percent is not None


@dataclass(slots=True)
class SignalBatch:
    """一个交易日内全部标的的信号集合。"""

    timestamp: date
    signals: dict[str, int] = field(default_factory=dict)

    def add(self, symbol: str, value: int) -> None:
        if value not in (1, -1, 0):
            raise ValueError(f"信号值必须为 1/-1/0,实际 {value}")
        self.signals[symbol] = value

    def get(self, symbol: str) -> int:
        return self.signals.get(symbol, 0)

"""Round-trip(回合)交易配对:FIFO 开仓→平仓,含双边佣金摊销。

从 fills 序列重建"一次完整买卖"的回合,得到每回合净盈亏(含双边佣金)与持仓
天数。统计层用 float(与 metrics float64 口径一致,不违反账本 Decimal 不变量);
账本层的 ``Position.realized_pnl`` 口径保持不变(不含费用,见 account docstring)。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass
class RoundTrip:
    """一个已平仓的交易回合。"""

    symbol: str
    open_date: date
    close_date: date
    qty: float  # 本回合股数
    open_price: float  # 加权平均开仓价
    close_price: float  # 加权平均平仓价
    pnl: float  # 净盈亏(含双边佣金摊销)
    holding_days: int  # (close_date - open_date).days


@dataclass
class _OpenLot:
    """FIFO 开仓队列中的一笔未平仓批次。"""

    date: date
    qty: float
    price: float
    commission: float


def pair_round_trips(fills: list[Any]) -> list[RoundTrip]:
    """按 FIFO 把 fills 配对为回合。

    规则:
    - 每个标的维护一个 FIFO 开仓队列;
    - 买单入队;卖单从队首依次冲销,每次冲销生成一个 :class:`RoundTrip`;
    - 佣金按被冲销股数比例摊派到开仓 / 平仓两侧;
    - 回测结束仍未平仓的开仓批不生成回合(浮盈不进胜率)。
    """
    open_lots: dict[str, list[_OpenLot]] = {}
    trips: list[RoundTrip] = []
    for f in fills:
        side = f.side
        qty = float(f.qty)
        price = float(f.price)
        comm = float(f.commission)
        ts = f.timestamp
        symbol = f.symbol
        lots = open_lots.setdefault(symbol, [])

        if side == "buy":
            lots.append(_OpenLot(date=ts, qty=qty, price=price, commission=comm))
            continue

        # sell:从队首 FIFO 冲销
        remaining = qty
        while remaining > 0 and lots:
            lot = lots[0]
            close_qty = min(remaining, lot.qty)
            open_comm = lot.commission * (close_qty / lot.qty) if lot.qty else 0.0
            close_comm = comm * (close_qty / qty) if qty else 0.0
            pnl = (price - lot.price) * close_qty - open_comm - close_comm
            trips.append(
                RoundTrip(
                    symbol=symbol,
                    open_date=lot.date,
                    close_date=ts,
                    qty=close_qty,
                    open_price=lot.price,
                    close_price=price,
                    pnl=pnl,
                    holding_days=(ts - lot.date).days,
                )
            )
            remaining -= close_qty
            if close_qty >= lot.qty:
                lots.pop(0)
            else:
                lots[0] = _OpenLot(
                    date=lot.date,
                    qty=lot.qty - close_qty,
                    price=lot.price,
                    commission=lot.commission * (1.0 - close_qty / lot.qty),
                )
    return trips

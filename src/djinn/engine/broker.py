"""Broker:订单撮合 → 成交 / 拒单。

工作流(每个交易日开盘):
1. 取昨日策略产生的 pending 订单;
2. 用当日开盘价(或指定参考价)应用滑点得成交价;
3. 交易约束校验(停牌/涨跌停/最小手/资金);
4. 佣金计算;
5. Account.buy/sell 成交(Decimal 记账),或记录 Rejection。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from djinn.data.schema import Bar
from djinn.engine.commission import CommissionModel
from djinn.engine.constraints import TradeConstraints, check_constraints
from djinn.engine.events import Fill, Order, Rejection
from djinn.engine.slippage import SlippageModel, fill_price
from djinn.portfolio.account import Account
from djinn.utils.decimalmath import D, floor_shares, q_shares
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


def _noop(order: Order, bar: Bar) -> Rejection:
    """目标已达成 / 无需操作的占位(带 tag=noop)。

    引擎据 ``tag`` 含 "noop" 判定为静默无操作,不计入真实拒单统计。
    """
    return Rejection(
        order_id=order.id,
        timestamp=bar.timestamp,
        symbol=order.symbol,
        side=order.side,
        reason="noop:目标已达成",
        requested_qty=0.0,
        tag=(order.tag + " noop").strip(),
    )


@dataclass
class Broker:
    """撮合经纪商。"""

    account: Account
    commission: CommissionModel
    slippage: SlippageModel
    constraints: TradeConstraints
    fills: list[Fill] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)

    def execute(
        self,
        order: Order,
        bar: Bar,
        prev_close: float | None,
        equity: float,
    ) -> Fill | Rejection:
        """撮合单笔订单(以 bar.open 为参考价,t+1 执行)。"""
        ref_price = bar.open if bar.open > 0 else bar.close
        # 1. 解析目标股数(target_percent -> qty)
        qty = self._resolve_qty(order, ref_price, equity)
        if qty is None or qty <= 0:
            # target_percent 已达成(无需调整):静默无操作,不计拒单
            if order.is_target():
                if (order.target_percent or 0) <= 0:
                    # 目标权重 0:若持仓则卖到 0,否则无操作
                    pos = self.account.positions.get(order.symbol)
                    if pos is None or pos.qty <= 0:
                        return _noop(order, bar)
                    qty = float(pos.available)
                else:
                    # 目标权重 > 0 且 delta<=0:已达成
                    return _noop(order, bar)
            # size 订单数量为 0:记拒单
            rej = Rejection(
                order_id=order.id,
                timestamp=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                reason="订单数量为0",
                requested_qty=0.0,
                tag=order.tag,
            )
            self.rejections.append(rej)
            return rej

        # 2. 滑点
        price = fill_price(
            self.slippage, order.side, ref_price, bar, order_qty=float(qty)
        )

        # 3. 约束校验(传入粗略资金)
        check = check_constraints(
            side=order.side,
            raw_qty=D(qty),
            bar=bar,
            prev_close=prev_close,
            account_cash=self.account.cash,
            ref_price=price,
            constraints=self.constraints,
        )
        if not check.ok:
            rej = Rejection(
                order_id=order.id,
                timestamp=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                reason=check.reason,
                requested_qty=float(qty),
                tag=order.tag,
            )
            self.rejections.append(rej)
            return rej
        final_qty = check.adjusted_qty if check.adjusted_qty is not None else D(qty)
        final_qty = q_shares(final_qty)
        if final_qty <= 0:
            rej = Rejection(
                order_id=order.id,
                timestamp=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                reason="调整后数量为0",
                requested_qty=float(qty),
                tag=order.tag,
            )
            self.rejections.append(rej)
            return rej

        # 4. 佣金
        comm = self.commission.cost(order.side, price, final_qty)

        # 5. 成交(Account 兜底资金/可用校验)
        # 资金不足时按可用资金缩减股数重试一次(处理 target_percent 满仓的尾差)
        if order.side == "buy":
            final_qty, comm = self._maybe_shrink_buy(order, price, final_qty, comm, bar)

        try:
            if order.side == "buy":
                self.account.buy(
                    order.symbol,
                    final_qty,
                    price,
                    comm,
                    timestamp=bar.timestamp,
                    tag=order.tag,
                )
            else:
                self.account.sell(
                    order.symbol,
                    final_qty,
                    price,
                    comm,
                    timestamp=bar.timestamp,
                    tag=order.tag,
                )
        except Exception as e:
            rej = Rejection(
                order_id=order.id,
                timestamp=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                reason=f"账户拒绝: {e}",
                requested_qty=float(final_qty),
                tag=order.tag,
            )
            self.rejections.append(rej)
            return rej

        fill = Fill(
            order_id=order.id,
            timestamp=bar.timestamp,
            symbol=order.symbol,
            side=order.side,
            qty=float(final_qty),
            price=float(price),
            commission=float(comm),
            tag=order.tag,
        )
        self.fills.append(fill)
        return fill

    def _maybe_shrink_buy(
        self,
        order: Order,
        price: float,
        qty: Decimal,
        comm: Decimal,
        bar: Bar,
    ) -> tuple[Decimal, Decimal]:
        """若买入资金不足,按可用现金缩减股数(扣佣金)一次。"""
        need = D(qty) * D(price) + comm
        if need <= self.account.cash:
            return qty, comm
        cash = self.account.cash
        if cash <= 0:
            return qty, comm
        lot = self.constraints.market.lot_size if self.constraints.enforce_lot else 1
        # 二分/迭代:估计 max_qty = cash / (price * (1+rate)),再按佣金修正
        # 简化:用 (cash - min_commission) / price 作为上界,向下取整到 lot
        from djinn.engine.commission import ConservativeCommissionModel

        min_comm = (
            self.commission.min_commission
            if isinstance(self.commission, ConservativeCommissionModel)
            else D(0)
        )
        usable = cash - min_comm
        if usable <= 0:
            return qty, comm
        max_qty = D(usable) / D(price)
        if lot > 1:
            max_qty = floor_shares(max_qty, lot)
        else:
            max_qty = q_shares(max_qty)
        if max_qty <= 0:
            return qty, comm
        new_comm = self.commission.cost(order.side, price, max_qty)
        # 二次校验:扣佣金后是否仍超
        if D(max_qty) * D(price) + new_comm > cash:
            # 再缩一档
            max_qty = (
                floor_shares(D(cash - new_comm) / D(price), lot)
                if lot > 1
                else q_shares(D(cash - new_comm) / D(price))
            )
            if max_qty <= 0:
                return qty, comm
            new_comm = self.commission.cost(order.side, price, max_qty)
        _log.debug("资金缩减 %s: %s → %s 股(现金 %s)", order.symbol, qty, max_qty, cash)
        return max_qty, new_comm

    def _resolve_qty(
        self, order: Order, ref_price: float, equity: float
    ) -> float | None:
        """把订单的 size 或 target_percent 统一为股数。"""
        if order.is_target():
            target_mv = (order.target_percent or 0.0) * equity
            if order.side == "buy":
                # 目标持仓市值 - 当前持仓市值 = 需买入
                pos = self.account.positions.get(order.symbol)
                cur_mv = float(pos.qty) * ref_price if pos else 0.0
                delta_mv = target_mv - cur_mv
                if delta_mv <= 0:
                    return None
                return delta_mv / ref_price if ref_price > 0 else 0.0
            else:  # sell
                pos = self.account.positions.get(order.symbol)
                cur_mv = float(pos.qty) * ref_price if pos else 0.0
                delta_mv = cur_mv - target_mv
                if delta_mv <= 0:
                    return None
                return delta_mv / ref_price if ref_price > 0 else 0.0
        # size 模式
        if order.qty is None:
            return None
        return float(order.qty)


def orders_from_intents(
    intents: list[Any],  # list[OrderIntent]
    ts: date,
    counter_start: int = 1,
) -> list[Order]:
    """把策略 OrderIntent 转成引擎 Order(分配自增 id)。"""
    out: list[Order] = []
    i = counter_start
    for it in intents:
        out.append(
            Order(
                id=i,
                symbol=it.symbol,
                side=it.side,
                qty=it.size,
                target_percent=it.target_percent,
                limit_price=it.limit_price,
                created_ts=it.created_ts or ts,
                tag=it.tag,
            )
        )
        i += 1
    return out

"""交易约束:停牌 / 涨跌停 / 最小手 / T+1 / 资金。

约束在 :meth:`Broker.execute` 撮合前依次校验,任一不满足返回 :class:`Rejection`,
否则放行。资金最终由 :class:`Account` 在成交时兜底校验(双保险)。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from djinn.data.schema import Bar, Market
from djinn.utils.decimalmath import D, floor_shares, q_shares
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class TradeConstraints:
    """交易约束配置。"""

    market: Market = Market.US
    enforce_lot: bool = True  # 最小手(A/港股 100,美股 1)
    enforce_price_limit: bool = True  # 涨跌停
    enforce_suspension: bool = True  # 停牌
    enforce_t_plus_1: bool = False  # T+1(A 股)
    # 涨跌停幅度覆盖(用于创业板/科创板/ST 特判;None 用 market 默认)
    price_limit_pct: float | None = None
    # 标的级的涨跌停幅度覆盖(代码 -> pct)
    symbol_limits: dict[str, float] | None = None
    # 单笔成交占当日成交量上限(0 = 不限制;选股回测建议 0.1)(A11)
    max_volume_share: float = 0.0


def _price_limit_for(symbol: str, constraints: TradeConstraints) -> float | None:
    if constraints.symbol_limits and symbol in constraints.symbol_limits:
        return constraints.symbol_limits[symbol]
    if constraints.price_limit_pct is not None:
        return constraints.price_limit_pct
    return constraints.market.price_limit_pct


def limit_prices(
    prev_close: float, symbol: str, constraints: TradeConstraints
) -> tuple[float, float] | None:
    """根据昨收计算涨跌停价 (limit_up, limit_down);无限制市场返回 None。"""
    pct = _price_limit_for(symbol, constraints)
    if pct is None:
        return None
    return prev_close * (1 + pct), prev_close * (1 - pct)


@dataclass
class CheckResult:
    """约束校验结果。"""

    ok: bool
    reason: str = ""
    adjusted_qty: Decimal | None = None
    retryable: bool = False  # 停牌等"次日续挂"类拒单(而非永久丢弃)


def check_constraints(
    side: str,
    raw_qty: Decimal,
    bar: Bar,
    prev_close: float | None,
    account_cash: Decimal,
    ref_price: float,
    constraints: TradeConstraints,
    *,
    available_qty: Decimal | None = None,
) -> CheckResult:
    """逐项校验订单约束(返回 ok 与调整后股数)。

    Args:
        available_qty: 当前可用股数(卖出方向用于夹到可用,允许零股全出)。
    """
    # 1. 停牌
    if constraints.enforce_suspension and bar.is_suspended:
        return CheckResult(False, reason=f"停牌({bar.timestamp})", retryable=True)

    # 2. 最小手:买入向下取整到 lot 的整数倍;卖出不取整(允许零股一次性卖出)
    qty = raw_qty
    lot = constraints.market.lot_size if constraints.enforce_lot else 1
    if constraints.enforce_lot and lot > 1 and side == "buy":
        qty = floor_shares(qty, lot)
        if qty <= 0:
            return CheckResult(False, reason=f"不足最小手 {lot} 股")
    elif constraints.enforce_lot and lot > 1 and side == "sell":
        # 卖出:整手部分之外的零股允许一次性卖出(不得超过可用股数);
        # 不向下取整,由后续 Account.sell 的 available 校验兜底。
        if available_qty is not None and raw_qty > available_qty:
            qty = available_qty  # 夹到可用(零股全出)
        else:
            qty = raw_qty  # 不取整,允许零股

    # 3. 涨跌停:开盘即涨停的买单、跌停的卖单拒单
    if constraints.enforce_price_limit and prev_close is not None:
        lim = limit_prices(prev_close, bar.symbol, constraints)
        if lim is not None:
            up, down = lim
            # 涨停:买单价 >= 涨停价且当日开盘即涨停(用 high 判断是否封板)
            if (
                side == "buy"
                and bar.high <= up * (1 + 1e-6)
                and bar.low >= up * (1 - 1e-6)
            ):
                # 全天封涨停:买单无法成交
                if bar.close >= up * (1 - 1e-6):
                    return CheckResult(False, reason=f"涨停封板({bar.symbol}@{up:.2f})")
            if (
                side == "sell"
                and bar.high <= down * (1 + 1e-6)
                and bar.low >= down * (1 - 1e-6)
            ):
                if bar.close <= down * (1 + 1e-6):
                    return CheckResult(
                        False, reason=f"跌停封板({bar.symbol}@{down:.2f})"
                    )

    # 4. 资金(买入):粗略校验 qty*ref_price <= cash(精确由 Account 兜底)
    if side == "buy":
        need = D(qty) * D(ref_price)
        if need > account_cash:
            # 尝试按可用资金缩减到最大可买手数
            if account_cash <= 0:
                return CheckResult(
                    False, reason=f"资金不足(需 {need},有 {account_cash})"
                )
            max_qty = (
                floor_shares(account_cash / D(ref_price), lot)
                if lot > 1
                else (account_cash / D(ref_price))
            )
            if max_qty <= 0:
                return CheckResult(False, reason="资金不足,无法买到最小手")
            return CheckResult(True, adjusted_qty=max_qty, reason="资金部分成交")

    # 5. 成交量上限(A11):单笔不超过当日成交量 × max_volume_share
    if constraints.max_volume_share > 0 and bar.volume > 0:
        cap = D(str(bar.volume)) * D(str(constraints.max_volume_share))
        if qty > cap:
            qty = floor_shares(cap, lot) if side == "buy" and lot > 1 else q_shares(cap)
            if qty <= 0:
                return CheckResult(False, reason="超过成交量上限")
            return CheckResult(True, adjusted_qty=qty, reason="成交量上限缩减")

    return CheckResult(True, adjusted_qty=qty)

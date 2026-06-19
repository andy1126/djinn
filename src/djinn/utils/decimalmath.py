"""Decimal 辅助:统一精度上下文与四舍五入策略。

设计原则:
- 现金账本、持仓股数、单笔成交金额、手续费一律 :class:`decimal.Decimal`。
- 会计运算必须可复现:固定 ``QUANT_CONTEXT`` 精度上下文,统一 ROUND_HALF_EVEN
  (银行家舍入,与 numpy/pandas 默认一致,避免系统性偏差)。
- 货币最小单位精度由 ``MONEY_QUANT`` / ``SHARES_QUANT`` 控制,按市场可配。
"""

from __future__ import annotations

from decimal import (
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    localcontext,
)
from typing import Final

# 统一精度上下文:28 位有效数字(IEEE 754 decimal64 扩展),消除累加漂移。
QUANT_CONTEXT: Final[Context] = Context(prec=28, rounding=ROUND_HALF_EVEN)

# 货币最小单位:分(CNY/USD/HKD 均到 0.01)。
MONEY_QUANT: Final[Decimal] = Decimal("0.01")
# 股数:支持碎股时到 0.0001 股(A 股最小手 100 股向下取整在约束层处理)。
SHARES_QUANT: Final[Decimal] = Decimal("0.0001")
# 比率/权重:4 位小数(0.01% 精度足够)。
RATIO_QUANT: Final[Decimal] = Decimal("0.0001")

_ZERO: Final[Decimal] = Decimal(0)


def D(value: int | float | str | Decimal) -> Decimal:
    """构造 Decimal,浮点先经 str 转换以避免二进制漂移。"""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def q_money(value: Decimal) -> Decimal:
    """按货币精度量化(到分)。"""
    with localcontext(QUANT_CONTEXT):
        return value.quantize(MONEY_QUANT, rounding=ROUND_HALF_EVEN)


def q_shares(value: Decimal) -> Decimal:
    """按股数精度量化。"""
    with localcontext(QUANT_CONTEXT):
        return value.quantize(SHARES_QUANT, rounding=ROUND_HALF_EVEN)


def q_ratio(value: Decimal) -> Decimal:
    """按比率精度量化。"""
    with localcontext(QUANT_CONTEXT):
        return value.quantize(RATIO_QUANT, rounding=ROUND_HALF_EVEN)


def floor_shares(value: Decimal, lot: int) -> Decimal:
    """按最小手 ``lot`` 向下取整股数(A 股/ETF 100 股/份;美股 lot=1)。"""
    if lot <= 1:
        return q_shares(value)
    with localcontext(QUANT_CONTEXT):
        units = (value / Decimal(lot)).to_integral_value(rounding=ROUND_DOWN)
        return q_shares(units * Decimal(lot))


def to_float(value: Decimal | float) -> float:
    """Decimal → float,用于指标 / 净值序列。"""
    if isinstance(value, Decimal):
        return float(value)
    return value


def is_zero(value: Decimal) -> bool:
    """判定 Decimal 是否数值为零(忽略符号与精度表示)。"""
    return value == _ZERO

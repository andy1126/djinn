"""佣金 / 印花税 / 过户费模型。

按市场差异化:
- A 股:佣金率(万分之几)+ 最低佣金 5 元 + 卖出印花税 1‰ + 沪市过户费 0.001‰;
- 港股:佣金率 + 最低佣金 + 印花税(双边 1.3‰)+ 交易费;
- 美股:每股佣金或按比例 + 最低佣金。

默认 ConservativeCommissionModel 提供按比例 + 最低佣金的通用模型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from djinn.data.schema import Market
from djinn.utils.decimalmath import D, q_money


class CommissionModel(ABC):
    """佣金模型基类。"""

    @abstractmethod
    def cost(self, side: str, price: Decimal | float, qty: Decimal | float) -> Decimal:
        """计算单笔成交佣金(Decimal,已量化到分)。"""


class ConservativeCommissionModel(CommissionModel):
    """通用比例 + 最低佣金模型。

    Args:
        rate: 佣金费率(如 0.0003 = 万三)。
        min_commission: 单笔最低佣金。
        stamp_duty_rate: 印花税率(仅卖出,如 0.001)。
        transfer_fee_rate: 过户费率(双边,如 0.00001)。
    """

    def __init__(
        self,
        rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.0,
        transfer_fee_rate: float = 0.0,
    ) -> None:
        self.rate = D(rate)
        self.min_commission = D(min_commission)
        self.stamp_duty_rate = D(stamp_duty_rate)
        self.transfer_fee_rate = D(transfer_fee_rate)

    def cost(self, side: str, price: Decimal | float, qty: Decimal | float) -> Decimal:
        amount = D(qty) * D(price)
        # 佣金 = max(amount * rate, min_commission)
        commission = max(amount * self.rate, self.min_commission)
        # 印花税(仅卖出)
        stamp = amount * self.stamp_duty_rate if side == "sell" else D(0)
        # 过户费(双边)
        transfer = amount * self.transfer_fee_rate
        return q_money(commission + stamp + transfer)


class ChinaCommissionModel(ConservativeCommissionModel):
    """A 股佣金模型(默认:万三佣金 + 5 元最低 + 卖出 1‰ 印花税 + 沪市过户费)。"""

    def __init__(
        self,
        rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.001,
        transfer_fee_rate: float = 0.00001,
    ) -> None:
        super().__init__(rate, min_commission, stamp_duty_rate, transfer_fee_rate)


class USCommissionModel(ConservativeCommissionModel):
    """美股佣金模型(默认:每股 0.005 美元 + 最低 1 美元,无印花税)。

    Note:
        简化为按比例 + 最低佣金(Interactive Brokers 固定定价近似)。
    """

    def __init__(self, rate: float = 0.0005, min_commission: float = 1.0) -> None:
        super().__init__(
            rate=rate,
            min_commission=min_commission,
            stamp_duty_rate=0.0,
            transfer_fee_rate=0.0,
        )


class HKCommissionModel(ConservativeCommissionModel):
    """港股佣金模型(双边印花税 1.3‰ + 佣金率 + 最低佣金)。"""

    def __init__(
        self,
        rate: float = 0.0005,
        min_commission: float = 30.0,
        stamp_duty_rate: float = 0.0013,
    ) -> None:
        super().__init__(
            rate=rate,
            min_commission=min_commission,
            stamp_duty_rate=stamp_duty_rate,
            transfer_fee_rate=0.0,
        )


def make_commission(market: Market, **overrides: float) -> CommissionModel:
    """按市场构造默认佣金模型,可用 overrides 覆盖字段。"""
    if market is Market.CN:
        return ChinaCommissionModel(**overrides)
    if market is Market.HK:
        return HKCommissionModel(**overrides)
    return USCommissionModel(**overrides)

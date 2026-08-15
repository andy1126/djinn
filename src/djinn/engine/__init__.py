"""djinn.engine — 事件驱动引擎:撮合 / 费用 / 滑点 / 约束 / 订单执行。"""

from __future__ import annotations

from djinn.engine.broker import Broker, orders_from_intents
from djinn.engine.commission import (
    ChinaCommissionModel,
    CommissionModel,
    ConservativeCommissionModel,
    HKCommissionModel,
    USCommissionModel,
    make_commission,
)
from djinn.engine.constraints import TradeConstraints, check_constraints, limit_prices
from djinn.engine.event_engine import BacktestResult, EngineConfig, EventDrivenEngine
from djinn.engine.events import Fill, Order, Rejection
from djinn.engine.slippage import (
    FixedBpsSlippage,
    RandomSlippage,
    SlippageModel,
    VolumeShareSlippage,
    ZeroSlippage,
    make_slippage,
)

__all__ = [
    "BacktestResult",
    "Broker",
    "ChinaCommissionModel",
    "CommissionModel",
    "ConservativeCommissionModel",
    "EngineConfig",
    "EventDrivenEngine",
    "Fill",
    "FixedBpsSlippage",
    "HKCommissionModel",
    "Order",
    "RandomSlippage",
    "Rejection",
    "SlippageModel",
    "TradeConstraints",
    "USCommissionModel",
    "VolumeShareSlippage",
    "ZeroSlippage",
    "check_constraints",
    "limit_prices",
    "make_commission",
    "make_slippage",
    "orders_from_intents",
]

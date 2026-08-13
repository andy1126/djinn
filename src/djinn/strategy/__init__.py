"""djinn.strategy — 策略层:ABC、Context、声明式参数、信号、内置库。"""

from __future__ import annotations

from djinn.strategy.base import (
    SCOPE_PER_SYMBOL,
    SCOPE_PORTFOLIO,
    Context,
    DataView,
    PortfolioView,
    SignalAdapter,
    Strategy,
    param,
)

# 内置策略库便捷导出
from djinn.strategy.library import (
    DCA,
    STRATEGY_REGISTRY,
    BollingerReversion,
    BuyAndHold,
    CrossSectionalMomentum,
    DualMomentum,
    Grid,
    MACDCrossover,
    MACrossover,
    Momentum,
    PairsSpread,
    RSIReversal,
    StochasticCross,
    Supertrend,
    TurtleATR,
    VolTarget,
    get_strategy_class,
)
from djinn.strategy.parameter import Parameter, ParamSchema, get_params, param_schema
from djinn.strategy.signal import OrderIntent, Signal, SignalBatch, SignalValue

__all__ = [
    "Strategy",
    "Context",
    "DataView",
    "PortfolioView",
    "SignalAdapter",
    "param",
    "Parameter",
    "ParamSchema",
    "param_schema",
    "get_params",
    "Signal",
    "SignalValue",
    "SignalBatch",
    "OrderIntent",
    "SCOPE_PER_SYMBOL",
    "SCOPE_PORTFOLIO",
    # 内置策略
    "MACrossover",
    "RSIReversal",
    "Momentum",
    "DCA",
    "MACDCrossover",
    "BollingerReversion",
    "Supertrend",
    "StochasticCross",
    "BuyAndHold",
    "CrossSectionalMomentum",
    "DualMomentum",
    "TurtleATR",
    "Grid",
    "PairsSpread",
    "VolTarget",
    "STRATEGY_REGISTRY",
    "get_strategy_class",
]

"""djinn.strategy.library — 内置策略库。"""

from __future__ import annotations

from djinn.strategy.base import Strategy
from djinn.strategy.library.dca import DCA
from djinn.strategy.library.ma_crossover import MACrossover
from djinn.strategy.library.momentum import Momentum
from djinn.strategy.library.rsi_reversal import RSIReversal, rsi

# 策略名 → 类的注册表(供 CLI / 配置按名实例化)。
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "MACrossover": MACrossover,
    "RSIReversal": RSIReversal,
    "Momentum": Momentum,
    "DCA": DCA,
}


def get_strategy_class(name: str) -> type[Strategy]:
    """按名取策略类(找不到抛 KeyError)。"""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"未知策略 {name!r},可用: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name]


__all__ = [
    "DCA",
    "STRATEGY_REGISTRY",
    "MACrossover",
    "Momentum",
    "RSIReversal",
    "get_strategy_class",
    "rsi",
]

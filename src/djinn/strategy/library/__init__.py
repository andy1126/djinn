"""djinn.strategy.library — 内置策略库 + 统一解析(内置 + 用户自定义)。"""

from __future__ import annotations

from djinn.indicators import rsi
from djinn.strategy.base import Strategy
from djinn.strategy.library.dca import DCA
from djinn.strategy.library.factor_portfolio import FactorPortfolioStrategy
from djinn.strategy.library.ma_crossover import MACrossover
from djinn.strategy.library.momentum import Momentum
from djinn.strategy.library.rsi_reversal import RSIReversal
from djinn.strategy.store import StrategyStore, get_strategy_store
from djinn.strategy.user import compile_user_strategy

# 策略名 → 类的注册表(供 CLI / 配置按名实例化)。
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "MACrossover": MACrossover,
    "RSIReversal": RSIReversal,
    "Momentum": Momentum,
    "DCA": DCA,
    "FactorPortfolio": FactorPortfolioStrategy,
}


def get_strategy_class(name: str, store: StrategyStore | None = None) -> type[Strategy]:
    """按名取策略类:先查内置注册表,再查用户策略存储(动态编译)。

    找不到抛 KeyError。``store`` 缺省用进程内单例(供 CLI / 后台 job)。
    """
    if name in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[name]
    store = store or get_strategy_store()
    rec = store.get_by_name(name)
    if rec is not None:
        return compile_user_strategy(rec.name, rec.source_code, rec.kind)
    raise KeyError(f"未知策略 {name!r},可用内置: {list(STRATEGY_REGISTRY)}")


__all__ = [
    "DCA",
    "STRATEGY_REGISTRY",
    "FactorPortfolioStrategy",
    "MACrossover",
    "Momentum",
    "RSIReversal",
    "get_strategy_class",
    "rsi",
]

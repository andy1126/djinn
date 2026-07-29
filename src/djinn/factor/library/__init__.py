"""djinn.factor.library — 内置因子库(每因子一文件,注册进 FACTOR_REGISTRY)。"""

from __future__ import annotations

from djinn.factor.base import Factor
from djinn.factor.library.growth import ProfitGrowthFactor, RevenueGrowthFactor
from djinn.factor.library.liquidity import TurnoverFactor
from djinn.factor.library.momentum import MomentumFactor, ReversalFactor
from djinn.factor.library.quality import GrossMarginFactor, ROEFactor
from djinn.factor.library.size import SizeFactor
from djinn.factor.library.value import BPFactor, EPFactor, SPFactor
from djinn.factor.library.volatility import BetaFactor, VolatilityFactor

# 因子名 → 类的注册表(供 CLI / API 按名实例化)。
FACTOR_REGISTRY: dict[str, type[Factor]] = {
    "momentum": MomentumFactor,
    "reversal": ReversalFactor,
    "volatility": VolatilityFactor,
    "beta": BetaFactor,
    "turnover": TurnoverFactor,
    "ep": EPFactor,
    "bp": BPFactor,
    "sp": SPFactor,
    "roe": ROEFactor,
    "gross_margin": GrossMarginFactor,
    "revenue_yoy": RevenueGrowthFactor,
    "profit_yoy": ProfitGrowthFactor,
    "size": SizeFactor,
}


def get_factor_class(name: str) -> type[Factor]:
    """按名取因子类(找不到抛 KeyError)。"""
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"未知因子 {name!r},可用: {list(FACTOR_REGISTRY)}")
    return FACTOR_REGISTRY[name]


def make_factor(name: str, **params: object) -> Factor:
    """按名 + 参数实例化因子。"""
    return get_factor_class(name)(**params)


__all__ = [
    "BPFactor",
    "BetaFactor",
    "EPFactor",
    "FACTOR_REGISTRY",
    "GrossMarginFactor",
    "MomentumFactor",
    "ProfitGrowthFactor",
    "ROEFactor",
    "ReversalFactor",
    "RevenueGrowthFactor",
    "SPFactor",
    "SizeFactor",
    "TurnoverFactor",
    "VolatilityFactor",
    "get_factor_class",
    "make_factor",
]

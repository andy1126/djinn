"""djinn.factor — 因子层:因子引擎、预处理、内置因子库、因子分析。"""

from __future__ import annotations

from djinn.factor.base import Factor
from djinn.factor.engine import FactorEngine, FactorPanel
from djinn.factor.library import FACTOR_REGISTRY, get_factor_class, make_factor
from djinn.factor.preprocess import neutralize, standardize, winsorize

__all__ = [
    "FACTOR_REGISTRY",
    "Factor",
    "FactorEngine",
    "FactorPanel",
    "get_factor_class",
    "make_factor",
    "neutralize",
    "standardize",
    "winsorize",
]

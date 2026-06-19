"""权重分配:等权 / 市值加权 / 自定义权重。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from djinn.utils.exceptions import StrategyError

AllocationType = Literal["equal", "market_cap", "custom"]


class Allocation(ABC):
    """分配策略基类:给定标的列表与上下文,产出目标权重 {symbol: weight}。"""

    @abstractmethod
    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """返回归一化目标权重(和为 1,空列表返回 {})。"""


class EqualWeight(Allocation):
    """等权分配:每个成分 1/N。"""

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        n = len(symbols)
        if n == 0:
            return {}
        w = 1.0 / n
        return dict.fromkeys(symbols, w)


class MarketCapWeight(Allocation):
    """市值加权:按最新价 * 流通股(此处用最新价代理,缺流通股数据时退化为等权)。"""

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not symbols or not prices:
            return EqualWeight().target_weights(symbols, ctx, prices)
        caps = {s: max(prices.get(s, 0.0), 0.0) for s in symbols}
        total = sum(caps.values())
        if total <= 0:
            return EqualWeight().target_weights(symbols, ctx, prices)
        return {s: caps[s] / total for s in symbols}


class CustomWeight(Allocation):
    """自定义权重:按用户给定 dict 归一化。"""

    def __init__(self, weights: dict[str, float]) -> None:
        # 校验非负
        for s, w in weights.items():
            if w < 0:
                raise StrategyError(f"自定义权重不能为负:{s}={w}")
        self._raw = dict(weights)

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        sub = {s: self._raw.get(s, 0.0) for s in symbols}
        total = sum(sub.values())
        if total <= 0:
            return dict.fromkeys(symbols, 0.0)
        return {s: sub[s] / total for s in symbols}


def make_allocation(
    kind: AllocationType, weights: dict[str, float] | None = None
) -> Allocation:
    """工厂:按类型字符串构造分配器。"""
    if kind == "equal":
        return EqualWeight()
    if kind == "market_cap":
        return MarketCapWeight()
    if kind == "custom":
        if not weights:
            raise StrategyError("custom 分配需要 weights 字典")
        return CustomWeight(weights)
    raise StrategyError(f"未知分配类型: {kind}")


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """归一化权重(和为 1)。"""
    total = sum(max(w, 0.0) for w in weights.values())
    if total <= 0:
        return dict.fromkeys(weights, 0.0)
    return {s: max(w, 0.0) / total for s, w in weights.items()}

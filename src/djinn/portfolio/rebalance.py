"""再平衡:按周期(日/周/月/季/年)与权重偏离阈值触发调仓。

支持两种触发(满足任一即调仓,可组合):
- 周期:距上次再平衡超过 ``period`` 对应交易日数;
- 阈值:任一成分当前权重与目标偏离 > ``threshold``。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from djinn.portfolio.allocation import Allocation
from djinn.strategy.signal import OrderIntent, Side
from djinn.utils.exceptions import StrategyError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

RebalancePeriod = Literal["none", "daily", "weekly", "monthly", "quarterly", "yearly"]

_PERIOD_DAYS: dict[str, int] = {
    "none": 0,
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
    "yearly": 252,
}


@dataclass
class RebalanceConfig:
    """再平衡配置。"""

    period: RebalancePeriod = "none"
    threshold: float = 0.0  # 权重偏离阈值(0 = 不用阈值触发)
    min_hold_days: int = 0  # 最小持有交易日(防止频繁调仓)

    def validate(self) -> None:
        if self.period not in _PERIOD_DAYS:
            raise StrategyError(f"未知再平衡周期: {self.period}")
        if not 0.0 <= self.threshold <= 1.0:
            raise StrategyError(f"threshold 必须在 [0,1],实际 {self.threshold}")


class Rebalancer:
    """组合再平衡器。

    与 :class:`Allocation` 配合:在触发日产出"调到目标权重"的 OrderIntent 列表。
    """

    def __init__(self, config: RebalanceConfig | None = None) -> None:
        self.config = config or RebalanceConfig()
        self.config.validate()
        self._last_rebalance: date | None = None
        self._bars_since_last = 0

    def maybe_rebalance(
        self,
        ts: date,
        symbols: list[str],
        allocation: Allocation,
        current_weights: dict[str, float],
        prices: dict[str, float] | None = None,
    ) -> list[OrderIntent]:
        """判定是否触发再平衡;触发则返回调仓订单,否则空列表。"""
        self._bars_since_last += 1
        triggered = self._triggered(ts, symbols, allocation, current_weights, prices)
        if not triggered:
            return []
        self._last_rebalance = ts
        self._bars_since_last = 0
        return self._build_orders(ts, symbols, allocation, current_weights, prices)

    def _triggered(
        self,
        ts: date,
        symbols: list[str],
        allocation: Allocation,
        current_weights: dict[str, float],
        prices: dict[str, float] | None,
    ) -> bool:
        cfg = self.config
        # 周期触发
        if cfg.period != "none":
            needed = _PERIOD_DAYS[cfg.period]
            if cfg.min_hold_days:
                needed = max(needed, cfg.min_hold_days)
            if self._bars_since_last >= needed:
                return True
        # 阈值触发
        if cfg.threshold > 0:
            target = allocation.target_weights(symbols, prices=prices)
            for s in set(target) | set(current_weights):
                if (
                    abs(target.get(s, 0.0) - current_weights.get(s, 0.0))
                    > cfg.threshold
                ):
                    return True
        return False

    def _build_orders(
        self,
        ts: date,
        symbols: list[str],
        allocation: Allocation,
        current_weights: dict[str, float],
        prices: dict[str, float] | None,
    ) -> list[OrderIntent]:
        target = allocation.target_weights(symbols, prices=prices)
        orders: list[OrderIntent] = []
        for s in symbols:
            cur = current_weights.get(s, 0.0)
            tgt = target.get(s, 0.0)
            if abs(tgt - cur) < 1e-6:
                continue
            side: Side = "buy" if tgt > cur else "sell"
            orders.append(
                OrderIntent(
                    symbol=s,
                    side=side,
                    target_percent=tgt,
                    created_ts=ts,
                    tag="rebalance",
                )
            )
        return orders

    def reset(self) -> None:
        self._last_rebalance = None
        self._bars_since_last = 0

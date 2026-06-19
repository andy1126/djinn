"""风控:仓位 / 单票 / 行业集中度限制(Phase 1 基础版)。

Phase 1 实现:最大单票权重、最大总仓位(预留现金下限)。
行业集中度依赖 provider 行业分类(Phase 2 补充),缺失时降级告警。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from djinn.strategy.signal import OrderIntent
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class RiskLimits:
    """风控阈值。"""

    max_single_weight: float = 1.0  # 单票最大权重 [0,1]
    max_total_position: float = 1.0  # 最大总仓位(1.0 = 满仓,0.9 = 至少 10% 现金)
    max_sector_weight: float | None = None  # 行业集中度(Phase 2)
    sector_map: dict[str, str] = field(default_factory=dict)  # symbol -> 行业


class RiskManager:
    """对策略产出的订单意图做风控过滤。

    Phase 1:对 ``target_percent`` 订单截断单票权重上限,并限制总仓位。
    数量订单(size)按"事后校验"——交由 Broker 资金校验兜底。
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        if self.limits.max_sector_weight is not None and not self.limits.sector_map:
            _log.warning("设置了 max_sector_weight 但无 sector_map,行业限制将降级跳过")

    def filter(
        self,
        orders: list[OrderIntent],
        current_weights: dict[str, float],
    ) -> list[OrderIntent]:
        """过滤 / 截断订单意图。"""
        if not orders:
            return orders
        max_single = self.limits.max_single_weight
        out: list[OrderIntent] = []
        for o in orders:
            if o.is_target():
                tp = o.target_percent or 0.0
                if tp > max_single:
                    _log.debug(
                        "风控截断 %s 目标权重 %.4f → %.4f", o.symbol, tp, max_single
                    )
                    o = OrderIntent(
                        symbol=o.symbol,
                        side=o.side,
                        target_percent=max_single,
                        created_ts=o.created_ts,
                        tag=(o.tag + " risk_capped").strip(),
                    )
            out.append(o)
        # 总仓位限制:若所有目标权重之和超过 max_total_position,等比缩放
        self._enforce_total_position(out)
        return out

    def _enforce_total_position(self, orders: list[OrderIntent]) -> None:
        cap = self.limits.max_total_position
        if cap >= 1.0:
            return
        total = sum(
            o.target_percent or 0.0
            for o in orders
            if o.is_target() and (o.target_percent or 0) > 0
        )
        if total <= cap:
            return
        scale = cap / total
        for o in orders:
            if o.is_target() and (o.target_percent or 0) > 0:
                o.target_percent = (o.target_percent or 0.0) * scale
        _log.debug("总仓位限制:目标缩放 %.4f → 满足上限 %.4f", total, cap)

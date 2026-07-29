"""风控:单票 / 总仓位 / 行业集中度 / 换手限制。

- ``max_single_weight``:截断单票目标权重上限;
- ``max_total_position``:目标权重之和超限时等比缩放(预留现金下限);
- ``max_sector_weight``:按 ``sector_map`` 聚合行业权重,超上限的行业内订单等比缩放
  (行业分类由 provider 提供,见 Phase 0 ``universe`` / ``get_industry_map``);
- ``max_turnover``:限制单次调仓换手(Σ|目标权重 − 当前权重|),超限时把所有目标权重
  向当前权重回缩,使总换手降到上限内。
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
    max_sector_weight: float | None = None  # 单行业最大权重 [0,1](需 sector_map)
    sector_map: dict[str, str] = field(default_factory=dict)  # symbol -> 行业
    max_turnover: float | None = None  # 单次调仓最大换手(Σ|Δweight| 上限)


class RiskManager:
    """对策略产出的订单意图做风控过滤。

    对 ``target_percent`` 订单依次施加:单票上限 → 行业集中度 → 总仓位 → 换手限制。
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
        self._enforce_sector_cap(out, current_weights)
        self._enforce_total_position(out)
        self._enforce_turnover(out, current_weights)
        return out

    def _enforce_sector_cap(
        self, orders: list[OrderIntent], current_weights: dict[str, float]
    ) -> None:
        """行业集中度:超上限行业内的目标订单等比缩回上限。"""
        cap = self.limits.max_sector_weight
        smap = self.limits.sector_map
        if cap is None or cap <= 0 or not smap:
            return
        # 目标权重 = 当前权重,再被目标订单覆盖
        target = dict(current_weights)
        for o in orders:
            if o.is_target():
                target[o.symbol] = o.target_percent or 0.0
        sector_weight: dict[str, float] = {}
        for sym, w in target.items():
            sec = smap.get(sym)
            if sec is not None and w > 0:
                sector_weight[sec] = sector_weight.get(sec, 0.0) + w
        over = {sec: w / cap for sec, w in sector_weight.items() if w > cap}
        if not over:
            return
        for o in orders:
            if not o.is_target():
                continue
            sec = smap.get(o.symbol)
            if sec is not None and sec in over:
                scaled = (o.target_percent or 0.0) / over[sec]
                _log.debug(
                    "行业上限 %s(%.4f>%.4f):%s %.4f → %.4f",
                    sec,
                    sector_weight[sec],
                    cap,
                    o.symbol,
                    o.target_percent or 0.0,
                    scaled,
                )
                o.target_percent = scaled

    def _enforce_total_position(self, orders: list[OrderIntent]) -> None:
        """总仓位限制:目标权重之和超上限时等比缩放。"""
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

    def _enforce_turnover(
        self, orders: list[OrderIntent], current_weights: dict[str, float]
    ) -> None:
        """换手限制:Σ|目标 − 当前| 超上限时,把所有目标权重向当前权重回缩。"""
        cap = self.limits.max_turnover
        if cap is None or cap < 0:
            return
        turnover = sum(
            abs((o.target_percent or 0.0) - current_weights.get(o.symbol, 0.0))
            for o in orders
            if o.is_target()
        )
        if turnover <= cap or turnover <= 0:
            return
        scale = cap / turnover
        for o in orders:
            if o.is_target():
                cur = current_weights.get(o.symbol, 0.0)
                tgt = o.target_percent or 0.0
                o.target_percent = cur + (tgt - cur) * scale
        _log.debug("换手限制:换手 %.4f → 上限 %.4f(缩放 %.3f)", turnover, cap, scale)

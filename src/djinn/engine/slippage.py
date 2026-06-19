"""滑点模型:成交价相对参考价的偏移。

- :class:`FixedBpsSlippage`:固定 bps 偏移(买入加价、卖出降价)。
- :class:`ZeroSlippage`:无滑点(确定性回测/测试)。
- :class:`VolumeShareSlippage`:按成交量占比影响价格(Phase 2 强化)。
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from djinn.data.schema import Bar
from djinn.engine.events import Fill as _Fill  # noqa: F401 - 仅供类型参考


class SlippageModel(ABC):
    """滑点模型基类。"""

    @abstractmethod
    def fill_price(self, side: str, ref_price: float, bar: Bar) -> float:
        """返回考虑滑点后的成交价。"""


@dataclass
class ZeroSlippage(SlippageModel):
    """无滑点(按参考价成交)。"""

    def fill_price(self, side: str, ref_price: float, bar: Bar) -> float:
        return ref_price


@dataclass
class FixedBpsSlippage(SlippageModel):
    """固定 bps 滑点:买入价 = ref*(1+bps),卖出价 = ref*(1-bps)。"""

    bps: float = 5.0  # 5 bps = 0.05%

    def fill_price(self, side: str, ref_price: float, bar: Bar) -> float:
        mult = 1 + self.bps / 10000.0 if side == "buy" else 1 - self.bps / 10000.0
        return ref_price * mult


@dataclass
class RandomSlippage(SlippageModel):
    """随机滑点(均匀 [0, bps],供蒙特卡洛)。需传入 random.Random 实例以保证可复现。"""

    bps: float = 5.0
    rng: random.Random | None = None

    def fill_price(self, side: str, ref_price: float, bar: Bar) -> float:
        r = self.rng or random
        bps = r.uniform(0, self.bps)
        mult = 1 + bps / 10000.0 if side == "buy" else 1 - bps / 10000.0
        return ref_price * mult


@dataclass
class VolumeShareSlippage(SlippageModel):
    """成交量占比滑点:订单量占 bar 成交量比例越高,滑点越大(线性)。"""

    max_bps: float = 10.0
    volume_impact: float = 0.1  # 订单量 / bar 量

    def fill_price(
        self, side: str, ref_price: float, bar: Bar, order_qty: float = 0.0
    ) -> float:
        vol = max(bar.volume, 1.0)
        share = min(order_qty / vol, 1.0)
        bps = self.max_bps * share * self.volume_impact * 10  # 缩放
        mult = 1 + bps / 10000.0 if side == "buy" else 1 - bps / 10000.0
        return ref_price * mult


def make_slippage(kind: str, **kwargs: Any) -> SlippageModel:
    """工厂:按类型字符串构造滑点模型。"""
    kind = kind.lower()
    if kind in ("zero", "none"):
        return ZeroSlippage()
    if kind in ("fixed_bps", "fixed"):
        return FixedBpsSlippage(**kwargs)
    if kind == "random":
        return RandomSlippage(**kwargs)
    if kind in ("volume_share", "volume"):
        return VolumeShareSlippage(**kwargs)
    raise ValueError(f"未知滑点类型: {kind}")


def fill_price(
    model: SlippageModel, side: str, ref_price: float, bar: Bar, order_qty: float = 0.0
) -> float:
    """统一调用入口(VolumeShareSlippage 需要 order_qty)。"""
    if isinstance(model, VolumeShareSlippage):
        return model.fill_price(side, ref_price, bar, order_qty=order_qty)
    return model.fill_price(side, ref_price, bar)

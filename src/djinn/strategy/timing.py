"""择时规则库:市场闸门 / 个股出场 / 入场确认(供组合策略叠加)。

所有规则以增量 deque 缓冲维护状态(O(1)/标的/日),只吃历史 append 数据,
天然无未来函数。t 日判断、t+1 成交由引擎撮合保证。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MarketRegimeFilter:
    """指数 SMA 闸门:收盘 < SMA(window) → 仓位上限 floor,否则 1.0。"""

    window: int = 200
    floor: float = 0.0
    _closes: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        # 缓冲随 window 自适应:旧实现 maxlen 硬编码 210,window>210 时
        # ``len < window`` 恒成立 → 闸门静默失效(永远放行)。
        self._closes = deque(maxlen=int(self.window) + 10)

    def update(self, close: float | None) -> None:
        if close is not None and close > 0:
            self._closes.append(close)

    def exposure_cap(self) -> float:
        if len(self._closes) < self.window:
            return 1.0  # 暖机期放行
        sma = float(np.mean(list(self._closes)[-self.window :]))
        return 1.0 if self._closes[-1] > sma else self.floor


class ExitRule:
    """个股出场规则基类。"""

    def update(self, sym: str, o: float, h: float, lo: float, c: float) -> None: ...
    def should_exit(self, sym: str) -> bool:
        return False

    def arm(self, sym: str, price: float) -> None: ...
    def disarm(self, sym: str) -> None: ...


@dataclass
class SMABreakExit(ExitRule):
    """收盘跌破 SMA(window) → 出场。无 arm 状态。"""

    window: int = 20
    _closes: dict[str, deque[float]] = field(default_factory=dict)

    def update(self, sym: str, o: float, h: float, lo: float, c: float) -> None:
        buf = self._closes.setdefault(sym, deque(maxlen=self.window + 5))
        buf.append(c)

    def should_exit(self, sym: str) -> bool:
        buf = self._closes.get(sym)
        if not buf or len(buf) < self.window:
            return False
        return bool(buf[-1] < float(np.mean(list(buf)[-self.window :])))


@dataclass
class ATRTrailingExit(ExitRule):
    """吊灯止损:收盘 < peak − mult × ATR(window);peak 自 arm 起跟踪最高价。"""

    mult: float = 3.0
    window: int = 14
    _bars: dict[str, deque[tuple[float, float, float]]] = field(default_factory=dict)
    _peak: dict[str, float] = field(default_factory=dict)

    def update(self, sym: str, o: float, h: float, lo: float, c: float) -> None:
        buf = self._bars.setdefault(sym, deque(maxlen=self.window + 10))
        buf.append((h, lo, c))
        if sym in self._peak:
            self._peak[sym] = max(self._peak[sym], h)

    def arm(self, sym: str, price: float) -> None:
        self._peak[sym] = price

    def disarm(self, sym: str) -> None:
        self._peak.pop(sym, None)

    def should_exit(self, sym: str) -> bool:
        if sym not in self._peak:
            return False
        buf = self._bars.get(sym)
        if not buf or len(buf) < self.window + 1:
            return False
        rows = list(buf)
        trs = [
            max(h - lo, abs(h - pc), abs(lo - pc))
            for (h, lo, _c), (_, _, pc) in zip(rows[1:], rows[:-1], strict=False)
        ]
        atr = float(np.mean(trs[-self.window :]))
        return bool(rows[-1][2] < self._peak[sym] - self.mult * atr)


@dataclass
class AboveSMAConfirm:
    """入场确认:收盘站上 SMA(window) 才允许买入;数据不足不拦截。"""

    window: int = 20

    def entry_ok(self, closes: pd.Series) -> bool:
        if closes is None or len(closes) < self.window:
            return True
        return float(closes.iloc[-1]) > float(closes.iloc[-self.window :].mean())

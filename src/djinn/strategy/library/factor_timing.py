"""因子选股(调仓频)+ 指标择时(日频)两层组合策略。"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from djinn.strategy.base import SCOPE_PORTFOLIO, Context
from djinn.strategy.library.factor_portfolio import FactorPortfolioStrategy
from djinn.strategy.timing import (
    AboveSMAConfirm,
    ATRTrailingExit,
    ExitRule,
    MarketRegimeFilter,
)


class FactorTimingStrategy(FactorPortfolioStrategy):
    scope = SCOPE_PORTFOLIO

    def __init__(
        self,
        *args: Any,
        regime: MarketRegimeFilter | None = None,
        exit_rule: ExitRule | None = None,
        entry_confirm: AboveSMAConfirm | None = None,
        cooldown_days: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._regime = regime
        self._exit = exit_rule
        self._confirm = entry_confirm
        self._cooldown_days = max(0, int(cooldown_days))
        self._pool: list[str] = []
        self._base_w: dict[str, float] = {}
        self._inactive: dict[str, int] = {}  # sym -> 冷却起始 bars_seen

    def on_bar(self, ctx: Context) -> None:
        n = self._bars_seen
        self._bars_seen += 1
        is_rebalance = n % self.rebalance_freq == 0

        # 0. 更新规则缓冲(基准 + 池内 ∪ 持仓)
        if self._regime is not None:
            self._regime.update(ctx.benchmark_close())
        if self._exit is not None:
            syms = set(self._pool) | {
                s for s, p in ctx.portfolio.positions.items() if p.qty > 0
            }
            for sym in syms:
                if sym not in ctx.data:
                    continue
                try:
                    o = ctx.data.latest(sym, "open")
                    h = ctx.data.latest(sym, "high")
                    lo = ctx.data.latest(sym, "low")
                    c = ctx.data.latest(sym, "close")
                except Exception:
                    continue
                self._exit.update(sym, o, h, lo, c)

        # A. 市场闸门
        cap = self._regime.exposure_cap() if self._regime is not None else 1.0

        # B. 调仓日:因子重选池(出池即卖,因子判决优先)
        if is_rebalance:
            selected, weights = self._select_pool(ctx)
            for s, pos in ctx.portfolio.positions.items():
                if pos.qty > 0 and s not in selected:
                    ctx.order_target_percent(s, 0.0)
                    ctx.orders[-1].tag = "rebalance:out"
                    if self._exit is not None:
                        self._exit.disarm(s)
            self._pool, self._base_w = selected, weights
            self._inactive.clear()  # 冷却重置

        # C. 每日:池内出场检查
        if self._exit is not None:
            for s, pos in list(ctx.portfolio.positions.items()):
                if pos.qty <= 0 or s not in self._pool:
                    continue
                if self._exit.should_exit(s):
                    ctx.order_target_percent(s, 0.0)
                    ctx.orders[-1].tag = f"exit:{type(self._exit).__name__}"
                    self._inactive[s] = n
                    self._exit.disarm(s)

        # D. 每日:入场确认(池内未持仓、过冷却、指标确认)
        for s in self._pool:
            if ctx.portfolio.has_position(s):
                continue
            if s in self._inactive and n - self._inactive[s] < self._cooldown_days:
                continue
            if s not in ctx.data:
                continue
            if self._confirm is not None:
                hist = ctx.data.history(s, "close", self._confirm.window + 5)
                if not self._confirm.entry_ok(hist):
                    continue
            w = self._base_w.get(s, 0.0) * cap  # 名义权重 × 闸门;被挡份额留现金
            if w > 0:
                ctx.order_target_percent(s, w)
                # 调仓日买入是因子重选结果 → rebalance:in;非调仓日(冷却后再入场) → entry:cap=*
                ctx.orders[-1].tag = (
                    "rebalance:in" if is_rebalance else f"entry:cap={cap:.2f}"
                )
                if isinstance(self._exit, ATRTrailingExit):
                    with suppress(Exception):
                        self._exit.arm(s, ctx.data.latest(s, "close"))

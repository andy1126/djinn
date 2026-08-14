"""因子组合策略:首个 ``scope=SCOPE_PORTFOLIO`` 策略。

每个调仓日(首个交易日 + 每隔 ``rebalance_freq`` 个交易日)在 ``on_bar(ctx)`` 内:
1. 取 ``ctx.data`` 的 ``<= now`` 行情组 ``date × symbol`` 面板(防未来函数);
2. 逐因子 :meth:`Factor.compute` 取最新截面,按 :class:`FactorScore` 合成综合得分;
3. 选得分最高的 Top ``n_stocks``,按 ``allocation`` 分配目标权重;
4. 通过 ``ctx.order_target_percent`` 调仓(持有但落选者清零)。

基本面因子(EP / BP / ROE 等)需注入 point-in-time 基本面面板(``fundamentals``,
通常由 :class:`~djinn.factor.engine.FactorEngine` 或 runner 预计算);价格 / 量 / 额
因子直接由 ``ctx.data`` 计算,无需注入。
"""

from __future__ import annotations

import pandas as pd

from djinn.data.schema import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_VOLUME,
)
from djinn.factor.base import Factor, Panel, PanelDict
from djinn.portfolio.allocation import Allocation, EqualWeight, estimate_covariance
from djinn.screen.scoring import FactorScore, score_cross_section
from djinn.strategy.base import SCOPE_PORTFOLIO, Context, Strategy
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


class FactorPortfolioStrategy(Strategy):
    """多因子打分 TopN 组合策略。"""

    scope = SCOPE_PORTFOLIO

    def __init__(
        self,
        factors: list[Factor],
        scores: list[FactorScore],
        n_stocks: int = 10,
        rebalance_freq: int = 20,
        allocation: Allocation | None = None,
        fundamentals: PanelDict | None = None,
        preprocess: bool = True,
    ) -> None:
        super().__init__()
        if not factors:
            raise ValueError("FactorPortfolioStrategy 需要至少一个因子")
        if not scores:
            raise ValueError("FactorPortfolioStrategy 需要至少一个打分权重")
        self._factors = list(factors)
        self._scores = list(scores)
        self.n_stocks = max(1, int(n_stocks))
        self.rebalance_freq = max(1, int(rebalance_freq))
        self.allocation = allocation or EqualWeight()
        self._fundamentals = fundamentals or {}
        self.preprocess = preprocess
        self._bars_seen = 0

    def on_bar(self, ctx: Context) -> None:
        n = self._bars_seen
        self._bars_seen += 1
        # 非调仓日直接返回(首日 n=0 必调仓)
        if n % self.rebalance_freq != 0:
            return
        prices, ohlcv = self._visible_panels(ctx)
        if prices.empty:
            return
        fundamentals = self._visible_fundamentals(ctx)
        # D3:截断到最大回看窗口(因子 rolling 只依赖最近 lb 日,截断后末行不变)
        lb = max((getattr(f, "max_lookback", 252) for f in self._factors), default=252)
        cutoff = pd.Timestamp(ctx.now) - pd.Timedelta(days=int(lb * 1.6) + 30)
        prices = prices.loc[prices.index >= cutoff]
        if len(prices) < 2:
            return
        ohlcv = {k: v.loc[v.index >= cutoff] for k, v in ohlcv.items()}
        fundamentals = {k: df.loc[df.index >= cutoff] for k, df in fundamentals.items()}
        # 逐因子取最新截面
        cross: dict[str, pd.Series] = {}
        for f in self._factors:
            try:
                panel = f.compute(prices, ohlcv, fundamentals)
            except Exception as e:
                _log.warning("因子 %s 计算失败 @%s: %s", f.name, ctx.now, e)
                continue
            if len(panel) == 0:
                continue
            cross[f.name] = panel.iloc[-1]
        if not cross:
            return
        cross_df = pd.DataFrame(cross)
        score = score_cross_section(cross_df, self._scores, self.preprocess)
        selected = score.dropna().nlargest(self.n_stocks).index.tolist()
        if not selected:
            return
        last_close = prices.iloc[-1]
        price_map = {
            s: float(last_close[s]) for s in selected if pd.notna(last_close.get(s))
        }
        # 打分 / 协方差供进阶分配器(score / risk_parity / min_variance / mean_variance)
        scores_map = {s: float(score[s]) for s in selected}
        cov = self._selected_cov(prices, selected)
        weights = self.allocation.target_weights(
            selected, prices=price_map, scores=scores_map, cov=cov
        )
        selected_set = set(selected)
        # 调出:当前持有但落选 → 清零
        for s, pos in ctx.portfolio.positions.items():
            if pos.qty > 0 and s not in selected_set:
                ctx.order_target_percent(s, 0.0)
        # 调入 / 调到目标权重
        for s, w in weights.items():
            ctx.order_target_percent(s, w)

    # ── 面板构建(<= now,防未来函数)──────────────────────
    def _visible_panels(self, ctx: Context) -> tuple[Panel, PanelDict]:
        closes: dict[str, pd.Series] = {}
        fields: dict[str, dict[str, pd.Series]] = {
            c: {} for c in (COL_OPEN, COL_HIGH, COL_LOW, COL_VOLUME, COL_AMOUNT)
        }
        for sym in ctx.data.symbols:
            df = ctx.data[sym]
            if len(df) == 0:
                continue
            closes[sym] = df[COL_CLOSE]
            for c in fields:
                if c in df.columns:
                    fields[c][sym] = df[c]
        if not closes:
            return pd.DataFrame(), {}
        prices = pd.DataFrame(closes).sort_index()
        ohlcv: PanelDict = {
            c: pd.DataFrame(s).reindex(prices.index) for c, s in fields.items()
        }
        return prices, ohlcv

    def _visible_fundamentals(self, ctx: Context) -> PanelDict:
        now = pd.Timestamp(ctx.now)
        return {k: df.loc[:now] for k, df in self._fundamentals.items()}

    def _selected_cov(self, prices: Panel, selected: list[str]) -> pd.DataFrame | None:
        """由可见收盘价面板估计选中标的的日收益协方差(不足时返回 None 退化等权)。

        用 complete-case(``dropna(how="any")``)保证协方差良态;样本太少返回 None,
        由分配器退化为等权,保证任何情形下都能产出合法权重。
        """
        cols = [s for s in selected if s in prices.columns]
        if len(cols) < 2:
            return None
        rets = prices[cols].pct_change().dropna(how="any")
        if len(rets) < 2:
            return None
        return estimate_covariance(rets)

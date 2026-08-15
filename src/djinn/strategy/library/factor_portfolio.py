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

from typing import Any

import numpy as np
import pandas as pd

from djinn.data.schema import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_MARKET_CAP,
    COL_OPEN,
    COL_VOLUME,
)
from djinn.factor.base import Factor, Panel, PanelDict
from djinn.factor.composite import composite_score, rolling_ic_weights
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
        weighting: str = "static",
        icir_window: int = 60,
        icir_min_periods: int = 20,
        min_amount: float | None = None,
        min_list_days: int | None = None,
        exclude_st: bool = False,
        names: dict[str, str] | None = None,
        industry_neutral: bool = False,
        industry_map: dict[str, str] | None = None,
        max_sector_weight: float | None = None,
        min_score_diff: float = 0.0,
        neutralize: bool = False,
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
        # C9:因子加权方式(static=手填权重;icir=滚动 ICIR 自动加权,符号自适配方向)
        if weighting not in ("static", "icir"):
            raise ValueError(f"weighting 只支持 static/icir,实际 {weighting!r}")
        self.weighting = weighting
        self.icir_window = max(2, int(icir_window))
        self.icir_min_periods = max(2, int(icir_min_periods))
        self._bars_seen = 0
        # G1~G4:选股流水线增强(全部可选,默认关闭,保持 G0 等价性)
        self.min_amount = min_amount
        self.min_list_days = min_list_days
        self.exclude_st = exclude_st
        self.names = names
        self.industry_neutral = industry_neutral
        self.industry_map = industry_map
        self.max_sector_weight = max_sector_weight
        self.min_score_diff = min_score_diff
        # C5:打分前行业/市值中性化(需 industry_map 或 fundamentals 市值面板)
        self.neutralize = neutralize
        # G9:调仓快照(每次 _select_pool 成功把日期/名单/得分 append 进来,供报告展示)
        self.selection_log: list[dict[str, Any]] = []

    def on_bar(self, ctx: Context) -> None:
        n = self._bars_seen
        self._bars_seen += 1
        # 非调仓日直接返回(首日 n=0 必调仓)
        if n % self.rebalance_freq != 0:
            return
        selected, weights = self._select_pool(ctx)
        if not selected:
            return
        selected_set = set(selected)
        # 调出:当前持有但落选 → 清零
        for s, pos in ctx.portfolio.positions.items():
            if pos.qty > 0 and s not in selected_set:
                ctx.order_target_percent(s, 0.0)
                ctx.orders[-1].tag = "rebalance:out"
        # 调入 / 调到目标权重
        for s, w in weights.items():
            ctx.order_target_percent(s, w)
            ctx.orders[-1].tag = "rebalance:in"

    def _select_pool(self, ctx: Context) -> tuple[list[str], dict[str, float]]:
        """因子打分 → (名单, 名义权重 dict)。

        防未来函数:只吃 ctx.data <= now 截面(经 _visible_panels)。
        返回空名单表示本日无法选股(数据不足),调用方跳过。
        """
        prices, ohlcv = self._visible_panels(ctx)
        if prices.empty:
            return [], {}
        fundamentals = self._visible_fundamentals(ctx)
        # D3:截断到最大回看窗口(因子 rolling 只依赖最近 lb 日,截断后末行不变)
        lb = max((getattr(f, "max_lookback", 252) for f in self._factors), default=252)
        cutoff = pd.Timestamp(ctx.now) - pd.Timedelta(days=int(lb * 1.6) + 30)
        prices = prices.loc[prices.index >= cutoff]
        if len(prices) < 2:
            return [], {}
        ohlcv = {k: v.loc[v.index >= cutoff] for k, v in ohlcv.items()}
        fundamentals = {k: df.loc[df.index >= cutoff] for k, df in fundamentals.items()}
        # C6:因子声明 benchmark 时注入真实基准日收益(替代截面等权代理);基准经
        # 引擎 ctx.benchmark(symbol, DataView) 提供,≤now 切片天然无未来函数。
        if any(getattr(f, "benchmark", None) for f in self._factors):
            bench = getattr(ctx, "benchmark", None)
            if bench is not None:
                sym, view = bench
                try:
                    closes = view.history(sym, "close", lb + 10)
                    if len(closes) >= 2:
                        bench_rets = closes.pct_change().dropna()
                        if len(bench_rets):
                            ohlcv = {**ohlcv, "__benchmark__": bench_rets}
                except Exception:
                    pass  # 基准不可用时退化截面等权代理
        # 打分:C9 icir 用滚动 ICIR 权重合成;否则 static 静态权重(逐因子最新截面)
        if self.weighting == "icir":
            score = self._icir_score(prices, ohlcv, fundamentals)
        else:
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
                return [], {}
            cross_df = pd.DataFrame(cross)
            # C5:neutralize=True 时打分前做行业/市值中性化
            log_mktcap = self._log_mktcap_row(fundamentals) if self.neutralize else None
            score = score_cross_section(
                cross_df,
                self._scores,
                self.preprocess,
                neutralize=self.neutralize,
                industry_map=self.industry_map,
                log_mktcap=log_mktcap,
            )
        score = score.dropna()
        if score.empty:
            return [], {}
        # G1:资格过滤(流动性/次新/ST/停牌),发生在 TopN 之前
        eligible = self._tradable(ctx, list(score.index), prices, ohlcv)
        if not eligible:
            return [], {}
        # G2:行业中性(或全池 TopN)
        if self.industry_neutral and self.industry_map:
            selected = self._pick_neutral(
                score[eligible], self.industry_map, self.n_stocks
            )
        else:
            if self.industry_neutral and not self.industry_map:
                _log.warning("industry_neutral=True 但无行业映射,退化为全池 TopN")
            selected = score[eligible].nlargest(self.n_stocks).index.tolist()
        if not selected:
            return [], {}
        # G4:换手惩罚(得分优势不足不换仓)
        selected = self._apply_turnover_penalty(ctx, selected, score)
        last_close = prices.iloc[-1]
        price_map = {
            s: float(last_close[s]) for s in selected if pd.notna(last_close.get(s))
        }
        scores_map = {s: float(score[s]) for s in selected if s in score.index}
        cov = self._selected_cov(prices, selected)
        weights = self.allocation.target_weights(
            selected, prices=price_map, scores=scores_map, cov=cov
        )
        # G3:行业暴露上限(策略层权重缩放)
        weights = self._apply_sector_cap(weights)
        # G9:记录本次调仓快照(日期/名单/得分),供报告 selection_log 展示
        self.selection_log.append(
            {
                "date": str(ctx.now),
                "selected": list(selected),
                "scores": {s: float(score[s]) for s in selected if s in score.index},
            }
        )
        return selected, weights

    # ── G1:资格过滤 ─────────────────────────────────────
    def _tradable(
        self, ctx: Context, candidates: list[str], prices: Panel, ohlcv: PanelDict
    ) -> list[str]:
        """资格过滤:流动性 / 次新 / ST / 当日停牌(无行情)。"""
        amount = ohlcv.get(COL_AMOUNT)
        last_ts = prices.index[-1]
        out: list[str] = []
        for s in candidates:
            if s not in prices.columns:
                continue
            ser = prices[s].dropna()
            # 当日无行情(union 日历下停牌/未上市)→ 排除
            if len(ser) == 0 or ser.index[-1] < last_ts:
                continue
            # 次新:数据首行距今日不足 min_list_days 个交易日
            if self.min_list_days is not None and len(ser) < self.min_list_days:
                continue
            # 流动性:近 20 日平均成交额
            if (
                self.min_amount is not None
                and amount is not None
                and s in amount.columns
            ):
                amt = amount[s].dropna().iloc[-20:]
                if len(amt) == 0 or float(amt.mean()) < self.min_amount:
                    continue
            # ST:名称含 "ST"
            if self.exclude_st and self.names and "ST" in self.names.get(s, "").upper():
                continue
            out.append(s)
        return out

    # ── G2:行业中性选股 ─────────────────────────────────
    @staticmethod
    def _pick_neutral(
        score: pd.Series, industry_map: dict[str, str], n: int
    ) -> list[str]:
        """行业中性 TopK:每行业 ceil(n/行业数) 名,超额砍尾、欠额全局补位。"""
        groups: dict[str, list[str]] = {}
        for s in score.index:
            groups.setdefault(industry_map.get(s, "未知"), []).append(s)
        k = max(1, -(-n // max(1, len(groups))))  # ceil(n/行业数)
        picked: list[str] = []
        ranked = score.sort_values(ascending=False)
        for _ind, members in groups.items():
            ordered = [s for s in ranked.index if s in members]
            picked.extend(ordered[:k])
        if len(picked) > n:  # 超额:全局按分砍尾
            picked = [s for s in ranked.index if s in picked][:n]
        elif len(picked) < n:  # 欠额:全局剩余按分补足
            rest = [s for s in ranked.index if s not in picked]
            picked.extend(rest[: n - len(picked)])
        return picked

    # ── G3:行业暴露上限 ─────────────────────────────────
    def _apply_sector_cap(self, weights: dict[str, float]) -> dict[str, float]:
        """行业权重超上限等比缩放到上限;腾出的权重留现金(不再分配)。"""
        if self.max_sector_weight is None or not self.industry_map:
            return weights
        cap = self.max_sector_weight
        by_ind: dict[str, float] = {}
        for s, w in weights.items():
            ind = self.industry_map.get(s, "未知")
            by_ind[ind] = by_ind.get(ind, 0.0) + w
        scale = {
            ind: min(1.0, cap / total) if total > cap else 1.0
            for ind, total in by_ind.items()
        }
        if all(v == 1.0 for v in scale.values()):
            return weights
        return {
            s: w * scale.get(self.industry_map.get(s, "未知"), 1.0)
            for s, w in weights.items()
        }

    # ── G4:换手惩罚 ─────────────────────────────────────
    def _apply_turnover_penalty(
        self, ctx: Context, selected: list[str], score: pd.Series
    ) -> list[str]:
        """换手惩罚:新入选票相对被替换持仓票的得分优势不足 min_score_diff 时保留老票。"""
        if self.min_score_diff <= 0:
            return selected
        holdings = [s for s, p in ctx.portfolio.positions.items() if p.qty > 0]
        keep = [s for s in holdings if s not in selected and s in score.index]
        if not keep:
            return selected
        new_in = [s for s in selected if s not in holdings]
        out = list(selected)
        for s_old in sorted(keep, key=lambda s: -score[s]):
            if not new_in:
                break
            s_new = min(new_in, key=lambda s: score[s])
            if score[s_new] - score[s_old] < self.min_score_diff:
                out.remove(s_new)
                out.append(s_old)
                new_in.remove(s_new)
        return out

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

    def _log_mktcap_row(self, fundamentals: PanelDict) -> pd.Series | None:
        """当日(截至 now 最新一期)对数市值截面,供中性化剥离规模暴露(C5)。

        市值面板按 announce_date point-in-time 生效,``iloc[-1]`` 即最新一期;
        非正 / 缺失市值置 NaN,由 :func:`~djinn.factor.preprocess.neutralize` 的
        lstsq mask 剔除。
        """
        cap = fundamentals.get(COL_MARKET_CAP)
        if cap is None or cap.empty:
            return None
        row = cap.iloc[-1].astype(float)
        positive = row.where(row > 0)
        return pd.Series(np.log(positive.to_numpy(dtype=float)), index=positive.index)

    def _icir_score(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> pd.Series:
        """滚动 ICIR 加权合成得分(C9):逐因子历史面板 → 前向收益 → ICIR 权重。

        防未来函数:``fwd_returns = prices.pct_change(p).shift(-p)`` 在 now 之后为
        NaN(未来价格不可见);``rolling_ic_weights(shift_periods=p)`` 再把 IC 序列
        右移 p 日,即 now 日只用 now−p 日(其前向收益窗口 now−p→now 已落定)的 IC。
        权重符号自适配方向(IC 为负的因子自动取负权),无需手工 ``direction``。
        """
        factor_panels: dict[str, Panel] = {}
        for f in self._factors:
            try:
                panel = f.compute(prices, ohlcv, fundamentals)
            except Exception as e:
                _log.warning("因子 %s 计算失败 @icir: %s", f.name, e)
                continue
            if len(panel) > 0:
                factor_panels[f.name] = panel
        if not factor_panels or len(prices) < 2:
            return pd.Series(dtype=float)
        p = self.rebalance_freq
        fwd = prices.pct_change(p).shift(-p)
        weights = rolling_ic_weights(
            factor_panels,
            fwd,
            window=self.icir_window,
            min_periods=self.icir_min_periods,
            shift_periods=p,
        )
        if weights.empty:
            return pd.Series(dtype=float)
        score = composite_score(factor_panels, weights)
        return score.iloc[-1] if not score.empty else pd.Series(dtype=float)

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

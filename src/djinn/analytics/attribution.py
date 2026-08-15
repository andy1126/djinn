"""绩效归因(Phase 5):Brinson 行业归因 + 因子归因 + 因子暴露 / 行业分布报告。

- :func:`brinson_attribution`:Brinson-Fachler 模型,把组合相对基准的超额收益按行业
  分解为**配置(allocation)+ 选股(selection)+ 交互(interaction)**三效应;
  未满仓部分视为"现金"段(收益 0),保证 **三效应之和 = 超额收益** 的恒等式成立。
- :func:`factor_attribution`:基本面因子模型归因,``R_p ≈ Σ_f 暴露_f × 因子收益_f
  + 特异收益(α)``。
- :func:`build_exposure_report` / :class:`FactorExposureReport`:由组合权重 × 因子面板
  × 行业映射,产出因子暴露时序与行业权重分布。

``to_dict()`` 一律沿用后端序列化约定:Series → ``{index, values}``,DataFrame →
``{index, columns, data}``,供 API 直接透传。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_CASH = "__CASH__"  # 合成现金段标的(收益恒为 0)
_CASH_INDUSTRY = "现金"
_OTHER = "其他"  # 缺行业映射的兜底桶


# ── 序列化(与 api/routers/backtests.py 约定一致)─────────────
def _series_payload(s: pd.Series) -> dict[str, Any]:
    if s is None or len(s) == 0:
        return {"index": [], "values": []}
    return {
        "index": [str(x) for x in s.index],
        "values": [float(v) for v in s.to_numpy(dtype=float)],
    }


def _df_payload(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "index": [],
            "columns": [] if df is None else list(df.columns),
            "data": [],
        }
    return {
        "index": [str(x) for x in df.index],
        "columns": [str(c) for c in df.columns],
        "data": [[float(v) for v in row] for row in df.to_numpy(dtype=float).tolist()],
    }


# ── Brinson 行业归因 ─────────────────────────────────────
@dataclass
class BrinsonResult:
    """Brinson-Fachler 归因结果(按行业分解,效应可累加)。"""

    allocation: pd.Series  # index=行业,配置效应
    selection: pd.Series  # index=行业,选股效应
    interaction: pd.Series  # index=行业,交互效应
    excess_return: float  # 超额收益 R_p − R_b

    @property
    def total_effect(self) -> float:
        """三效应之和(恒等于 :attr:`excess_return`)。"""
        return float(
            self.allocation.sum() + self.selection.sum() + self.interaction.sum()
        )

    def by_industry(self) -> pd.DataFrame:
        """按行业汇总三效应(columns=[allocation, selection, interaction])。"""
        return pd.DataFrame(
            {
                "allocation": self.allocation,
                "selection": self.selection,
                "interaction": self.interaction,
            }
        ).fillna(0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation": _series_payload(self.allocation),
            "selection": _series_payload(self.selection),
            "interaction": _series_payload(self.interaction),
            "excess_return": float(self.excess_return),
            "total_effect": self.total_effect,
        }


def brinson_attribution(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: dict[str, float],
    returns: pd.DataFrame,
    industry_map: dict[str, str],
) -> BrinsonResult:
    """Brinson-Fachler 行业归因(逐日分解后对全区间的效应求和)。

    Args:
        portfolio_weights: date × symbol 组合权重(如 ``weights_curve``),允许不满仓。
        benchmark_weights: symbol → 基准静态权重(自动归一化;基准视为每日再平衡到该权重)。
        returns: date × symbol 标的日收益(与组合权重同频)。
        industry_map: symbol → 行业名;缺失归入"其他"。

    Returns:
        :class:`BrinsonResult`,满足 ``allocation + selection + interaction ==
        excess_return``(恒等式,逐日成立 hence 求和成立)。
    """
    symbols = [str(c) for c in returns.columns]
    ind: dict[str, str] = {s: industry_map.get(s, _OTHER) for s in symbols}
    rets = returns.copy()
    rets.columns = symbols
    rets = rets.fillna(0.0)
    # 进入当日的权重 = 前一日收盘权重(T+0 用前一日持仓赚当日收益)
    wp = portfolio_weights.reindex(index=rets.index, columns=symbols).fillna(0.0)
    wp_prev = wp.shift(1).fillna(0.0)
    # 剔除首个交易日:其"进入权重"恒为 0(shift 暖场产物),不应计入归因
    rets = rets.iloc[1:]
    wp_prev = wp_prev.iloc[1:]
    # 基准静态权重(归一化)
    wb = pd.Series({s: float(benchmark_weights.get(s, 0.0)) for s in symbols})
    if float(wb.sum()) > 0:
        wb = wb / float(wb.sum())

    # 现金段:组合未满仓部分视作收益 0 的配置段,保证恒等式
    wp_full = wp_prev.copy()
    rets_full = rets.copy()
    wb_full = wb.copy()
    ind_full = dict(ind)
    cash_w = (1.0 - wp_prev.sum(axis=1)).clip(lower=0.0)
    if float(cash_w.sum()) > 1e-9:
        wp_full[_CASH] = cash_w
        rets_full[_CASH] = 0.0
        wb_full[_CASH] = 0.0
        ind_full[_CASH] = _CASH_INDUSTRY

    industries = sorted(set(ind_full.values()))
    r_b = (rets_full * wb_full).sum(axis=1)  # 基准日收益
    excess = float((wp_full * rets_full).sum(axis=1).sum() - r_b.sum())

    alloc = pd.Series(0.0, index=industries)
    select = pd.Series(0.0, index=industries)
    interact = pd.Series(0.0, index=industries)
    for g in industries:
        cols = [s for s in wp_full.columns if ind_full.get(str(s), _OTHER) == g]
        wp_g = wp_full[cols].sum(axis=1)
        contrib_p = (wp_full[cols] * rets_full[cols]).sum(axis=1)
        r_p_g = contrib_p.div(wp_g.where(wp_g > 0)).fillna(0.0)
        wb_g = float(wb_full[cols].sum())
        contrib_b = (rets_full[cols] * wb_full[cols]).sum(axis=1)
        r_b_g = contrib_b / wb_g if wb_g > 0 else pd.Series(0.0, index=rets_full.index)
        alloc[g] = float(((wp_g - wb_g) * (r_b_g - r_b)).sum())
        select[g] = float((wb_g * (r_p_g - r_b_g)).sum())
        interact[g] = float(((wp_g - wb_g) * (r_p_g - r_b_g)).sum())
    return BrinsonResult(
        allocation=alloc, selection=select, interaction=interact, excess_return=excess
    )


# ── 因子归因 ─────────────────────────────────────────────
@dataclass
class FactorAttributionResult:
    """基本面因子模型归因结果。"""

    contributions: pd.Series  # index=因子,各因子累计贡献
    alpha: float  # 特异收益(组合总收益 − Σ 因子贡献)
    total_return: float  # 组合区间总收益(算术和)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contributions": _series_payload(self.contributions),
            "alpha": float(self.alpha),
            "total_return": float(self.total_return),
            "attributed": float(self.contributions.sum()),
        }


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> FactorAttributionResult:
    """基本面因子模型归因:``contribution_f = Σ_t 暴露_f(t) × 因子收益_f(t)``。

    特异收益(α)= 组合总收益 − Σ_f contribution_f,满足
    ``contributions.sum() + alpha == total_return``(恒等式)。
    三路输入按交易日取交集对齐,缺失暴露 / 因子收益按 0 处理。
    """
    common = portfolio_returns.index.intersection(factor_exposures.index).intersection(
        factor_returns.index
    )
    # B7:暴露滞后一日 —— t 日收益由 t−1 日收盘暴露解释(与 Brinson 一致,防前视)
    expo = factor_exposures.reindex(common).fillna(0.0).shift(1).fillna(0.0)
    fret = factor_returns.reindex(common).fillna(0.0)
    port = portfolio_returns.reindex(common).fillna(0.0)
    factors = [c for c in expo.columns if c in fret.columns]
    contrib = (expo[factors] * fret[factors]).sum()
    total = float(port.sum())
    alpha = total - float(contrib.sum())
    return FactorAttributionResult(
        contributions=contrib.astype(float), alpha=alpha, total_return=total
    )


# ── 因子暴露 / 行业分布报告 ───────────────────────────────
@dataclass
class FactorExposureReport:
    """因子暴露时序 + 行业权重分布。"""

    exposures: pd.DataFrame = field(default_factory=pd.DataFrame)  # date × factor
    industry_distribution: pd.DataFrame = field(  # date × industry
        default_factory=pd.DataFrame
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exposures": _df_payload(self.exposures),
            "industry_distribution": _df_payload(self.industry_distribution),
        }


def build_exposure_report(
    weights: pd.DataFrame,
    factor_panels: dict[str, pd.DataFrame],
    industry_map: dict[str, str],
) -> FactorExposureReport:
    """由组合权重 × 因子面板 × 行业映射,计算因子暴露时序与行业分布。

    - 因子暴露 ``exposure_f(t) = Σ_s w_s(t) × factor_{f,s}(t)``(权重加权的因子值);
    - 行业分布 ``dist_g(t) = Σ_{s ∈ g} w_s(t)``(各行业权重占比)。
    """
    symbols = [str(c) for c in weights.columns]
    w = weights.copy()
    w.columns = symbols
    exposures: dict[str, pd.Series] = {}
    for name, panel in factor_panels.items():
        p = panel.reindex(index=w.index, columns=symbols).fillna(0.0)
        exposures[name] = (w * p).sum(axis=1)
    expo_df = pd.DataFrame(exposures)

    ind = {s: industry_map.get(s, _OTHER) for s in symbols}
    dist: dict[str, pd.Series] = {}
    for g in sorted(set(ind.values())):
        cols = [s for s in symbols if ind[s] == g]
        dist[g] = w[cols].sum(axis=1)
    ind_df = pd.DataFrame(dist)
    return FactorExposureReport(exposures=expo_df, industry_distribution=ind_df)

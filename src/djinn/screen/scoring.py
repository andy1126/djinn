"""多因子打分:把若干因子按方向 + 权重合成为综合得分,并支持 TopN 选股。

核心是两个粒度:
- :func:`score_cross_section`:单截面(symbol × factor)→ symbol 得分,供策略逐日调用;
- :func:`score_universe`:整个 :class:`~djinn.factor.engine.FactorPanel` → date × symbol
  得分宽表,供分析 / 动态股票池构建。

合成规则:每因子截面先去极值 + z-score 标准化(``preprocess``),乘以 ``direction``
(``1`` = 值越高越好,``-1`` = 值越低越好,实现低值优选因子如波动率),再按 ``weight``
加权求和。NaN 因子值按 0 计入(该标的在该因子上不得分)。
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from djinn.factor.base import Panel, PanelDict
from djinn.factor.engine import FactorPanel
from djinn.factor.preprocess import standardize, winsorize


class FactorScore(BaseModel):
    """单因子打分权重。"""

    model_config = ConfigDict(extra="forbid")

    factor: str = Field(..., description="因子名(须出现在因子面板中)")
    weight: float = Field(default=1.0, description="加权权重(可正可负)")
    direction: Literal[1, -1] = Field(
        default=1, description="1=因子值越高越好;-1=越低越好"
    )


def score_cross_section(
    cross: pd.DataFrame,
    scores: list[FactorScore],
    preprocess: bool = True,
) -> pd.Series:
    """单截面合成打分。

    Args:
        cross: index=symbol、columns=因子名 的因子值截面。
        scores: 各因子的权重 / 方向。
        preprocess: 是否逐因子去极值 + z-score 标准化(量纲统一,推荐)。

    Returns:
        index=symbol 的综合得分 Series(越高越优)。
    """
    if cross.empty:
        return pd.Series(dtype=float)
    total = pd.Series(0.0, index=cross.index, dtype=float)
    for fs in scores:
        if fs.factor not in cross.columns:
            continue
        # 转置成 1×N 单行面板,复用 preprocess 的行向(截面)实现
        row = cross[[fs.factor]].T.astype(float)
        if preprocess:
            row = standardize(winsorize(row))
        vals = row.iloc[0] * float(fs.direction)
        total = total + vals.fillna(0.0) * float(fs.weight)
    return total


def score_universe(
    factor_panel: FactorPanel | PanelDict,
    scores: list[FactorScore],
    preprocess: bool = True,
) -> Panel:
    """对因子面板逐日合成打分,返回 ``date × symbol`` 综合得分宽表。"""
    data: PanelDict = (
        factor_panel.data if isinstance(factor_panel, FactorPanel) else factor_panel
    )
    if not data:
        return pd.DataFrame()
    # 所有因子日期并集(排序)与标的并集
    dates: pd.DatetimeIndex = pd.DatetimeIndex([])
    symbols: list[str] = []
    seen: set[str] = set()
    for df in data.values():
        dates = dates.union(pd.DatetimeIndex(df.index))
        for s in df.columns:
            if s not in seen:
                seen.add(s)
                symbols.append(s)
    out: dict[pd.Timestamp, pd.Series] = {}
    for ts in dates:
        cross = pd.DataFrame(
            {
                name: df.loc[ts].reindex(symbols)
                for name, df in data.items()
                if ts in df.index
            }
        )
        out[ts] = score_cross_section(cross, scores, preprocess)
    return pd.DataFrame(out).T.sort_index()


def top_n(score_df: Panel, when: object, n: int) -> list[str]:
    """取 ``when``(或之前最近一日)截面得分最高的前 ``n`` 个标的。"""
    if score_df.empty or n <= 0:
        return []
    ts = pd.Timestamp(when)  # type: ignore[arg-type]
    if ts in score_df.index:
        row = score_df.loc[ts]
    else:
        prior = score_df.index[score_df.index <= ts]
        if len(prior) == 0:
            return []
        row = score_df.loc[prior[-1]]
    if isinstance(row, pd.DataFrame):  # 重复日期标签 → 取首行
        row = row.iloc[0]
    row = row.dropna()
    if len(row) == 0:
        return []
    return row.nlargest(int(n)).index.tolist()

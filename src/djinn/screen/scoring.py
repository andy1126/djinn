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
from djinn.factor.preprocess import neutralize as preprocess_neutralize
from djinn.factor.preprocess import orthogonalize as preprocess_orthogonalize
from djinn.factor.preprocess import standardize, winsorize
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

# C14:缺失打分因子的告警去重(score_universe 逐日调用时只警一次,避免刷屏)
_WARNED_MISSING: set[str] = set()
# C14:最近一次 score_cross_section 的 meta(实际参与 / 缺失的因子名单,供调用方/测试断言)
LAST_SCORE_META: dict[str, list[str]] = {}


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
    *,
    neutralize: bool = False,
    industry_map: dict[str, str] | None = None,
    log_mktcap: pd.Series | None = None,
    orthogonalize: bool = False,
) -> pd.Series:
    """单截面合成打分。

    Args:
        cross: index=symbol、columns=因子名 的因子值截面。
        scores: 各因子的权重 / 方向。
        preprocess: 是否逐因子去极值 + z-score 标准化(量纲统一,推荐)。
        neutralize: 是否在去极值后、标准化前做行业/市值中性化(需给 industry_map
            或 log_mktcap,否则 warning 并跳过)。
        industry_map: symbol → 行业名;log_mktcap: index=symbol 的对数市值 Series。
        orthogonalize: 是否在标准化前对因子截面做 Schmidt 正交化(按 scores 顺序,
            后序因子对前序取残差,剥离因子间线性重叠)。

    Returns:
        index=symbol 的综合得分 Series(越高越优)。
    """
    if cross.empty:
        return pd.Series(dtype=float)
    # 可用因子列(按 scores 顺序);缺失因子显式告警,每个因子只警一次(C14)
    cols = [fs.factor for fs in scores if fs.factor in cross.columns]
    missing = [fs.factor for fs in scores if fs.factor not in cross.columns]
    for name in missing:
        if name not in _WARNED_MISSING:
            _WARNED_MISSING.add(name)
            _log.warning(
                "打分因子 %s 不在因子面板中(可用: %s),已跳过(仅告警一次)",
                name,
                list(cross.columns),
            )
    LAST_SCORE_META["factors_used"] = list(cols)
    LAST_SCORE_META["missing"] = missing
    # 组装 factor × symbol 面板,整体去极值(winsorize 逐行独立,等价逐因子)
    panel = cross[cols].T.astype(float)
    if preprocess:
        panel = winsorize(panel)
        # C10:标准化前 Schmidt 正交化(后序因子对前序取残差)
        if orthogonalize and len(cols) >= 2:
            # 单截面伪造成「1 日 × symbol」多因子面板,复用 orthogonalize 的逐日 Schmidt
            # (其按 date 遍历、各因子共享 index;这里共享一个虚拟 index)。
            dummy = pd.Index([0])
            ortho_input = {
                c: pd.DataFrame(
                    [panel.loc[c].to_numpy()], index=dummy, columns=panel.columns
                )
                for c in cols
            }
            ortho = preprocess_orthogonalize(ortho_input, order=cols)
            panel = pd.DataFrame({c: ortho[c].iloc[0] for c in cols}).T
    total = pd.Series(0.0, index=cross.index, dtype=float)
    for fs in scores:
        if fs.factor not in cols:
            continue
        row = panel.loc[[fs.factor]]
        if preprocess:
            if neutralize:
                if industry_map is not None or log_mktcap is not None:
                    logcap_panel: Panel | None = None
                    if log_mktcap is not None:
                        logcap_panel = pd.DataFrame(
                            [log_mktcap.reindex(cross.index).astype(float).to_numpy()],
                            index=row.index,
                            columns=cross.index,
                        )
                    row = preprocess_neutralize(
                        row, industry_map=industry_map, log_mktcap=logcap_panel
                    )
                else:
                    _log.warning(
                        "neutralize=True 但缺 industry_map/log_mktcap,跳过中性化"
                    )
            row = standardize(row)
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
    # D9:searchsorted 二分定位(替代线性扫 O(T)→O(log T))
    pos = score_df.index.searchsorted(ts, side="right")
    if pos == 0:
        return []
    row = score_df.iloc[pos - 1]  # iloc 单行定位恒返回 Series
    row = row.dropna()
    if len(row) == 0:
        return []
    return row.nlargest(int(n)).index.tolist()

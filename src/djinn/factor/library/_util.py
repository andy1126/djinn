"""library 内部工具:基本面字段取数对齐。"""

from __future__ import annotations

import pandas as pd

from djinn.factor.base import Panel, PanelDict


def fund_panel(fundamentals: PanelDict, key: str, like: Panel) -> Panel:
    """取基本面字段宽表并对齐到 ``like``(缺失则全 NaN)。"""
    df = fundamentals.get(key)
    if df is None:
        return pd.DataFrame(float("nan"), index=like.index, columns=like.columns)
    return df.reindex(index=like.index, columns=like.columns).astype(float)

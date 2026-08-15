"""截面筛选条件模型(ScreenCondition)。

从 :mod:`djinn.screen` 上移到配置层,消除"配置模型依赖选股层"的分层倒置:
``config/models.py`` 直接引用本模型,而 :mod:`djinn.screen.screener` 反向 import。
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

ScreenOp = Literal["gt", "lt", "ge", "le", "eq", "between", "in"]


class ScreenCondition(BaseModel):
    """单条筛选条件(``field`` / ``op`` / ``value`` + 取值校验 + 截面求掩码)。"""

    model_config = ConfigDict(extra="forbid")

    field: str
    op: ScreenOp
    value: Any  # 标量 / [lo,hi] / 成员列表

    @field_validator("value")
    @classmethod
    def _check_value(cls, v: Any, info: Any) -> Any:
        op = info.data.get("op")
        if op == "between":
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(f"between 需要 [lo, hi] 两元素,实际 {v!r}")
            if float(v[0]) > float(v[1]):
                raise ValueError(f"between 下界需 ≤ 上界,实际 {v!r}")
        if op == "in" and not isinstance(v, (list, tuple, set)):
            raise ValueError(f"in 需要列表值,实际 {v!r}")
        return v

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """在截面 ``df`` 上求该条件的布尔掩码(index=symbol)。"""
        if self.field not in df.columns:
            # 无该字段:全部不通过
            return pd.Series(False, index=df.index)
        col = df[self.field]
        op, v = self.op, self.value
        res: pd.Series
        if op == "gt":
            res = col > v
        elif op == "lt":
            res = col < v
        elif op == "ge":
            res = col >= v
        elif op == "le":
            res = col <= v
        elif op == "eq":
            res = col == v
        elif op == "between":
            lo, hi = float(v[0]), float(v[1])
            res = (col >= lo) & (col <= hi)
        else:  # in
            res = col.isin(list(v))
        return res


__all__ = ["ScreenCondition", "ScreenOp"]

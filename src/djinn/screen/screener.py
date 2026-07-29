"""条件筛选器:把一组截面条件作用到基本面 / 行情衍生宽表,产出通过的股票列表。

筛选为**纯截面**操作:输入 ``index=symbol、columns=field`` 的截面 DataFrame
(基本面字段 + 可选行情衍生列),逐条件生成布尔掩码取交集,返回通过标的。

条件 op 语义:
- ``gt/lt/ge/le/eq``:标量比较(``eq`` 亦可用于行业等字符串字段);
- ``between``:``value=[lo, hi]`` 闭区间;
- ``in``:``value=[...]`` 成员判断(行业 / 代码集合)。

缺失字段或字段值为 NaN 的标的在该条件上判为不通过(NaN 比较恒为 False)。
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

ScreenOp = Literal["gt", "lt", "ge", "le", "eq", "between", "in"]


class ScreenCondition(BaseModel):
    """单条筛选条件。"""

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


class Screener:
    """截面筛选器(无状态,静态方法)。"""

    @staticmethod
    def apply(
        conditions: list[ScreenCondition],
        fundamentals_df: pd.DataFrame,
        ohlcv_derived: pd.DataFrame | None = None,
    ) -> list[str]:
        """对截面应用全部条件(取交集),返回通过标的代码(排序后)。

        Args:
            conditions: 条件列表(空列表 = 全通过)。
            fundamentals_df: 基本面截面,index=symbol、columns=field。
            ohlcv_derived: 可选行情衍生截面(如 turnover / momentum),按 symbol join。
        """
        df = fundamentals_df
        if ohlcv_derived is not None:
            df = df.join(ohlcv_derived, how="left", rsuffix="_mkt")
        if len(df) == 0:
            return []
        mask = pd.Series(True, index=df.index)
        for cond in conditions:
            mask &= cond.mask(df)
        return sorted(df.index[mask.fillna(False)].tolist())

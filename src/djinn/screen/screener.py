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

import pandas as pd

from djinn.config.screen_models import ScreenCondition, ScreenOp

__all__ = ["ScreenCondition", "ScreenOp", "Screener"]


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

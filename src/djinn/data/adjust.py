"""复权处理:在前复权 / 后复权 / 不复权之间切换。

数据层存储 ``raw_close`` + ``adj_factor``(后复权因子,close = raw_close * adj_factor)。
本模块负责根据 :class:`Adjust` 生成对应口径的 OHLC 列,供回测使用。

约定:
- provider 原始输出若是"已复权"价格,则 ``adj_factor`` 视为相对该口径的因子,
  ``raw_close`` 存未复权价(若可得,否则等于 close)。
- 回测默认**后复权**:保证净值曲线连续,与持仓成本口径一致。
"""

from __future__ import annotations

import pandas as pd

from djinn.data.schema import (
    COL_ADJ_FACTOR,
    COL_CLOSE,
    COL_DIVIDEND,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_RAW_CLOSE,
    COL_SPLIT_RATIO,
    Adjust,
)
from djinn.utils.exceptions import DataError


def ensure_adjust_columns(df: pd.DataFrame) -> pd.DataFrame:
    """确保 df 含 raw_close / adj_factor / dividend / split_ratio 列(缺失则补默认)。"""
    out = df.copy()
    if COL_ADJ_FACTOR not in out.columns:
        out[COL_ADJ_FACTOR] = 1.0
    if COL_RAW_CLOSE not in out.columns:
        # 无原始价时,默认认为 close 即未复权
        out[COL_RAW_CLOSE] = out[COL_CLOSE]
    if COL_DIVIDEND not in out.columns:
        out[COL_DIVIDEND] = 0.0
    if COL_SPLIT_RATIO not in out.columns:
        out[COL_SPLIT_RATIO] = 1.0
    return out


def apply_adjust(df: pd.DataFrame, adjust: Adjust) -> pd.DataFrame:
    """将含 raw_close/adj_factor 的 df 转换为指定复权口径。

    返回的 df 中 open/high/low/close 为目标口径价格,raw_close/adj_factor 保留不变。
    """
    df = ensure_adjust_columns(df)
    if adjust is Adjust.NONE:
        # 用 raw_close 口径:close = raw_close * adj_factor => raw_close = close / adj_factor
        ratio = df[COL_RAW_CLOSE] / df[COL_CLOSE].replace(0, pd.NA)
        ratio = ratio.fillna(1.0)
        out = df.copy()
        for col in (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE):
            out[col] = out[col] * ratio
        return out

    if adjust is Adjust.BACKWARD:
        # 后复权:close = raw_close * adj_factor;若 provider 已给后复权 close,直接用
        return df.copy()

    if adjust is Adjust.FORWARD:
        # 前复权:以最新一日为基准,向前缩放
        # 前复权价 = 后复权价 / 末日 adj_factor
        last_factor = df[COL_ADJ_FACTOR].iloc[-1]
        if last_factor == 0 or pd.isna(last_factor):
            raise DataError("adj_factor 末值为 0,无法计算前复权")
        out = df.copy()
        scale = df[COL_ADJ_FACTOR] / last_factor
        for col in (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE):
            out[col] = out[col] * scale
        return out

    raise DataError(f"未知复权方式: {adjust}")

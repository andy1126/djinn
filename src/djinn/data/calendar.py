"""多市场交易日历封装(基于 exchange_calendars)。

提供:
- 按市场生成交易日索引;
- 对齐 MarketData(reindex + 停牌标记);
- 判定某日是否为交易日。
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Final

import numpy as np
import pandas as pd

from djinn.data.schema import (
    COL_AMOUNT,
    COL_IS_SUSPENDED,
    COL_VOLUME,
    Market,
)
from djinn.utils.exceptions import DataError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

# exchange_calendars 的市场代码映射。
_CAL_MAP: Final[dict[Market, str]] = {
    Market.CN: "XSHG",  # 上交所;A 股以 XSHG 为代表(XSHE 交易日基本一致)
    Market.HK: "XHKG",
    Market.US: "XNYS",
}


@lru_cache(maxsize=8)
def _get_calendar(market: Market) -> object:
    """惰性加载并缓存 exchange_calendars 日历实例。"""
    try:
        import exchange_calendars as xc
    except ImportError as e:  # pragma: no cover - 依赖已声明
        raise DataError("exchange_calendars 未安装") from e
    code = _CAL_MAP[market]
    return xc.get_calendar(code)


def trading_days(market: Market, start: date, end: date) -> pd.DatetimeIndex:
    """返回 [start, end] 闭区间内该市场的交易日索引。"""
    cal = _get_calendar(market)
    sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))  # type: ignore[attr-defined]
    return pd.DatetimeIndex(sessions)


def is_trading_day(market: Market, day: date) -> bool:
    """判定 day 是否为该市场交易日。"""
    cal = _get_calendar(market)
    return bool(cal.is_session(pd.Timestamp(day)))  # type: ignore[attr-defined]


def align_to_calendar(
    df: pd.DataFrame,
    market: Market,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """将行情 df reindex 到市场交易日,缺失日标记为停牌。

    - 索引被对齐到交易日;
    - 行情缺失的交易日:OHLC 用前一日 close 填充(停牌时持仓按最后价 mark),
      volume/amount 置 0,``is_suspended`` 置 True;
    - 开盘前(start 之前)不填充。
    """
    if len(df) == 0:
        return df
    if start is None:
        start = df.index[0].date()
    if end is None:
        end = df.index[-1].date()
    days = trading_days(market, start, end)
    aligned = df.reindex(days)
    # 停牌日:量/额置 0,价用前收盘 ffill
    if COL_VOLUME in aligned.columns:
        aligned[COL_VOLUME] = aligned[COL_VOLUME].fillna(0.0)
    if COL_AMOUNT in aligned.columns:
        aligned[COL_AMOUNT] = aligned[COL_AMOUNT].fillna(0.0)
    # 价与因子类列前向填充(停牌沿用上一交易日值,保持口径连续)
    ffill_cols = ("open", "high", "low", "close", "raw_close", "adj_factor")
    for col in ffill_cols:
        if col in aligned.columns:
            aligned[col] = aligned[col].ffill()
    # dividend/split_ratio 缺失日视为 0(无分红/拆股事件)
    for col in ("dividend", "split_ratio"):
        if col in aligned.columns:
            aligned[col] = aligned[col].fillna(0.0)
    # 停牌标记:原 df 的标记(若存在)用于已有行;缺失/对齐产生的行由 missing 推断
    was_suspended: np.ndarray
    if COL_IS_SUSPENDED in aligned.columns:
        was_suspended = np.asarray(
            aligned[COL_IS_SUSPENDED].notna() & aligned[COL_IS_SUSPENDED].astype(bool),
            dtype=bool,
        )
    else:
        was_suspended = np.zeros(len(aligned), dtype=bool)
    # 缺失(原 df 没有该交易日)即停牌
    missing = ~np.isin(aligned.index, df.index)
    aligned[COL_IS_SUSPENDED] = (was_suspended | missing).astype(bool)
    if aligned[COL_IS_SUSPENDED].any():
        n = int(aligned[COL_IS_SUSPENDED].sum())
        _log.debug("对齐 %s:标记 %d 个停牌/缺失交易日", market.value, n)
    return aligned

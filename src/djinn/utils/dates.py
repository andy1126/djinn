"""日期 / 交易日处理辅助。

统一将各种日期输入规范化为 :class:`datetime.date`,并提供与交易日历无关的
纯函数工具;真正的多市场交易日历封装见 :mod:`djinn.data.calendar`。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Final

import pandas as pd

# 各市场年化交易日数(用于年化收益/波动换算)。
TRADING_DAYS_PER_YEAR: Final[dict[str, int]] = {
    "CN": 242,  # A 股
    "HK": 246,  # 港股
    "US": 252,  # 美股
    "DEFAULT": 252,
}

_EPOCH = date(1970, 1, 1)


def parse_date(value: str | date | datetime | pd.Timestamp) -> date:
    """将多种日期表示统一转为 :class:`datetime.date`。

    接受 ``str``(ISO 8601)、``date``、``datetime``、``pandas.Timestamp``。
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        # datetime 也覆盖 pandas.Timestamp(其子类)
        return value.date()
    if isinstance(value, str):
        # 兼容 "2014-01-01" / "2014/01/01" / "20140101"
        s = value.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        # 兜底:让 pandas 解析
        return pd.Timestamp(s).date()
    raise TypeError(f"无法解析日期: {value!r} ({type(value).__name__})")


def to_timestamp(d: date) -> pd.Timestamp:
    """date → pandas.Timestamp(纳秒,naive,用作 DataFrame 索引)。"""
    return pd.Timestamp(d)


def date_range(start: date, end: date) -> list[date]:
    """闭区间 [start, end] 的自然日列表(不含交易日历过滤)。"""
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def trading_days_per_year(market: str) -> int:
    """按市场代码返回年化交易日数。"""
    return TRADING_DAYS_PER_YEAR.get(market, TRADING_DAYS_PER_YEAR["DEFAULT"])


def days_since_epoch(d: date) -> int:
    """date 相对 1970-01-01 的天数,便于排序 / 哈希。"""
    return (d - _EPOCH).days

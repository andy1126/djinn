"""基本面数据模型与 point-in-time 对齐工具。

提供:
- :class:`Fundamentals`:规范化基本面容器(快照截面 / 单标的时序报告);
- :class:`FundamentalsSource`:基本面来源抽象(截面快照 + 历史报告两个入口);
- :func:`asof_snapshot`:point-in-time 硬性不变量的核心 —— 截面取值时只使用
  ``announce_date <= date`` 的最新一期财报,杜绝未来函数。

规范化列名集中在 :mod:`djinn.data.schema`(``COL_MARKET_CAP`` 等),下游因子层
不直接依赖任何 provider 原始字段。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd

from djinn.data.schema import (
    COL_ANNOUNCE_DATE,
    COL_REPORT_DATE,
    Market,
)
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class Fundamentals:
    """规范化基本面数据容器。

    两种形态:
    - **快照(截面)**:``df`` index=symbol,columns=基本面字段;``asof`` 为取值日。
    - **时序(单标的)**:``df`` index=报告期日期,含 ``report_date``/``announce_date``
      与各基本面字段,用于成长 / 质量因子回看。
    """

    df: pd.DataFrame
    asof: date | None = None

    def __len__(self) -> int:
        return len(self.df)


def asof_snapshot(history: pd.DataFrame, when: date) -> pd.Series | None:
    """从单标的的历史报告表(point-in-time)取 ``when`` 当日可见的最新一期。

    Args:
        history: index 任意,必须含 ``COL_ANNOUNCE_DATE`` 列(announce_date)。
        when: 取值日。

    Returns:
        最新可见报告的一行(``announce_date <= when`` 中 report_date 最新者);
        无可见报告返回 None。
    """
    if history is None or len(history) == 0:
        return None
    if COL_ANNOUNCE_DATE not in history.columns:
        # 无公告日信息时保守起见不返回(无法保证 point-in-time)。
        _log.warning("基本面历史缺 announce_date,无法保证 point-in-time")
        return None
    ann = pd.to_datetime(history[COL_ANNOUNCE_DATE])
    visible = history[ann <= pd.Timestamp(when)]
    if len(visible) == 0:
        return None
    if COL_REPORT_DATE in visible.columns:
        rep = pd.to_datetime(visible[COL_REPORT_DATE])
        latest = visible.iloc[int(rep.to_numpy().argmax())]
    else:
        latest = visible.iloc[-1]
    return latest


class FundamentalsSource(ABC):
    """基本面来源抽象。

    provider / 路由器按需实现;两个入口分别对应截面选股与时序因子回看。
    """

    name: str = "base"

    @abstractmethod
    def get_snapshot(
        self, symbols: list[str], when: date, market: Market | None = None
    ) -> pd.DataFrame:
        """返回 ``when`` 当日的截面快照(point-in-time)。

        Returns:
            index=symbol,columns=基本面字段(``COL_*``),数值为 float。
        """

    @abstractmethod
    def get_history(
        self, symbol: str, start: date, end: date, market: Market | None = None
    ) -> pd.DataFrame:
        """返回单标的在 [start, end] 可见的财报时序(含 announce/report_date)。"""

    def get_daily_valuation(
        self, symbol: str, start: date, end: date, market: Market | None = None
    ) -> pd.DataFrame:
        """返回单标的在 [start, end] 的日频估值时序(point-in-time,无前视)。

        语义不同于财报时序(按 report/announce_date 生效):估值是**日频行情衍生
        序列**(pe/pb/ps 等,每天随收盘价更新),index=交易日。基类默认返回空
        DataFrame(不支持);provider / 路由器按需覆写。
        """
        return pd.DataFrame()

    def get_daily_dividends(
        self, symbol: str, start: date, end: date, market: Market | None = None
    ) -> pd.DataFrame:
        """返回单标的在 [start, end] 的每股现金分红事件序列(股息率因子回看)。

        语义:分红是**事件型日频序列**(除息日生效),index=除息日,含 ``COL_DIVIDEND``
        列(每股现金分红,税前)。供股息率因子做 TTM 滚动求和。基类默认返回空
        DataFrame(不支持);provider / 路由器按需覆写。
        """
        return pd.DataFrame()

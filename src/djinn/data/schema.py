"""数据规范化 schema:市场、复权、OHLCV 列定义、Bar。

所有 provider 输出统一为 :class:`MarketData`(见 ``market_data.py``),其
``df`` 列名 / dtype / 索引在此处集中定义,确保下游策略与引擎不关心数据来源。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

# 规范化 OHLCV 列名(所有 provider 输出统一使用这些列)。
COL_OPEN: Final[str] = "open"
COL_HIGH: Final[str] = "high"
COL_LOW: Final[str] = "low"
COL_CLOSE: Final[str] = "close"
COL_VOLUME: Final[str] = "volume"
COL_AMOUNT: Final[str] = "amount"  # 成交额(可选,部分市场提供)
COL_RAW_CLOSE: Final[str] = "raw_close"  # 未复权收盘
COL_ADJ_FACTOR: Final[str] = "adj_factor"  # 后复权因子(close = raw_close * adj_factor)
COL_DIVIDEND: Final[str] = "dividend"  # 每股分红(现金,未复权口径)
COL_SPLIT_RATIO: Final[str] = "split_ratio"  # 拆股比例(1:N 记为 N)
COL_IS_SUSPENDED: Final[str] = "is_suspended"  # 停牌标记

# ── 基本面 / 行业规范化列(供 factor / screen 层统一引用)─────────
COL_MARKET_CAP: Final[str] = "market_cap"  # 总市值
COL_FLOAT_CAP: Final[str] = "float_cap"  # 流通市值
COL_PE: Final[str] = "pe"  # 市盈率
COL_PB: Final[str] = "pb"  # 市净率
COL_PS: Final[str] = "ps"  # 市销率
COL_ROE: Final[str] = "roe"  # 净资产收益率
COL_GROSS_MARGIN: Final[str] = "gross_margin"  # 毛利率
COL_REVENUE: Final[str] = "revenue"  # 营业收入
COL_NET_PROFIT: Final[str] = "net_profit"  # 净利润
COL_OCF: Final[str] = "ocf"  # 经营活动现金流净额
COL_TOTAL_ASSETS: Final[str] = "total_assets"  # 总资产
COL_REVENUE_YOY: Final[str] = "revenue_yoy"  # 营收同比
COL_PROFIT_YOY: Final[str] = "profit_yoy"  # 净利同比
COL_INDUSTRY: Final[str] = "industry"  # 所属行业
COL_REPORT_DATE: Final[str] = "report_date"  # 报告期(财报所属期)
COL_ANNOUNCE_DATE: Final[str] = "announce_date"  # 公告日(point-in-time 关键)

# 数值型基本面字段(截面 / 时序面板的核心列)。
FUNDAMENTAL_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    COL_MARKET_CAP,
    COL_FLOAT_CAP,
    COL_PE,
    COL_PB,
    COL_PS,
    COL_ROE,
    COL_GROSS_MARGIN,
    COL_REVENUE,
    COL_NET_PROFIT,
    COL_OCF,
    COL_TOTAL_ASSETS,
    COL_REVENUE_YOY,
    COL_PROFIT_YOY,
)

OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOLUME,
)
ALL_COLUMNS: Final[tuple[str, ...]] = (
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOLUME,
    COL_AMOUNT,
    COL_RAW_CLOSE,
    COL_ADJ_FACTOR,
    COL_DIVIDEND,
    COL_SPLIT_RATIO,
    COL_IS_SUSPENDED,
)


class Market(StrEnum):
    """市场标识。"""

    CN = "CN"  # A 股
    HK = "HK"  # 港股
    US = "US"  # 美股

    @property
    def lot_size(self) -> int:
        """最小交易手数(A/港股按手,美股 1 股)。"""
        if self is Market.US:
            return 1
        return 100  # A 股 / 港股 / ETF 默认 100(港股按手,此处取通用最小)

    @property
    def currency(self) -> str:
        return {Market.CN: "CNY", Market.HK: "HKD", Market.US: "USD"}[self]

    @property
    def calendar_name(self) -> str:
        """exchange_calendars 的日历代码。"""
        return {Market.CN: "XSHG", Market.HK: "XHKG", Market.US: "XNYS"}[self]

    @property
    def price_limit_pct(self) -> float | None:
        """常规涨跌停幅度(None 表示无限制,如美股)。"""
        if self is Market.CN:
            return 0.10  # 主板 ±10%(创业板/科创板/ST 在约束层特判)
        return None  # 港股 / 美股无涨跌停(港股有冷静期但回测不建模)


class Adjust(StrEnum):
    """复权方式。"""

    NONE = "none"  # 不复权(raw_close)
    FORWARD = "forward"  # 前复权
    BACKWARD = "backward"  # 后复权(默认,保证净值连续)


@dataclass(frozen=True, slots=True)
class Bar:
    """单标的单根 K 线(不可变快照)。

    引擎撮合时使用;价格用 float(行情数据天然浮点),数量用 int。
    """

    timestamp: date
    symbol: str
    market: Market
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float = 0.0
    raw_close: float = 0.0
    adj_factor: float = 1.0
    dividend: float = 0.0
    split_ratio: float = 1.0
    is_suspended: bool = False

    @property
    def has_trade(self) -> bool:
        """当日是否有成交(非停牌且量价非零)。"""
        return (not self.is_suspended) and self.volume > 0


def detect_market(symbol: str) -> Market:
    """根据代码格式启发式推断市场(供无显式市场信息时使用)。

    - ``.SH`` / ``.SZ`` / ``.BJ`` 或 6/0/3 开头 6 位数字 → A 股
    - 5 位数字 → 港股
    - 字母代码 → 美股
    """
    s = symbol.upper()
    if s.endswith((".SH", ".SZ", ".BJ")):
        return Market.CN
    digits = s.lstrip("=").replace(".", "").replace("^", "")
    if digits.isdigit() and len(digits) == 6:
        return Market.CN
    if digits.isdigit() and len(digits) == 5:
        return Market.HK
    return Market.US

"""
Djinn 日期工具模块。

这个模块提供了日期处理和交易日历相关的功能。
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple, Union, Dict, Any
from enum import Enum
import holidays
from pandas_market_calendars import get_calendar

from .logger import logger
from .exceptions import ValidationError


class MarketCalendarType(str, Enum):
    """市场日历类型枚举。"""

    US = "XNYS"  # 纽约证券交易所
    HK = "XHKG"  # 香港交易所
    CN = "XSHG"  # 上海证券交易所


class DateFormat(str, Enum):
    """日期格式枚举。"""

    ISO = "%Y-%m-%d"  # ISO 格式: 2023-12-31
    US = "%m/%d/%Y"  # 美国格式: 12/31/2023
    CN = "%Y年%m月%d日"  # 中文格式: 2023年12月31日
    COMPACT = "%Y%m%d"  # 紧凑格式: 20231231


class DateUtils:
    """日期工具类。"""

    # 市场日历缓存
    _calendar_cache: Dict[str, Any] = {}

    @staticmethod
    def parse_date(
        date_str: str,
        format_str: Optional[str] = None,
        raise_error: bool = True,
    ) -> Optional[date]:
        """
        解析日期字符串。

        Args:
            date_str: 日期字符串
            format_str: 日期格式字符串，如果为 None 则自动检测
            raise_error: 是否在解析失败时抛出异常

        Returns:
            Optional[date]: 解析后的日期对象，如果解析失败且 raise_error=False 则返回 None

        Raises:
            ValidationError: 如果日期解析失败且 raise_error=True
        """
        if format_str:
            formats = [format_str]
        else:
            # 尝试常见的日期格式
            formats = [
                "%Y-%m-%d",  # ISO
                "%Y/%m/%d",  # 斜杠格式
                "%Y%m%d",  # 紧凑格式
                "%m/%d/%Y",  # 美国格式
                "%d/%m/%Y",  # 欧洲格式
                "%Y-%m-%d %H:%M:%S",  # 带时间
                "%Y-%m-%dT%H:%M:%S",  # ISO 带时间
            ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.date()
            except ValueError:
                continue

        if raise_error:
            raise ValidationError(
                f"无法解析日期字符串: {date_str}",
                field="date_str",
                value=date_str,
                expected="有效的日期格式",
            )

        return None

    @staticmethod
    def format_date(
        date_obj: Union[date, datetime, pd.Timestamp, str],
        format_str: str = DateFormat.ISO,
    ) -> str:
        """
        格式化日期对象。

        Args:
            date_obj: 日期对象或字符串
            format_str: 输出格式

        Returns:
            str: 格式化后的日期字符串
        """
        # 如果输入是字符串，先解析
        if isinstance(date_obj, str):
            date_obj = DateUtils.parse_date(date_obj)

        # 转换为 datetime 对象
        if isinstance(date_obj, date):
            dt = datetime.combine(date_obj, datetime.min.time())
        elif isinstance(date_obj, pd.Timestamp):
            dt = date_obj.to_pydatetime()
        elif isinstance(date_obj, datetime):
            dt = date_obj
        else:
            raise ValidationError(
                f"不支持的日期类型: {type(date_obj)}",
                field="date_obj",
                value=str(date_obj),
                expected="date, datetime, pd.Timestamp 或 str",
            )

        return dt.strftime(format_str)

    @staticmethod
    def get_market_calendar(
        market: MarketCalendarType,
        cache: bool = True,
    ) -> Any:
        """
        获取市场交易日历。

        Args:
            market: 市场类型
            cache: 是否使用缓存

        Returns:
            Any: 市场日历对象
        """
        market_code = market.value

        # 检查缓存
        if cache and market_code in DateUtils._calendar_cache:
            return DateUtils._calendar_cache[market_code]

        try:
            calendar = get_calendar(market_code)
            if cache:
                DateUtils._calendar_cache[market_code] = calendar

            logger.debug(f"加载市场日历: {market_code}")
            return calendar

        except Exception as e:
            logger.error(f"加载市场日历失败: {market_code}, 错误: {e}")
            raise ValidationError(
                f"无法加载市场日历: {market_code}",
                field="market",
                value=market_code,
                details={"error": str(e)},
            )

    @staticmethod
    def is_trading_day(
        date_obj: Union[date, datetime, str],
        market: MarketCalendarType = MarketCalendarType.US,
    ) -> bool:
        """
        检查指定日期是否为交易日。

        Args:
            date_obj: 日期对象或字符串
            market: 市场类型

        Returns:
            bool: 如果是交易日返回 True，否则返回 False
        """
        # 解析日期
        if isinstance(date_obj, str):
            date_obj = DateUtils.parse_date(date_obj)

        # 获取日历
        calendar = DateUtils.get_market_calendar(market)

        # 检查是否为交易日
        try:
            date_str = DateUtils.format_date(date_obj, DateFormat.ISO)
            schedule = calendar.schedule(start_date=date_str, end_date=date_str)
            return not schedule.empty
        except Exception as e:
            logger.warning(f"检查交易日失败: {date_obj}, 错误: {e}")
            return False

    @staticmethod
    def get_trading_days(
        start_date: Union[date, datetime, str],
        end_date: Union[date, datetime, str],
        market: MarketCalendarType = MarketCalendarType.US,
    ) -> List[date]:
        """
        获取指定时间范围内的所有交易日。

        Args:
            start_date: 开始日期
            end_date: 结束日期
            market: 市场类型

        Returns:
            List[date]: 交易日列表
        """
        # 解析日期
        start_date_obj = (
            DateUtils.parse_date(start_date)
            if isinstance(start_date, str)
            else start_date
        )
        end_date_obj = (
            DateUtils.parse_date(end_date)
            if isinstance(end_date, str)
            else end_date
        )

        # 格式化日期字符串
        start_str = DateUtils.format_date(start_date_obj, DateFormat.ISO)
        end_str = DateUtils.format_date(end_date_obj, DateFormat.ISO)

        # 获取日历
        calendar = DateUtils.get_market_calendar(market)

        try:
            # 获取交易日历
            schedule = calendar.schedule(start_date=start_str, end_date=end_str)

            if schedule.empty:
                return []

            # 转换为日期列表
            trading_days = [
                pd.Timestamp(idx).date()
                for idx in schedule.index
            ]

            logger.debug(
                f"获取交易日: {start_str} 到 {end_str}, "
                f"共 {len(trading_days)} 个交易日"
            )

            return trading_days

        except Exception as e:
            logger.error(f"获取交易日失败: {start_str} 到 {end_str}, 错误: {e}")
            raise ValidationError(
                f"获取交易日失败",
                details={
                    "start_date": start_str,
                    "end_date": end_str,
                    "market": market.value,
                    "error": str(e),
                },
            )

    @staticmethod
    def get_next_trading_day(
        date_obj: Union[date, datetime, str],
        market: MarketCalendarType = MarketCalendarType.US,
        n: int = 1,
    ) -> date:
        """
        获取第 n 个下一个交易日。

        Args:
            date_obj: 起始日期
            market: 市场类型
            n: 向前跳过的交易日数

        Returns:
            date: 第 n 个下一个交易日
        """
        if n < 1:
            raise ValidationError(
                "n 必须大于等于 1",
                field="n",
                value=n,
                expected=">= 1",
            )

        # 解析日期
        if isinstance(date_obj, str):
            date_obj = DateUtils.parse_date(date_obj)

        # 获取交易日历
        start_str = DateUtils.format_date(date_obj, DateFormat.ISO)

        # 扩展搜索范围，确保能找到足够的交易日
        days_to_add = n * 5  # 保守估计，每个交易日之间最多有4个非交易日（周末+节假日）
        end_date = date_obj + timedelta(days=days_to_add)
        end_str = DateUtils.format_date(end_date, DateFormat.ISO)

        trading_days = DateUtils.get_trading_days(start_str, end_str, market)

        # 找到起始日期之后的交易日
        future_days = [d for d in trading_days if d > date_obj]

        if len(future_days) < n:
            # 如果找不到足够的交易日，扩展搜索范围
            logger.warning(
                f"在 {days_to_add} 天内找不到 {n} 个交易日，扩展搜索范围"
            )
            end_date = date_obj + timedelta(days=days_to_add * 2)
            end_str = DateUtils.format_date(end_date, DateFormat.ISO)
            trading_days = DateUtils.get_trading_days(start_str, end_str, market)
            future_days = [d for d in trading_days if d > date_obj]

            if len(future_days) < n:
                raise ValidationError(
                    f"在扩展范围内也找不到 {n} 个交易日",
                    details={
                        "start_date": start_str,
                        "n": n,
                        "found_days": len(future_days),
                    },
                )

        return future_days[n - 1]

    @staticmethod
    def get_previous_trading_day(
        date_obj: Union[date, datetime, str],
        market: MarketCalendarType = MarketCalendarType.US,
        n: int = 1,
    ) -> date:
        """
        获取第 n 个上一个交易日。

        Args:
            date_obj: 起始日期
            market: 市场类型
            n: 向后跳过的交易日数

        Returns:
            date: 第 n 个上一个交易日
        """
        if n < 1:
            raise ValidationError(
                "n 必须大于等于 1",
                field="n",
                value=n,
                expected=">= 1",
            )

        # 解析日期
        if isinstance(date_obj, str):
            date_obj = DateUtils.parse_date(date_obj)

        # 获取交易日历
        end_str = DateUtils.format_date(date_obj, DateFormat.ISO)

        # 扩展搜索范围
        days_to_subtract = n * 5
        start_date = date_obj - timedelta(days=days_to_subtract)
        start_str = DateUtils.format_date(start_date, DateFormat.ISO)

        trading_days = DateUtils.get_trading_days(start_str, end_str, market)

        # 找到起始日期之前的交易日
        past_days = [d for d in trading_days if d < date_obj]

        if len(past_days) < n:
            # 如果找不到足够的交易日，扩展搜索范围
            logger.warning(
                f"在 {days_to_subtract} 天内找不到 {n} 个交易日，扩展搜索范围"
            )
            start_date = date_obj - timedelta(days=days_to_subtract * 2)
            start_str = DateUtils.format_date(start_date, DateFormat.ISO)
            trading_days = DateUtils.get_trading_days(start_str, end_str, market)
            past_days = [d for d in trading_days if d < date_obj]

            if len(past_days) < n:
                raise ValidationError(
                    f"在扩展范围内也找不到 {n} 个交易日",
                    details={
                        "end_date": end_str,
                        "n": n,
                        "found_days": len(past_days),
                    },
                )

        return past_days[-n]

    @staticmethod
    def is_holiday(
        date_obj: Union[date, datetime, str],
        country: str = "US",
        market: Optional[MarketCalendarType] = None,
    ) -> bool:
        """
        检查指定日期是否为节假日。

        Args:
            date_obj: 日期对象或字符串
            country: 国家代码 (US, CN, HK等)
            market: 市场类型，如果提供则同时检查市场特定节假日

        Returns:
            bool: 如果是节假日返回 True，否则返回 False
        """
        # 解析日期
        if isinstance(date_obj, str):
            date_obj = DateUtils.parse_date(date_obj)

        # 检查国家节假日
        try:
            country_holidays = holidays.country_holidays(country)
            is_country_holiday = date_obj in country_holidays
        except Exception as e:
            logger.warning(f"检查国家节假日失败: {country}, 错误: {e}")
            is_country_holiday = False

        # 检查市场节假日
        is_market_holiday = False
        if market:
            is_market_holiday = not DateUtils.is_trading_day(date_obj, market)

        return is_country_holiday or is_market_holiday

    @staticmethod
    def get_date_range(
        start_date: Union[date, datetime, str],
        end_date: Union[date, datetime, str],
        freq: str = "D",
        trading_days_only: bool = False,
        market: MarketCalendarType = MarketCalendarType.US,
    ) -> List[date]:
        """
        获取日期范围。

        Args:
            start_date: 开始日期
            end_date: 结束日期
            freq: 频率 (D: 日, W: 周, M: 月, Q: 季, Y: 年)
            trading_days_only: 是否只包含交易日
            market: 市场类型（仅当 trading_days_only=True 时有效）

        Returns:
            List[date]: 日期列表
        """
        # 解析日期
        start_date_obj = (
            DateUtils.parse_date(start_date)
            if isinstance(start_date, str)
            else start_date
        )
        end_date_obj = (
            DateUtils.parse_date(end_date)
            if isinstance(end_date, str)
            else end_date
        )

        # 生成日期范围
        date_range = pd.date_range(
            start=start_date_obj,
            end=end_date_obj,
            freq=freq,
        )

        # 转换为日期列表
        dates = [d.date() for d in date_range]

        # 如果只需要交易日，进行过滤
        if trading_days_only:
            dates = [
                d for d in dates
                if DateUtils.is_trading_day(d, market)
            ]

        return dates

    @staticmethod
    def calculate_age(
        start_date: Union[date, datetime, str],
        end_date: Union[date, datetime, str],
        unit: str = "years",
    ) -> float:
        """
        计算两个日期之间的时间差。

        Args:
            start_date: 开始日期
            end_date: 结束日期
            unit: 时间单位 (years, months, days, trading_days)

        Returns:
            float: 时间差
        """
        # 解析日期
        start_date_obj = (
            DateUtils.parse_date(start_date)
            if isinstance(start_date, str)
            else start_date
        )
        end_date_obj = (
            DateUtils.parse_date(end_date)
            if isinstance(end_date, str)
            else end_date
        )

        if unit == "days":
            delta = (end_date_obj - start_date_obj).days
            return float(delta)

        elif unit == "trading_days":
            # 需要市场信息，这里使用默认的美国市场
            trading_days = DateUtils.get_trading_days(
                start_date_obj, end_date_obj, MarketCalendarType.US
            )
            return float(len(trading_days))

        elif unit == "months":
            # 计算月份差
            months = (end_date_obj.year - start_date_obj.year) * 12
            months += end_date_obj.month - start_date_obj.month
            # 调整天数差异
            if end_date_obj.day < start_date_obj.day:
                months -= 1
            return float(months)

        elif unit == "years":
            # 计算年份差（考虑天数）
            days_diff = (end_date_obj - start_date_obj).days
            years = days_diff / 365.25
            return years

        else:
            raise ValidationError(
                f"不支持的时间单位: {unit}",
                field="unit",
                value=unit,
                expected="years, months, days, trading_days",
            )


# 导出
__all__ = [
    "MarketCalendarType",
    "DateFormat",
    "DateUtils",
]
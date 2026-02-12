"""
Djinn 数据提供器基类。

这个模块定义了数据提供器的抽象基类和通用功能。
"""

import abc
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, date, timedelta
import pandas as pd

from .market_data import (
    MarketData,
    MarketDataRequest,
    MarketDataType,
    AdjustmentType,
    OHLCV,
    FundamentalData,
)
from ..utils.logger import logger
from ..utils.exceptions import DataError, NetworkError
from ..utils.file_utils import CacheManager, FileUtils
from ..utils.date_utils import DateUtils
from ..utils.config import config_manager


class DataProvider(abc.ABC):
    """数据提供器抽象基类。"""

    def __init__(
        self,
        name: str,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        retry_delay: int = 1,
    ):
        """
        初始化数据提供器。

        Args:
            name: 提供器名称
            cache_enabled: 是否启用缓存
            cache_ttl: 缓存过期时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.name = name
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 初始化缓存
        if cache_enabled:
            cache_dir = config_manager.get_setting("cache_dir", "./.cache/djinn")
            self.cache = CacheManager(
                cache_dir=cache_dir,
                default_ttl=cache_ttl,
            )
        else:
            self.cache = None

        logger.info(f"初始化数据提供器: {name}")

    @abc.abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d",
        adjustment: AdjustmentType = AdjustmentType.ADJ,
    ) -> MarketData:
        """
        获取 OHLCV 数据。

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            adjustment: 价格调整类型

        Returns:
            MarketData: 市场数据对象

        Raises:
            DataError: 如果数据获取失败
        """
        pass

    @abc.abstractmethod
    def get_fundamentals(
        self,
        symbol: str,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> MarketData:
        """
        获取基本面数据。

        Args:
            symbol: 股票代码
            as_of_date: 截至日期，如果为 None 则获取最新数据

        Returns:
            MarketData: 市场数据对象

        Raises:
            DataError: 如果数据获取失败
        """
        pass

    @abc.abstractmethod
    def get_market_status(self) -> Dict[str, Any]:
        """
        获取市场状态。

        Returns:
            Dict[str, Any]: 市场状态信息
        """
        pass

    @abc.abstractmethod
    def search_symbols(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        搜索股票代码。

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        pass

    def get_data(self, request: MarketDataRequest) -> MarketData:
        """
        获取市场数据（通用方法）。

        Args:
            request: 市场数据请求

        Returns:
            MarketData: 市场数据对象
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(request)

        # 尝试从缓存获取
        if self.cache_enabled and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"从缓存获取数据: {cache_key}")
                return MarketData(
                    symbol=request.symbol,
                    data_type=request.data_type,
                    data=cached_data,
                    interval=request.interval,
                    adjustment=request.adjustment,
                    metadata={"source": self.name, "cached": True},
                )

        # 根据数据类型调用相应的方法
        try:
            if request.data_type == MarketDataType.OHLCV:
                data = self.get_ohlcv(
                    symbol=request.symbol,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    interval=request.interval,
                    adjustment=request.adjustment,
                )
            elif request.data_type == MarketDataType.FUNDAMENTAL:
                data = self.get_fundamentals(
                    symbol=request.symbol,
                    as_of_date=request.end_date,
                )
            else:
                raise DataError(
                    f"不支持的数据类型: {request.data_type}",
                    data_source=self.name,
                    symbol=request.symbol,
                )

            # 缓存数据
            if self.cache_enabled and self.cache:
                try:
                    # 转换为 DataFrame 进行缓存
                    df = data.to_dataframe()
                    self.cache.set(cache_key, df, ttl=self.cache_ttl)
                    logger.debug(f"缓存数据: {cache_key}")
                except Exception as e:
                    logger.warning(f"缓存数据失败: {e}")

            # 添加元数据
            data.metadata.update({
                "source": self.name,
                "cached": False,
                "retrieved_at": datetime.now().isoformat(),
            })

            return data

        except Exception as e:
            raise DataError(
                f"获取数据失败: {request.symbol}",
                data_source=self.name,
                symbol=request.symbol,
                details={
                    "request": request.dict() if hasattr(request, "dict") else str(request),
                    "error": str(e),
                },
            )

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d",
        adjustment: AdjustmentType = AdjustmentType.ADJ,
        parallel: bool = False,
    ) -> Dict[str, MarketData]:
        """
        获取多个股票代码的数据。

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            adjustment: 价格调整类型
            parallel: 是否并行获取

        Returns:
            Dict[str, MarketData]: 股票代码到市场数据的映射
        """
        results = {}

        if parallel:
            # 并行获取（需要实现）
            logger.warning("并行获取尚未实现，使用串行方式")
            parallel = False

        if not parallel:
            # 串行获取
            for i, symbol in enumerate(symbols):
                try:
                    logger.info(f"获取数据 [{i+1}/{len(symbols)}]: {symbol}")
                    data = self.get_ohlcv(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                        adjustment=adjustment,
                    )
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"获取 {symbol} 数据失败: {e}")
                    results[symbol] = None

        return results

    def validate_symbol(self, symbol: str) -> bool:
        """
        验证股票代码是否有效。

        Args:
            symbol: 股票代码

        Returns:
            bool: 如果有效返回 True，否则返回 False
        """
        try:
            # 尝试搜索该代码
            results = self.search_symbols(symbol, limit=1)
            return len(results) > 0
        except Exception as e:
            logger.debug(f"验证股票代码失败 {symbol}: {e}")
            return False

    def get_available_intervals(self) -> List[str]:
        """
        获取可用的数据间隔。

        Returns:
            List[str]: 可用的数据间隔列表
        """
        # 默认实现，子类可以覆盖
        return ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]

    def get_available_markets(self) -> List[str]:
        """
        获取支持的市场。

        Returns:
            List[str]: 支持的市场列表
        """
        # 默认实现，子类可以覆盖
        return ["US", "HK", "CN"]

    def clear_cache(self) -> None:
        """清除缓存。"""
        if self.cache:
            self.cache.clear()
            logger.info(f"清除 {self.name} 数据提供器缓存")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息。

        Returns:
            Dict[str, Any]: 缓存信息
        """
        if self.cache:
            return self.cache.get_info()
        else:
            return {
                "enabled": False,
                "message": "缓存未启用",
            }

    def _generate_cache_key(self, request: MarketDataRequest) -> str:
        """
        生成缓存键。

        Args:
            request: 市场数据请求

        Returns:
            str: 缓存键
        """
        # 使用请求的字符串表示作为缓存键的基础
        request_dict = request.dict() if hasattr(request, "dict") else str(request)
        key_parts = [
            self.name,
            str(request_dict),
        ]

        # 添加时间戳以确保唯一性（按小时）
        current_hour = datetime.now().strftime("%Y%m%d%H")
        key_parts.append(current_hour)

        return ":".join(key_parts)

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        使用指数退避重试函数。

        Args:
            func: 要重试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            Any: 函数返回值

        Raises:
            Exception: 如果所有重试都失败
        """
        import time
        from functools import wraps

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (NetworkError, ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # 指数退避
                    logger.warning(
                        f"第 {attempt + 1} 次尝试失败，{delay} 秒后重试: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"所有 {self.max_retries} 次尝试都失败")
                    raise last_exception
            except Exception as e:
                # 对于其他异常，不重试
                raise e

        raise last_exception

    def _parse_date(self, date_value: Union[str, date, datetime]) -> date:
        """
        解析日期。

        Args:
            date_value: 日期值

        Returns:
            date: 解析后的日期对象
        """
        if isinstance(date_value, str):
            return DateUtils.parse_date(date_value)
        elif isinstance(date_value, datetime):
            return date_value.date()
        elif isinstance(date_value, date):
            return date_value
        else:
            raise DataError(
                f"无效的日期类型: {type(date_value)}",
                data_source=self.name,
                details={"date_value": str(date_value)},
            )

    def _format_date(self, date_obj: date) -> str:
        """
        格式化日期为字符串。

        Args:
            date_obj: 日期对象

        Returns:
            str: 格式化后的日期字符串 (YYYY-MM-DD)
        """
        return date_obj.strftime("%Y-%m-%d")

    def _ensure_dataframe_columns(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
    ) -> pd.DataFrame:
        """
        确保 DataFrame 包含必需的列。

        Args:
            df: 数据框
            required_columns: 必需的列名列表

        Returns:
            pd.DataFrame: 确保包含必需列的数据框
        """
        df = df.copy()

        # 重命名列以标准化
        column_mapping = {
            # 开高低收成交量
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            # 其他常见列名
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
            "ADJ_CLOSE": "adj_close",
        }

        df.rename(columns=column_mapping, inplace=True)

        # 检查缺失的列
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            logger.warning(f"DataFrame 缺少列: {missing_columns}")

            # 尝试使用默认值填充缺失列
            for col in missing_columns:
                if col == "volume":
                    df[col] = 0
                elif col == "adj_close":
                    df[col] = df.get("close", 0)
                else:
                    df[col] = 0

        return df


# 导出
__all__ = [
    "DataProvider",
]
"""
Yahoo Finance 数据提供器。

这个模块提供了从 Yahoo Finance 获取美股数据的实现。
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, date, timedelta
import time

from ..base import DataProvider
from ..market_data import (
    MarketData,
    MarketDataRequest,
    MarketDataType,
    AdjustmentType,
    OHLCV,
    FundamentalData,
    MarketStatus,
)
from ...utils.logger import logger
from ...utils.exceptions import DataError, NetworkError
from ...utils.date_utils import DateUtils
from ...utils.validation import Validator, DataCleaner


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance 数据提供器。"""

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        retry_delay: int = 1,
        request_delay: float = 0.1,  # 请求之间的延迟，避免被限制
    ):
        """
        初始化 Yahoo Finance 数据提供器。

        Args:
            cache_enabled: 是否启用缓存
            cache_ttl: 缓存过期时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            request_delay: 请求之间的延迟（秒）
        """
        super().__init__(
            name="yahoo_finance",
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        self.request_delay = request_delay
        self.last_request_time = 0

        logger.info("Yahoo Finance 数据提供器初始化完成")

    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d",
        adjustment: AdjustmentType = AdjustmentType.ADJ,
    ) -> MarketData:
        """
        从 Yahoo Finance 获取 OHLCV 数据。

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
        # 验证参数
        symbol = Validator.validate_stock_symbol(symbol, "symbol", market="US")

        # 解析日期
        start_date_obj = self._parse_date(start_date)
        end_date_obj = self._parse_date(end_date)

        # 验证日期范围
        if end_date_obj <= start_date_obj:
            raise DataError(
                "结束日期必须晚于开始日期",
                data_source=self.name,
                symbol=symbol,
                details={
                    "start_date": self._format_date(start_date_obj),
                    "end_date": self._format_date(end_date_obj),
                },
            )

        # 检查日期范围是否合理
        days_diff = (end_date_obj - start_date_obj).days
        if days_diff > 365 * 20:  # 20年
            logger.warning(f"请求的日期范围较大: {days_diff} 天")

        try:
            # 添加请求延迟
            self._throttle_requests()

            # 下载数据
            logger.info(
                f"从 Yahoo Finance 下载数据: {symbol} "
                f"({self._format_date(start_date_obj)} 到 {self._format_date(end_date_obj)})"
            )

            ticker = yf.Ticker(symbol)

            # 根据调整类型设置参数
            if adjustment == AdjustmentType.RAW:
                auto_adjust = False
            else:
                auto_adjust = True

            # 下载数据
            df = ticker.history(
                start=start_date_obj,
                end=end_date_obj + timedelta(days=1),  # yfinance 的 end 是 exclusive
                interval=interval,
                auto_adjust=auto_adjust,
                actions=True,  # 包含股息和拆股信息
            )

            if df.empty:
                raise DataError(
                    "未找到数据",
                    data_source=self.name,
                    symbol=symbol,
                    details={
                        "start_date": self._format_date(start_date_obj),
                        "end_date": self._format_date(end_date_obj),
                        "interval": interval,
                    },
                )

            # 重命名和标准化列
            df = self._ensure_dataframe_columns(
                df,
                required_columns=["open", "high", "low", "close", "volume"],
            )

            # 添加调整后收盘价（如果不存在）
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]

            # 清洗数据
            df = DataCleaner.clean_dataframe(df)

            # 创建市场数据对象
            market_data = MarketData(
                symbol=symbol,
                data_type=MarketDataType.OHLCV,
                data=df,
                interval=interval,
                adjustment=adjustment,
                metadata={
                    "source": self.name,
                    "symbol": symbol,
                    "start_date": self._format_date(start_date_obj),
                    "end_date": self._format_date(end_date_obj),
                    "interval": interval,
                    "adjustment": adjustment.value,
                    "rows": len(df),
                    "retrieved_at": datetime.now().isoformat(),
                },
            )

            logger.info(
                f"成功获取 {symbol} 数据: {len(df)} 行, "
                f"日期范围: {df.index[0].date()} 到 {df.index[-1].date()}"
            )

            return market_data

        except DataError:
            raise
        except Exception as e:
            raise DataError(
                f"从 Yahoo Finance 获取数据失败",
                data_source=self.name,
                symbol=symbol,
                details={
                    "start_date": self._format_date(start_date_obj),
                    "end_date": self._format_date(end_date_obj),
                    "interval": interval,
                    "error": str(e),
                },
            )

    def get_fundamentals(
        self,
        symbol: str,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> MarketData:
        """
        从 Yahoo Finance 获取基本面数据。

        Args:
            symbol: 股票代码
            as_of_date: 截至日期，如果为 None 则获取最新数据

        Returns:
            MarketData: 市场数据对象

        Raises:
            DataError: 如果数据获取失败
        """
        # 验证参数
        symbol = Validator.validate_stock_symbol(symbol, "symbol", market="US")

        try:
            # 添加请求延迟
            self._throttle_requests()

            logger.info(f"从 Yahoo Finance 获取 {symbol} 的基本面数据")

            ticker = yf.Ticker(symbol)

            # 获取基本信息
            info = ticker.info

            if not info:
                raise DataError(
                    "未找到基本面数据",
                    data_source=self.name,
                    symbol=symbol,
                )

            # 解析日期
            if as_of_date is None:
                as_of_date_obj = date.today()
            else:
                as_of_date_obj = self._parse_date(as_of_date)

            # 创建基本面数据对象
            fundamental_data = FundamentalData(
                symbol=symbol,
                date=as_of_date_obj,
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                pb_ratio=info.get("priceToBook"),
                ps_ratio=info.get("priceToSalesTrailing12Months"),
                dividend_yield=info.get("dividendYield"),
                revenue=info.get("totalRevenue"),
                net_income=info.get("netIncomeToCommon"),
                gross_profit=info.get("grossProfits"),
                operating_income=info.get("operatingIncome"),
                total_assets=info.get("totalAssets"),
                total_liabilities=info.get("totalLiab"),
                total_equity=info.get("totalStockholderEquity"),
                current_ratio=info.get("currentRatio"),
                debt_to_equity=info.get("debtToEquity"),
                roe=info.get("returnOnEquity"),
                roa=info.get("returnOnAssets"),
                roi=info.get("returnOnInvestment"),
            )

            # 创建市场数据对象
            market_data = MarketData(
                symbol=symbol,
                data_type=MarketDataType.FUNDAMENTAL,
                data=[fundamental_data],
                interval="1d",
                adjustment=AdjustmentType.RAW,
                metadata={
                    "source": self.name,
                    "symbol": symbol,
                    "as_of_date": self._format_date(as_of_date_obj),
                    "retrieved_at": datetime.now().isoformat(),
                },
            )

            logger.info(f"成功获取 {symbol} 的基本面数据")

            return market_data

        except DataError:
            raise
        except Exception as e:
            raise DataError(
                f"从 Yahoo Finance 获取基本面数据失败",
                data_source=self.name,
                symbol=symbol,
                details={"error": str(e)},
            )

    def get_market_status(self) -> Dict[str, Any]:
        """
        获取市场状态。

        Returns:
            Dict[str, Any]: 市场状态信息
        """
        try:
            # 添加请求延迟
            self._throttle_requests()

            logger.debug("获取市场状态")

            # 使用 SPY（标普500 ETF）作为市场状态指标
            ticker = yf.Ticker("SPY")

            # 获取最新数据
            today = date.today()
            df = ticker.history(period="1d")

            if df.empty:
                # 尝试获取最近的数据
                df = ticker.history(period="5d")

            if df.empty:
                return {
                    "status": MarketStatus.CLOSED,
                    "last_update": datetime.now().isoformat(),
                    "message": "无法获取市场数据",
                }

            # 判断市场状态
            last_trade_date = df.index[-1].date()
            now = datetime.now()

            if last_trade_date == today:
                # 今天有交易
                last_trade_time = df.index[-1].to_pydatetime()

                # 检查是否在交易时间内
                market_open = datetime.combine(today, datetime.min.time().replace(hour=9, minute=30))
                market_close = datetime.combine(today, datetime.min.time().replace(hour=16, minute=0))

                if market_open <= now <= market_close:
                    status = MarketStatus.OPEN
                elif now < market_open:
                    status = MarketStatus.PRE_MARKET
                elif now > market_close:
                    status = MarketStatus.AFTER_HOURS
                else:
                    status = MarketStatus.CLOSED
            else:
                # 今天没有交易，可能是节假日或周末
                if DateUtils.is_holiday(today, country="US"):
                    status = MarketStatus.HOLIDAY
                else:
                    status = MarketStatus.CLOSED

            return {
                "status": status,
                "last_trade_date": last_trade_date.isoformat(),
                "last_trade_time": last_trade_time.isoformat() if 'last_trade_time' in locals() else None,
                "last_price": float(df["Close"].iloc[-1]) if not df.empty else None,
                "last_update": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"获取市场状态失败: {e}")
            return {
                "status": MarketStatus.CLOSED,
                "last_update": datetime.now().isoformat(),
                "error": str(e),
            }

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
        try:
            # 添加请求延迟
            self._throttle_requests()

            logger.debug(f"搜索股票代码: {query}")

            # 使用 yfinance 的搜索功能
            search_results = yf.Ticker(query)

            # 获取基本信息
            info = search_results.info

            if not info:
                return []

            # 构建结果
            result = {
                "symbol": info.get("symbol", query),
                "name": info.get("longName") or info.get("shortName", ""),
                "exchange": info.get("exchange", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency", "USD"),
            }

            return [result]

        except Exception as e:
            logger.error(f"搜索股票代码失败: {e}")
            return []

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
        获取多个股票代码的数据（优化版本）。

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
        # 验证所有股票代码
        validated_symbols = []
        for symbol in symbols:
            try:
                validated_symbol = Validator.validate_stock_symbol(symbol, "symbol", market="US")
                validated_symbols.append(validated_symbol)
            except Exception as e:
                logger.warning(f"无效的股票代码 {symbol}: {e}")

        if not validated_symbols:
            return {}

        # 解析日期
        start_date_obj = self._parse_date(start_date)
        end_date_obj = self._parse_date(end_date)

        try:
            # 添加请求延迟
            self._throttle_requests()

            logger.info(
                f"批量下载 {len(validated_symbols)} 个股票的数据: "
                f"{self._format_date(start_date_obj)} 到 {self._format_date(end_date_obj)}"
            )

            # 使用 yfinance 的批量下载功能
            df = yf.download(
                tickers=validated_symbols,
                start=start_date_obj,
                end=end_date_obj + timedelta(days=1),  # yfinance 的 end 是 exclusive
                interval=interval,
                auto_adjust=(adjustment != AdjustmentType.RAW),
                group_by="ticker",
                progress=False,
            )

            results = {}

            for symbol in validated_symbols:
                try:
                    # 提取单个股票的数据
                    if len(validated_symbols) == 1:
                        symbol_df = df
                    else:
                        # 多级列索引
                        if symbol in df.columns.levels[0]:
                            symbol_df = df[symbol]
                        else:
                            logger.warning(f"未找到 {symbol} 的数据")
                            results[symbol] = None
                            continue

                    if symbol_df.empty:
                        logger.warning(f"{symbol} 数据为空")
                        results[symbol] = None
                        continue

                    # 重命名和标准化列
                    symbol_df = self._ensure_dataframe_columns(
                        symbol_df,
                        required_columns=["open", "high", "low", "close", "volume"],
                    )

                    # 添加调整后收盘价（如果不存在）
                    if "adj_close" not in symbol_df.columns:
                        symbol_df["adj_close"] = symbol_df["close"]

                    # 清洗数据
                    symbol_df = DataCleaner.clean_dataframe(symbol_df)

                    # 创建市场数据对象
                    market_data = MarketData(
                        symbol=symbol,
                        data_type=MarketDataType.OHLCV,
                        data=symbol_df,
                        interval=interval,
                        adjustment=adjustment,
                        metadata={
                            "source": self.name,
                            "symbol": symbol,
                            "start_date": self._format_date(start_date_obj),
                            "end_date": self._format_date(end_date_obj),
                            "interval": interval,
                            "adjustment": adjustment.value,
                            "rows": len(symbol_df),
                            "retrieved_at": datetime.now().isoformat(),
                        },
                    )

                    results[symbol] = market_data

                    logger.debug(f"成功获取 {symbol} 数据: {len(symbol_df)} 行")

                except Exception as e:
                    logger.error(f"处理 {symbol} 数据失败: {e}")
                    results[symbol] = None

            success_count = sum(1 for v in results.values() if v is not None)
            logger.info(f"批量下载完成: 成功 {success_count}/{len(validated_symbols)}")

            return results

        except Exception as e:
            raise DataError(
                f"批量下载数据失败",
                data_source=self.name,
                details={
                    "symbols": validated_symbols,
                    "start_date": self._format_date(start_date_obj),
                    "end_date": self._format_date(end_date_obj),
                    "error": str(e),
                },
            )

    def get_available_intervals(self) -> List[str]:
        """
        获取可用的数据间隔。

        Returns:
            List[str]: 可用的数据间隔列表
        """
        return ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]

    def get_available_markets(self) -> List[str]:
        """
        获取支持的市场。

        Returns:
            List[str]: 支持的市场列表
        """
        return ["US"]

    def _throttle_requests(self) -> None:
        """限制请求频率。"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()


# Convenience function
def create_yahoo_finance_provider(
    cache_enabled: bool = True,
    cache_ttl: int = 3600,
    max_retries: int = 3,
    retry_delay: int = 1,
    request_delay: float = 0.5,
) -> YahooFinanceProvider:
    """
    Create a YahooFinanceProvider instance.

    Args:
        cache_enabled: Whether to enable caching
        cache_ttl: Cache TTL in seconds
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        request_delay: Delay between requests to avoid rate limiting

    Returns:
        YahooFinanceProvider instance
    """
    return YahooFinanceProvider(
        cache_enabled=cache_enabled,
        cache_ttl=cache_ttl,
        max_retries=max_retries,
        retry_delay=retry_delay,
        request_delay=request_delay,
    )


# 导出
__all__ = [
    "YahooFinanceProvider",
    "create_yahoo_finance_provider",
]
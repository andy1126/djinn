"""
AKShare Data Provider for Chinese and Hong Kong Markets.

This provider uses the AKShare library to fetch data for A-shares and H-shares.
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Any
import warnings
import time

from ..base import DataProvider
from ..market_data import (
    MarketData,
    MarketDataType,
    AdjustmentType,
    OHLCV,
    FundamentalData
)
from ...utils.exceptions import DataError, NetworkError
from ...utils.logger import logger
from ...utils.date_utils import DateUtils


class AKShareProvider(DataProvider):
    """
    AKShare data provider for Chinese and Hong Kong markets.

    Supports:
    - A-shares (Shanghai/Shenzhen)
    - H-shares (Hong Kong)
    - ETFs, Funds, and other Chinese financial instruments
    """

    # Market mapping
    MARKET_MAPPING = {
        "SH": "sh",      # Shanghai
        "SZ": "sz",      # Shenzhen
        "BJ": "bj",      # Beijing (new third board)
        "HK": "hk",      # Hong Kong
    }

    # Interval mapping
    INTERVAL_MAPPING = {
        "1d": "daily",
        "1wk": "weekly",
        "1mo": "monthly",
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "60min",
    }

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        retry_delay: int = 2,
        rate_limit_delay: float = 0.5,  # Delay between requests to avoid rate limiting
    ):
        """
        Initialize AKShare data provider.

        Args:
            cache_enabled: Whether to enable caching
            cache_ttl: Cache TTL in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        super().__init__(
            name="akshare",
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

        # Initialize AKShare session
        self._init_akshare()

        logger.info("Initialized AKShare provider for Chinese/HK markets")

    def _init_akshare(self) -> None:
        """Initialize AKShare library."""
        try:
            # Check if AKShare is available
            import akshare as ak
            self.ak = ak
            logger.debug("AKShare library loaded successfully")
        except ImportError as e:
            raise ImportError(
                "AKShare library is not installed. "
                "Install with: pip install akshare"
            ) from e

    def _throttle_request(self) -> None:
        """Throttle requests to avoid rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _parse_symbol(self, symbol: str) -> Dict[str, str]:
        """
        Parse symbol to extract market and code.

        Args:
            symbol: Symbol in format like "000001.SZ" or "00700.HK"

        Returns:
            Dictionary with 'code' and 'market' keys

        Raises:
            DataError: If symbol format is invalid
        """
        if "." not in symbol:
            raise DataError(
                f"Invalid symbol format: {symbol}. Expected format: 'CODE.MARKET'",
                data_source=self.name,
                symbol=symbol,
            )

        code, market = symbol.split(".", 1)
        market = market.upper()

        if market not in self.MARKET_MAPPING:
            raise DataError(
                f"Unsupported market: {market}. Supported markets: {list(self.MARKET_MAPPING.keys())}",
                data_source=self.name,
                symbol=symbol,
            )

        return {
            "code": code,
            "market": market,
            "ak_market": self.MARKET_MAPPING[market]
        }

    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d",
        adjustment: AdjustmentType = AdjustmentType.ADJ,
    ) -> MarketData:
        """
        Get OHLCV data for a symbol.

        Args:
            symbol: Stock symbol (e.g., "000001.SZ", "00700.HK")
            start_date: Start date
            end_date: End date
            interval: Data interval
            adjustment: Price adjustment type

        Returns:
            MarketData object with OHLCV data

        Raises:
            DataError: If data retrieval fails
        """
        try:
            # Throttle request
            self._throttle_request()

            # Parse symbol
            symbol_info = self._parse_symbol(symbol)
            code = symbol_info["code"]
            market = symbol_info["market"]
            ak_market = symbol_info["ak_market"]

            # Parse dates
            start_date_obj = self._parse_date(start_date)
            end_date_obj = self._parse_date(end_date)

            # Format dates for AKShare
            start_str = start_date_obj.strftime("%Y%m%d")
            end_str = end_date_obj.strftime("%Y%m%d")

            logger.info(
                f"Fetching OHLCV data for {symbol} ({market}) "
                f"from {start_str} to {end_str}, interval: {interval}"
            )

            # Map interval
            ak_interval = self.INTERVAL_MAPPING.get(interval)
            if not ak_interval:
                raise DataError(
                    f"Unsupported interval: {interval}. "
                    f"Supported intervals: {list(self.INTERVAL_MAPPING.keys())}",
                    data_source=self.name,
                    symbol=symbol,
                )

            # Fetch data based on market
            df = self._fetch_ohlcv_data(
                code=code,
                market=market,
                ak_market=ak_market,
                start_date=start_str,
                end_date=end_str,
                period=ak_interval,
                adjustment=adjustment,
            )

            if df is None or df.empty:
                raise DataError(
                    f"No data found for {symbol} in the specified date range",
                    data_source=self.name,
                    symbol=symbol,
                    details={
                        "start_date": start_str,
                        "end_date": end_str,
                        "interval": interval,
                    },
                )

            # Standardize column names
            df = self._standardize_dataframe(df)

            # Apply adjustment if needed
            if adjustment == AdjustmentType.ADJ:
                df = self._apply_adjustment(df, symbol)

            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                data_type=MarketDataType.OHLCV,
                data=df,
                interval=interval,
                adjustment=adjustment,
                metadata={
                    "source": self.name,
                    "market": market,
                    "code": code,
                    "start_date": start_str,
                    "end_date": end_str,
                    "interval": interval,
                    "adjustment": adjustment.value,
                    "rows": len(df),
                },
            )

            logger.info(f"Retrieved {len(df)} rows for {symbol}")
            return market_data

        except DataError:
            raise
        except Exception as e:
            raise DataError(
                f"Failed to get OHLCV data for {symbol}",
                data_source=self.name,
                symbol=symbol,
                details={"error": str(e)},
            )

    def _fetch_ohlcv_data(
        self,
        code: str,
        market: str,
        ak_market: str,
        start_date: str,
        end_date: str,
        period: str,
        adjustment: AdjustmentType,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data using appropriate AKShare function."""
        try:
            if market == "HK":
                # Hong Kong stocks
                df = self.ak.stock_hk_hist(
                    symbol=code,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjustment.value if adjustment == AdjustmentType.ADJ else "",
                )
            elif market in ["SH", "SZ", "BJ"]:
                # A-shares
                symbol = f"{code}.{ak_market}"
                if period == "daily":
                    df = self.ak.stock_zh_a_hist(
                        symbol=symbol,
                        period=period,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjustment.value if adjustment == AdjustmentType.ADJ else "qfq",
                    )
                else:
                    # For intraday data
                    df = self.ak.stock_zh_a_hist_min_em(
                        symbol=symbol,
                        period=period,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjustment.value if adjustment == AdjustmentType.ADJ else "",
                    )
            else:
                raise DataError(
                    f"Unsupported market for OHLCV: {market}",
                    data_source=self.name,
                )

            return df

        except Exception as e:
            logger.error(f"AKShare fetch failed for {code}.{market}: {e}")
            return None

    def get_fundamentals(
        self,
        symbol: str,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> MarketData:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: Stock symbol
            as_of_date: As-of date (optional)

        Returns:
            MarketData object with fundamental data

        Raises:
            DataError: If data retrieval fails
        """
        try:
            # Throttle request
            self._throttle_request()

            # Parse symbol
            symbol_info = self._parse_symbol(symbol)
            code = symbol_info["code"]
            market = symbol_info["market"]
            ak_market = symbol_info["ak_market"]

            logger.info(f"Fetching fundamental data for {symbol} ({market})")

            # Fetch fundamental data based on market
            df = self._fetch_fundamental_data(code, market, ak_market, as_of_date)

            if df is None or df.empty:
                raise DataError(
                    f"No fundamental data found for {symbol}",
                    data_source=self.name,
                    symbol=symbol,
                )

            # Standardize column names
            df = self._standardize_fundamental_dataframe(df)

            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                data_type=MarketDataType.FUNDAMENTAL,
                data=df,
                interval="1d",  # Fundamental data is daily
                adjustment=AdjustmentType.RAW,
                metadata={
                    "source": self.name,
                    "market": market,
                    "code": code,
                    "as_of_date": as_of_date.isoformat() if as_of_date else None,
                    "rows": len(df),
                },
            )

            logger.info(f"Retrieved fundamental data for {symbol}: {len(df)} rows")
            return market_data

        except DataError:
            raise
        except Exception as e:
            raise DataError(
                f"Failed to get fundamental data for {symbol}",
                data_source=self.name,
                symbol=symbol,
                details={"error": str(e)},
            )

    def _fetch_fundamental_data(
        self,
        code: str,
        market: str,
        ak_market: str,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch fundamental data using appropriate AKShare function."""
        try:
            if market in ["SH", "SZ", "BJ"]:
                # A-share fundamentals
                symbol = f"{code}.{ak_market}"

                # Try to get balance sheet data
                df = self.ak.stock_financial_report_sina(
                    stock=symbol,
                    symbol="资产负债表"
                )
            elif market == "HK":
                # Hong Kong fundamentals
                df = self.ak.stock_financial_hk_report_em(
                    stock=code,
                    symbol="资产负债表"
                )
            else:
                logger.warning(f"Fundamental data not supported for market: {market}")
                return None

            return df

        except Exception as e:
            logger.error(f"Failed to fetch fundamental data for {code}.{market}: {e}")
            return None

    def get_market_status(self) -> Dict[str, Any]:
        """
        Get market status information.

        Returns:
            Dictionary with market status information
        """
        try:
            self._throttle_request()

            status = {
                "timestamp": datetime.now().isoformat(),
                "markets": {},
                "overall_status": "unknown",
            }

            # Check Shanghai market
            try:
                sh_status = self.ak.stock_szse_summary()
                status["markets"]["shanghai"] = {
                    "status": "open" if not sh_status.empty else "closed",
                    "last_updated": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"Failed to get Shanghai market status: {e}")
                status["markets"]["shanghai"] = {"status": "error", "error": str(e)}

            # Check Shenzhen market
            try:
                sz_status = self.ak.stock_sse_summary()
                status["markets"]["shenzhen"] = {
                    "status": "open" if not sz_status.empty else "closed",
                    "last_updated": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"Failed to get Shenzhen market status: {e}")
                status["markets"]["shenzhen"] = {"status": "error", "error": str(e)}

            # Determine overall status
            market_statuses = [m["status"] for m in status["markets"].values()]
            if "open" in market_statuses:
                status["overall_status"] = "open"
            elif all(s == "closed" for s in market_statuses):
                status["overall_status"] = "closed"
            elif any(s == "error" for s in market_statuses):
                status["overall_status"] = "partial_error"
            else:
                status["overall_status"] = "unknown"

            logger.debug(f"Market status retrieved: {status['overall_status']}")
            return status

        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e),
            }

    def search_symbols(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for symbols.

        Args:
            query: Search query (can be code, name, or pinyin)
            limit: Maximum number of results

        Returns:
            List of dictionaries with symbol information
        """
        try:
            self._throttle_request()

            results = []

            # Search A-shares
            try:
                a_share_stocks = self.ak.stock_info_a_code_name()
                if a_share_stocks is not None and not a_share_stocks.empty:
                    # Filter by query
                    mask = (
                        a_share_stocks["code"].astype(str).str.contains(query, case=False) |
                        a_share_stocks["name"].str.contains(query, case=False)
                    )
                    filtered = a_share_stocks[mask].head(limit)

                    for _, row in filtered.iterrows():
                        code = str(row["code"]).zfill(6)
                        # Determine market based on code prefix
                        if code.startswith(("6", "9")):
                            market = "SH"
                        elif code.startswith(("0", "3")):
                            market = "SZ"
                        elif code.startswith(("4", "8")):
                            market = "BJ"
                        else:
                            market = "SH"  # default

                        results.append({
                            "symbol": f"{code}.{market}",
                            "code": code,
                            "name": row["name"],
                            "market": market,
                            "type": "A-share",
                            "exchange": "Shanghai" if market == "SH" else "Shenzhen" if market == "SZ" else "Beijing",
                        })
            except Exception as e:
                logger.warning(f"Failed to search A-shares: {e}")

            # Search Hong Kong stocks
            try:
                hk_stocks = self.ak.stock_hk_spot_em()
                if hk_stocks is not None and not hk_stocks.empty:
                    # Filter by query
                    mask = (
                        hk_stocks["代码"].astype(str).str.contains(query, case=False) |
                        hk_stocks["名称"].str.contains(query, case=False)
                    )
                    filtered = hk_stocks[mask].head(limit)

                    for _, row in filtered.iterrows():
                        results.append({
                            "symbol": f"{row['代码']}.HK",
                            "code": row["代码"],
                            "name": row["名称"],
                            "market": "HK",
                            "type": "H-share",
                            "exchange": "Hong Kong",
                        })
            except Exception as e:
                logger.warning(f"Failed to search H-shares: {e}")

            # Remove duplicates and limit results
            seen_symbols = set()
            unique_results = []
            for result in results:
                if result["symbol"] not in seen_symbols:
                    seen_symbols.add(result["symbol"])
                    unique_results.append(result)
                if len(unique_results) >= limit:
                    break

            logger.info(f"Symbol search for '{query}' returned {len(unique_results)} results")
            return unique_results

        except Exception as e:
            logger.error(f"Failed to search symbols: {e}")
            return []

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame column names and format."""
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            date_cols = ["日期", "date", "Date", "交易时间", "时间"]
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break

        # Standardize column names
        column_mapping = {
            # Chinese column names
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交额": "volume",
            "成交量": "volume",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
            # English column names
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            "Amount": "amount",
        }

        df.rename(columns=column_mapping, inplace=True)

        # Ensure required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                if col == "volume":
                    df[col] = 0
                else:
                    # Try to find similar columns
                    for df_col in df.columns:
                        if col in df_col.lower():
                            df.rename(columns={df_col: col}, inplace=True)
                            break
                    else:
                        df[col] = df.get("close", 0)

        # Convert numeric columns
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by date
        df.sort_index(inplace=True)

        # Drop duplicates
        df = df[~df.index.duplicated(keep="first")]

        return df

    def _standardize_fundamental_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize fundamental DataFrame."""
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # Ensure date index if available
        date_cols = ["报告期", "公告日期", "date", "Date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break

        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except:
                pass

        return df

    def _apply_adjustment(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply price adjustment to OHLCV data.

        Note: AKShare usually returns adjusted prices directly,
        so this is mostly for backward compatibility.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol

        Returns:
            Adjusted DataFrame
        """
        # AKShare usually provides adjusted prices directly
        # If adjustment is needed, we can implement it here
        # For now, just return the dataframe as-is
        return df

    def get_available_intervals(self) -> List[str]:
        """Get available data intervals."""
        return list(self.INTERVAL_MAPPING.keys())

    def get_available_markets(self) -> List[str]:
        """Get available markets."""
        return list(self.MARKET_MAPPING.keys())

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists."""
        try:
            symbol_info = self._parse_symbol(symbol)
            # Try to get a small amount of data to validate
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            df = self._fetch_ohlcv_data(
                code=symbol_info["code"],
                market=symbol_info["market"],
                ak_market=symbol_info["ak_market"],
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                period="daily",
                adjustment=AdjustmentType.RAW,
            )

            return df is not None and not df.empty
        except Exception as e:
            logger.debug(f"Symbol validation failed for {symbol}: {e}")
            return False


# Convenience function
def create_akshare_provider(
    cache_enabled: bool = True,
    cache_ttl: int = 3600,
    max_retries: int = 3,
    retry_delay: int = 2,
    rate_limit_delay: float = 0.5,
) -> AKShareProvider:
    """
    Create an AKShareProvider instance.

    Args:
        cache_enabled: Whether to enable caching
        cache_ttl: Cache TTL in seconds
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        rate_limit_delay: Delay between requests to avoid rate limiting

    Returns:
        AKShareProvider instance
    """
    return AKShareProvider(
        cache_enabled=cache_enabled,
        cache_ttl=cache_ttl,
        max_retries=max_retries,
        retry_delay=retry_delay,
        rate_limit_delay=rate_limit_delay,
    )


# Export
__all__ = [
    "AKShareProvider",
    "create_akshare_provider",
]
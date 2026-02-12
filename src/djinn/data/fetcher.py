"""
Djinn 数据获取器。

这个模块提供了统一的数据获取接口，支持多数据源和智能缓存。
"""

from typing import Optional, Dict, Any, List, Union
from datetime import datetime, date
import pandas as pd

from .base import DataProvider
from .providers.yahoo_finance import YahooFinanceProvider
from .market_data import (
    MarketData,
    MarketDataRequest,
    MarketDataType,
    AdjustmentType,
)
from ..utils.logger import logger
from ..utils.exceptions import DataError
from ..utils.config import config_manager, DataSourceType
from ..utils.file_utils import CacheManager


class DataFetcher:
    """数据获取器。"""

    def __init__(
        self,
        cache_enabled: bool = True,
        default_provider: Optional[str] = None,
    ):
        """
        初始化数据获取器。

        Args:
            cache_enabled: 是否启用缓存
            default_provider: 默认数据提供器
        """
        self.cache_enabled = cache_enabled
        self.default_provider = default_provider or DataSourceType.YAHOO_FINANCE.value

        # 初始化数据提供器
        self.providers: Dict[str, DataProvider] = {}
        self._init_providers()

        # 初始化缓存管理器
        if cache_enabled:
            cache_dir = config_manager.get_setting("cache_dir", "./.cache/djinn")
            self.cache = CacheManager(
                cache_dir=cache_dir,
                default_ttl=config_manager.get_setting("cache_ttl", 3600),
            )
        else:
            self.cache = None

        logger.info(f"数据获取器初始化完成，默认提供器: {self.default_provider}")

    def _init_providers(self) -> None:
        """初始化数据提供器。"""
        # 注册 Yahoo Finance 提供器
        yahoo_provider = YahooFinanceProvider(
            cache_enabled=self.cache_enabled,
            cache_ttl=config_manager.get_setting("cache_ttl", 3600),
        )
        self.register_provider(DataSourceType.YAHOO_FINANCE.value, yahoo_provider)

        # 可以根据需要注册其他提供器
        # TODO: 注册 AKShare 提供器
        # TODO: 注册 Tushare 提供器

    def register_provider(self, name: str, provider: DataProvider) -> None:
        """
        注册数据提供器。

        Args:
            name: 提供器名称
            provider: 数据提供器实例
        """
        self.providers[name] = provider
        logger.info(f"注册数据提供器: {name}")

    def get_provider(self, name: Optional[str] = None) -> DataProvider:
        """
        获取数据提供器。

        Args:
            name: 提供器名称，如果为 None 则使用默认提供器

        Returns:
            DataProvider: 数据提供器实例

        Raises:
            DataError: 如果提供器不存在
        """
        provider_name = name or self.default_provider

        if provider_name not in self.providers:
            raise DataError(
                f"数据提供器不存在: {provider_name}",
                data_source=provider_name,
                details={"available_providers": list(self.providers.keys())},
            )

        return self.providers[provider_name]

    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d",
        adjustment: AdjustmentType = AdjustmentType.ADJ,
        provider: Optional[str] = None,
        use_cache: bool = True,
    ) -> MarketData:
        """
        获取 OHLCV 数据。

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            adjustment: 价格调整类型
            provider: 数据提供器名称，如果为 None 则使用默认提供器
            use_cache: 是否使用缓存

        Returns:
            MarketData: 市场数据对象
        """
        # 创建请求
        request = MarketDataRequest(
            symbol=symbol,
            start_date=start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d"),
            end_date=end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d"),
            interval=interval,
            adjustment=adjustment,
            data_type=MarketDataType.OHLCV,
        )

        return self.get_data(request, provider, use_cache)

    def get_fundamentals(
        self,
        symbol: str,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        provider: Optional[str] = None,
        use_cache: bool = True,
    ) -> MarketData:
        """
        获取基本面数据。

        Args:
            symbol: 股票代码
            as_of_date: 截至日期，如果为 None 则获取最新数据
            provider: 数据提供器名称，如果为 None 则使用默认提供器
            use_cache: 是否使用缓存

        Returns:
            MarketData: 市场数据对象
        """
        # 格式化日期
        if as_of_date is None:
            end_date_str = datetime.now().strftime("%Y-%m-%d")
        elif isinstance(as_of_date, str):
            end_date_str = as_of_date
        else:
            end_date_str = as_of_date.strftime("%Y-%m-%d")

        # 创建请求
        request = MarketDataRequest(
            symbol=symbol,
            start_date=end_date_str,  # 对于基本面数据，开始日期不重要
            end_date=end_date_str,
            interval="1d",
            adjustment=AdjustmentType.RAW,
            data_type=MarketDataType.FUNDAMENTAL,
        )

        return self.get_data(request, provider, use_cache)

    def get_data(
        self,
        request: MarketDataRequest,
        provider: Optional[str] = None,
        use_cache: bool = True,
    ) -> MarketData:
        """
        获取市场数据。

        Args:
            request: 市场数据请求
            provider: 数据提供器名称，如果为 None 则使用默认提供器
            use_cache: 是否使用缓存

        Returns:
            MarketData: 市场数据对象
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(request, provider)

        # 尝试从缓存获取
        if self.cache_enabled and use_cache and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"从缓存获取数据: {cache_key}")
                return MarketData(
                    symbol=request.symbol,
                    data_type=request.data_type,
                    data=cached_data,
                    interval=request.interval,
                    adjustment=request.adjustment,
                    metadata={
                        "source": provider or self.default_provider,
                        "cached": True,
                        "cache_key": cache_key,
                    },
                )

        # 获取数据提供器
        data_provider = self.get_provider(provider)

        try:
            # 获取数据
            data = data_provider.get_data(request)

            # 缓存数据
            if self.cache_enabled and use_cache and self.cache:
                try:
                    # 转换为 DataFrame 进行缓存
                    df = data.to_dataframe()
                    self.cache.set(cache_key, df)
                    logger.debug(f"缓存数据: {cache_key}")
                except Exception as e:
                    logger.warning(f"缓存数据失败: {e}")

            # 更新元数据
            data.metadata.update({
                "cache_key": cache_key,
                "cached": False,
            })

            return data

        except Exception as e:
            # 如果主要提供器失败，尝试备用提供器
            if provider is None and len(self.providers) > 1:
                logger.warning(f"默认提供器失败，尝试备用提供器: {e}")

                # 尝试其他提供器
                for alt_provider_name, alt_provider in self.providers.items():
                    if alt_provider_name != self.default_provider:
                        try:
                            logger.info(f"尝试备用提供器: {alt_provider_name}")
                            data = alt_provider.get_data(request)

                            # 缓存数据
                            if self.cache_enabled and use_cache and self.cache:
                                try:
                                    df = data.to_dataframe()
                                    self.cache.set(cache_key, df)
                                except Exception as cache_error:
                                    logger.warning(f"缓存备用数据失败: {cache_error}")

                            # 更新元数据
                            data.metadata.update({
                                "source": alt_provider_name,
                                "fallback": True,
                                "original_error": str(e),
                                "cache_key": cache_key,
                                "cached": False,
                            })

                            logger.info(f"备用提供器成功: {alt_provider_name}")
                            return data

                        except Exception as alt_error:
                            logger.warning(f"备用提供器 {alt_provider_name} 也失败: {alt_error}")
                            continue

            # 所有提供器都失败
            raise DataError(
                f"获取数据失败: {request.symbol}",
                data_source=provider or self.default_provider,
                symbol=request.symbol,
                details={
                    "request": request.dict(),
                    "error": str(e),
                    "available_providers": list(self.providers.keys()),
                },
            )

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d",
        adjustment: AdjustmentType = AdjustmentType.ADJ,
        provider: Optional[str] = None,
        use_cache: bool = True,
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
            provider: 数据提供器名称
            use_cache: 是否使用缓存
            parallel: 是否并行获取

        Returns:
            Dict[str, MarketData]: 股票代码到市场数据的映射
        """
        # 获取数据提供器
        data_provider = self.get_provider(provider)

        # 检查是否有批量下载功能
        if hasattr(data_provider, "get_multiple_symbols"):
            try:
                return data_provider.get_multiple_symbols(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    adjustment=adjustment,
                    parallel=parallel,
                )
            except Exception as e:
                logger.warning(f"批量下载失败，回退到逐个下载: {e}")

        # 逐个下载
        results = {}
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"获取数据 [{i+1}/{len(symbols)}]: {symbol}")
                data = self.get_ohlcv(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    adjustment=adjustment,
                    provider=provider,
                    use_cache=use_cache,
                )
                results[symbol] = data
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")
                results[symbol] = None

        return results

    def get_market_status(
        self,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        获取市场状态。

        Args:
            provider: 数据提供器名称

        Returns:
            Dict[str, Any]: 市场状态信息
        """
        data_provider = self.get_provider(provider)
        return data_provider.get_market_status()

    def search_symbols(
        self,
        query: str,
        provider: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        搜索股票代码。

        Args:
            query: 搜索查询
            provider: 数据提供器名称
            limit: 返回结果数量限制

        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        data_provider = self.get_provider(provider)
        return data_provider.search_symbols(query, limit)

    def validate_symbol(
        self,
        symbol: str,
        provider: Optional[str] = None,
    ) -> bool:
        """
        验证股票代码是否有效。

        Args:
            symbol: 股票代码
            provider: 数据提供器名称

        Returns:
            bool: 如果有效返回 True，否则返回 False
        """
        data_provider = self.get_provider(provider)
        return data_provider.validate_symbol(symbol)

    def clear_cache(
        self,
        provider: Optional[str] = None,
    ) -> None:
        """
        清除缓存。

        Args:
            provider: 数据提供器名称，如果为 None 则清除所有缓存
        """
        if provider is None:
            # 清除所有提供器的缓存
            for prov in self.providers.values():
                if hasattr(prov, "clear_cache"):
                    prov.clear_cache()

            # 清除中央缓存
            if self.cache:
                self.cache.clear()

            logger.info("清除所有缓存")
        else:
            # 清除特定提供器的缓存
            data_provider = self.get_provider(provider)
            if hasattr(data_provider, "clear_cache"):
                data_provider.clear_cache()
                logger.info(f"清除 {provider} 缓存")

    def get_cache_info(
        self,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        获取缓存信息。

        Args:
            provider: 数据提供器名称，如果为 None 则获取所有缓存信息

        Returns:
            Dict[str, Any]: 缓存信息
        """
        if provider is None:
            # 获取所有提供器的缓存信息
            info = {
                "total_providers": len(self.providers),
                "providers": {},
            }

            for name, prov in self.providers.items():
                if hasattr(prov, "get_cache_info"):
                    info["providers"][name] = prov.get_cache_info()

            # 添加中央缓存信息
            if self.cache:
                info["central_cache"] = self.cache.get_info()

            return info
        else:
            # 获取特定提供器的缓存信息
            data_provider = self.get_provider(provider)
            if hasattr(data_provider, "get_cache_info"):
                return data_provider.get_cache_info()
            else:
                return {"enabled": False, "message": "该提供器不支持缓存信息"}

    def get_available_providers(self) -> List[str]:
        """
        获取可用的数据提供器。

        Returns:
            List[str]: 可用的数据提供器列表
        """
        return list(self.providers.keys())

    def get_provider_info(self, name: str) -> Dict[str, Any]:
        """
        获取数据提供器信息。

        Args:
            name: 提供器名称

        Returns:
            Dict[str, Any]: 提供器信息
        """
        data_provider = self.get_provider(name)

        info = {
            "name": name,
            "type": type(data_provider).__name__,
            "cache_enabled": getattr(data_provider, "cache_enabled", False),
            "available_intervals": data_provider.get_available_intervals(),
            "available_markets": data_provider.get_available_markets(),
        }

        return info

    def _generate_cache_key(
        self,
        request: MarketDataRequest,
        provider: Optional[str] = None,
    ) -> str:
        """
        生成缓存键。

        Args:
            request: 市场数据请求
            provider: 数据提供器名称

        Returns:
            str: 缓存键
        """
        provider_name = provider or self.default_provider

        # 使用请求的哈希作为缓存键
        import hashlib
        import json

        request_dict = request.dict()
        request_str = json.dumps(request_dict, sort_keys=True)

        key_parts = [
            "djinn",
            provider_name,
            hashlib.md5(request_str.encode()).hexdigest(),
        ]

        return ":".join(key_parts)


# 全局数据获取器实例
data_fetcher = DataFetcher(cache_enabled=True)


# 导出
__all__ = [
    "DataFetcher",
    "data_fetcher",
]
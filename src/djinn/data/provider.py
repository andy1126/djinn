"""DataProvider 抽象基类 + 注册表。

所有数据源(Yahoo / AkShare / Tushare / CSV)实现统一接口,由 :class:`ProviderRegistry`
按 ``supports()`` 优先级路由。provider 内部应处理限流与缓存命中。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import date

from djinn.data.market_data import MarketData
from djinn.data.schema import Adjust, Market
from djinn.utils.exceptions import SymbolNotFoundError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


class DataProvider(ABC):
    """数据提供器抽象。

    子类需实现 :meth:`get_ohlcv` 与 :meth:`supports`。
    """

    name: str = "base"
    market: Market

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: Adjust = Adjust.BACKWARD,
    ) -> MarketData:
        """获取 [start, end] 闭区间 OHLCV(已复权、已对齐交易日历)。"""

    @abstractmethod
    def supports(self, symbol: str, market: Market | None = None) -> bool:
        """是否能为该标的/市场提供数据。"""

    def get_benchmark(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: Adjust = Adjust.BACKWARD,
    ) -> MarketData:
        """获取基准行情(默认等同于普通标的,子类可覆写以使用专属接口)。"""
        return self.get_ohlcv(symbol, start, end, adjust)


class ProviderRegistry:
    """按注册顺序尝试多个 provider,首个 ``supports`` 命中者负责拉取。"""

    def __init__(self, providers: Iterable[DataProvider] | None = None) -> None:
        self._providers: list[DataProvider] = list(providers or [])

    def register(self, provider: DataProvider) -> None:
        self._providers.append(provider)
        _log.debug("注册 provider: %s", provider.name)

    @property
    def providers(self) -> list[DataProvider]:
        return list(self._providers)

    def resolve(self, symbol: str, market: Market | None = None) -> DataProvider:
        for p in self._providers:
            if p.supports(symbol, market):
                return p
        raise SymbolNotFoundError(
            f"无 provider 支持标的 {symbol!r}(market={market.value if market else 'auto'})"
        )

    def get_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: Adjust = Adjust.BACKWARD,
        market: Market | None = None,
    ) -> MarketData:
        provider = self.resolve(symbol, market)
        return provider.get_ohlcv(symbol, start, end, adjust)

"""DataProvider 抽象基类 + 注册表。

所有数据源(Yahoo / AkShare / Tushare / CSV)实现统一接口,由 :class:`ProviderRegistry`
按 ``supports()`` 优先级路由。provider 内部应处理限流与缓存命中。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import date
from typing import Any

import pandas as pd

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

    # ── 可选扩展(股票池 / 行业 / 基本面)────────────────────
    # 以下均为非抽象:provider 按需覆写,默认抛 NotImplementedError。
    # 横截面 alpha 层(factor / screen)依赖这些接口;仅做行情回测可不实现。

    def get_stock_list(self, market: Market | None = None) -> pd.DataFrame:
        """全市场股票列表。

        Returns:
            index=symbol,columns 至少含 ``name``、``market``。
        """
        raise NotImplementedError(f"{type(self).__name__} 不支持 get_stock_list")

    def get_index_components(self, index: str) -> list[str]:
        """指数成分股代码列表(djinn 标准后缀形式,如 ``000300.SH``)。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持 get_index_components")

    def get_index_component_names(self, index: str) -> dict[str, str]:
        """指数成分 symbol → 名称映射(与 :meth:`get_index_components` 同源)。

        无名称数据返回空 dict;不支持抛 NotImplementedError。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 get_index_component_names"
        )

    def search_symbols(
        self, query: str, market: Market | None = None
    ) -> list[tuple[str, str]]:
        """按代码 / 名称搜索标的,返回 ``(symbol, name)`` 列表。

        供股票搜索端点联想用;不支持抛 NotImplementedError。
        """
        raise NotImplementedError(f"{type(self).__name__} 不支持 search_symbols")

    def get_stock_name(self, symbol: str, market: Market | None = None) -> str:
        """单标的名称;不支持抛 NotImplementedError。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持 get_stock_name")

    def get_stock_price(self, symbol: str, market: Market | None = None) -> float:
        """单标的当前价;不支持抛 NotImplementedError。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持 get_stock_price")

    def get_industry_map(self, symbols: list[str]) -> dict[str, str]:
        """symbol → 行业名映射(缺失的 symbol 不在返回里)。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持 get_industry_map")

    def get_fundamentals(self, symbols: list[str], when: date) -> pd.DataFrame:
        """``when`` 当日截面基本面快照(point-in-time)。

        Returns:
            index=symbol,columns=规范化基本面字段(见 :mod:`djinn.data.schema`)。
        """
        raise NotImplementedError(f"{type(self).__name__} 不支持 get_fundamentals")

    def get_fundamentals_history(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        """单标的财报时序(含 ``announce_date``/``report_date``),供成长/质量回看。"""
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 get_fundamentals_history"
        )

    def get_daily_valuation(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """单标的日频估值时序(index=交易日,columns 含 pe/pb/ps)。

        仅部分 provider(如 akshare 乐咕 / tushare daily_basic)支持,其余抛
        NotImplementedError。
        """
        raise NotImplementedError(f"{type(self).__name__} 不支持 get_daily_valuation")

    def get_daily_dividends(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """单标的每股现金分红事件序列(index=除息日,columns 含 ``dividend``)。

        仅部分 provider(akshare 新浪 / yfinance)支持,其余抛 NotImplementedError。
        """
        raise NotImplementedError(f"{type(self).__name__} 不支持 get_daily_dividends")

    def get_profile(self, symbol: str, market: Market | None = None) -> dict[str, Any]:
        """单标的扩展档案(估值扩展/盈利质量/财务健康/分析师/公司概况等)。

        仅部分 provider(如 yfinance)支持,其余抛 NotImplementedError。
        """
        raise NotImplementedError(f"{type(self).__name__} 不支持 get_profile")


class ProviderRegistry:
    """按注册顺序尝试多个 provider,首个 ``supports`` 命中者负责拉取。"""

    def __init__(self, providers: Iterable[DataProvider] | None = None) -> None:
        self._providers: list[DataProvider] = list(providers or [])
        # 单飞:同键(provider, symbol, adjust)并发拉取串行化,后者命中前者写入的缓存。
        self._fetch_locks: dict[tuple[str, str, str], threading.Lock] = {}
        self._fetch_locks_guard = threading.Lock()

    def _fetch_lock(self, key: tuple[str, str, str]) -> threading.Lock:
        with self._fetch_locks_guard:
            return self._fetch_locks.setdefault(key, threading.Lock())

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
        key = (provider.name, symbol, adjust.value)
        # 单飞:同键并发拉取串行,后者在锁后 provider.get_ohlcv 内命中缓存,不打重复网络。
        with self._fetch_lock(key):
            return provider.get_ohlcv(symbol, start, end, adjust)

"""djinn.data — 数据层:提供器、复权、缓存、日历、基准。"""

from __future__ import annotations

from djinn.data.adjust import apply_adjust, ensure_adjust_columns
from djinn.data.benchmark import (
    BENCHMARK_ALIASES,
    BENCHMARK_SYMBOLS,
    load_benchmark,
    resolve_benchmark,
)
from djinn.data.cache import DataCache, env_cache_dir
from djinn.data.calendar import align_to_calendar, is_trading_day, trading_days
from djinn.data.market_data import MarketData
from djinn.data.provider import DataProvider, ProviderRegistry
from djinn.data.providers import (
    AkShareProvider,
    CSVProvider,
    TushareProvider,
    YahooProvider,
)
from djinn.data.schema import (
    Adjust,
    Bar,
    Market,
    detect_market,
)

__all__ = [
    "BENCHMARK_ALIASES",
    "BENCHMARK_SYMBOLS",
    "Adjust",
    "AkShareProvider",
    "Bar",
    "CSVProvider",
    "DataCache",
    "DataProvider",
    "Market",
    "MarketData",
    "ProviderRegistry",
    "TushareProvider",
    "YahooProvider",
    "align_to_calendar",
    "apply_adjust",
    "detect_market",
    "ensure_adjust_columns",
    "env_cache_dir",
    "is_trading_day",
    "load_benchmark",
    "resolve_benchmark",
    "trading_days",
]


def default_registry(
    csv_dir: str | None = None,
    cache: DataCache | None = None,
    *,
    enable_akshare: bool = True,
    enable_tushare: bool = True,
    enable_yahoo: bool = True,
) -> ProviderRegistry:
    """构建默认 provider 注册表。

    顺序:CSV(本地优先)→ AkShare(A/港股)→ Tushare(A 股)→ Yahoo(美股)。
    缺少可选依赖的 provider 会跳过(supports 返回 False,不会报错)。
    """
    shared_cache = cache or DataCache()
    providers: list[DataProvider] = []
    if csv_dir:
        providers.append(CSVProvider(csv_dir))
    if enable_akshare:
        from djinn.data.providers.akshare import _has_akshare

        if _has_akshare():
            providers.append(AkShareProvider(cache=shared_cache))
    if enable_tushare:
        from djinn.data.providers.tushare import _has_tushare

        if _has_tushare():
            providers.append(TushareProvider(cache=shared_cache))
    if enable_yahoo:
        providers.append(YahooProvider(cache=shared_cache))
    return ProviderRegistry(providers)

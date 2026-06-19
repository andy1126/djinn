"""基准数据加载:内置各市场基准代码映射。"""

from __future__ import annotations

from datetime import date

from djinn.data.market_data import MarketData
from djinn.data.provider import ProviderRegistry
from djinn.data.schema import Adjust, Market
from djinn.utils.exceptions import DataError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

# 各市场默认基准代码。
BENCHMARK_SYMBOLS: dict[Market, str] = {
    Market.CN: "000300.SH",  # 沪深 300
    Market.HK: "^HSI",  # 恒生指数
    Market.US: "^GSPC",  # 标普 500
}

# 友好名 -> (symbol, market)
BENCHMARK_ALIASES: dict[str, tuple[str, Market]] = {
    "csi300": ("000300.SH", Market.CN),
    "沪深300": ("000300.SH", Market.CN),
    "hsi": ("^HSI", Market.HK),
    "恒生指数": ("^HSI", Market.HK),
    "sp500": ("^GSPC", Market.US),
    "标普500": ("^GSPC", Market.US),
    "spx": ("^GSPC", Market.US),
}


def resolve_benchmark(name: str, market: Market | None = None) -> tuple[str, Market]:
    """将基准别名/代码解析为 (symbol, market)。"""
    key = name.strip().lower()
    if key in BENCHMARK_ALIASES:
        return BENCHMARK_ALIASES[key]
    if market is None:
        raise DataError(f"无法解析基准 {name!r},且未提供 market")
    return name, market


def load_benchmark(
    registry: ProviderRegistry,
    name: str,
    start: date,
    end: date,
    market: Market | None = None,
    adjust: Adjust = Adjust.BACKWARD,
) -> MarketData:
    """通过 registry 加载基准行情。"""
    symbol, mkt = resolve_benchmark(name, market)
    _log.info("加载基准 %s (%s) [%s ~ %s]", name, symbol, start, end)
    return registry.get_ohlcv(symbol, start, end, adjust, market=mkt)

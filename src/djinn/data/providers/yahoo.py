"""Yahoo Finance 数据提供器(美股 / 部分港股 / 指数)。

通过 ``yfinance`` 拉取日线,统一规范化列名,应用复权与日历对齐。
内置 Parquet + 内存缓存,命中完整区间时直接返回。
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd

from djinn.data.adjust import apply_adjust, ensure_adjust_columns
from djinn.data.cache import DataCache
from djinn.data.calendar import align_to_calendar
from djinn.data.market_data import MarketData
from djinn.data.provider import DataProvider
from djinn.data.schema import (
    COL_ADJ_FACTOR,
    COL_AMOUNT,
    COL_CLOSE,
    COL_DIVIDEND,
    COL_HIGH,
    COL_IS_SUSPENDED,
    COL_LOW,
    COL_OPEN,
    COL_RAW_CLOSE,
    COL_SPLIT_RATIO,
    COL_VOLUME,
    Adjust,
    Market,
    detect_market,
)
from djinn.utils.exceptions import DataError, ProviderError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

# yfinance 列名 → 规范化列名。
_YF_MAP = {
    "Open": COL_OPEN,
    "High": COL_HIGH,
    "Low": COL_LOW,
    "Close": COL_CLOSE,
    "Adj Close": "adj_close",
    "Volume": COL_VOLUME,
}


class YahooProvider(DataProvider):
    """yfinance 数据提供器(美股默认,亦支持 ^GSPC / ^HSI 等指数)。"""

    name = "yahoo"
    market = Market.US

    def __init__(
        self,
        cache: DataCache | None = None,
        rate_limit_sec: float = 0.0,
    ) -> None:
        self.cache = cache or DataCache()
        self.rate_limit_sec = rate_limit_sec
        self._last_request = 0.0

    def supports(self, symbol: str, market: Market | None = None) -> bool:
        # yfinance 能拉美股字母代码与指数(^GSPC / ^HSI);A 股 6 位数字代码不交给它
        if market is Market.CN:
            return False
        detected = detect_market(symbol)
        if detected is Market.CN:
            return False
        return True

    def get_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: Adjust = Adjust.BACKWARD,
    ) -> MarketData:
        cached = self.cache.get(self.name, symbol, adjust)
        if DataCache.covers(cached, start, end):
            assert cached is not None
            df = cached.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        else:
            new = self._fetch(symbol, start, end)
            df = self.cache.merge(self.name, symbol, adjust, new)
            df = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if len(df) == 0:
            raise DataError(f"Yahoo {symbol} 在 [{start}, {end}] 无数据")
        market = detect_market(symbol)
        if market is Market.CN:
            market = Market.US  # 不应发生(supports 已排除)
        df = ensure_adjust_columns(df)
        df = align_to_calendar(df, market, start, end)
        df = apply_adjust(df, adjust)
        df = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return MarketData(symbol=symbol, market=market, df=df, adjust=adjust)

    def _fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise ProviderError("yfinance 未安装") from e
        # 限流
        if self.rate_limit_sec > 0:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.rate_limit_sec:
                time.sleep(self.rate_limit_sec - elapsed)
            self._last_request = time.monotonic()
        _log.info("yfinance 拉取 %s [%s ~ %s]", symbol, start, end)
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(
                start=start.isoformat(), end=end.isoformat(), auto_adjust=False
            )
        except Exception as e:
            raise ProviderError(f"yfinance 拉取 {symbol} 失败: {e}") from e
        if raw is None or len(raw) == 0:
            raise DataError(f"yfinance 返回空: {symbol}")
        return self._normalize(raw)

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns=_YF_MAP).copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        # 用 Adj Close / Close 推导复权因子与 raw_close
        if "adj_close" in df.columns and COL_CLOSE in df.columns:
            df[COL_ADJ_FACTOR] = (df["adj_close"] / df[COL_CLOSE]).replace(0, 1.0)
            df[COL_RAW_CLOSE] = df[COL_CLOSE]  # 原始(未复权)收盘
        else:
            df[COL_ADJ_FACTOR] = 1.0
            df[COL_RAW_CLOSE] = df[COL_CLOSE]
        # yfinance 的 dividends/splits 在 history 中为 0(在 actions 中),此处留默认
        if COL_DIVIDEND not in df.columns:
            df[COL_DIVIDEND] = 0.0
        if COL_SPLIT_RATIO not in df.columns:
            df[COL_SPLIT_RATIO] = 1.0
        if COL_AMOUNT not in df.columns:
            df[COL_AMOUNT] = 0.0
        df[COL_IS_SUSPENDED] = df[COL_VOLUME] == 0
        # drop adj_close(已编码进 adj_factor)
        if "adj_close" in df.columns:
            df = df.drop(columns=["adj_close"])
        # 丢弃 yfinance 多余列
        keep = [
            c
            for c in df.columns
            if c
            in (
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
        ]
        return df[keep]

"""Tushare 数据提供器(A 股,需 token)。

Phase 1 留作高质量补充 provider;依赖与 token 均可选。
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from djinn.data.adjust import apply_adjust, ensure_adjust_columns
from djinn.data.cache import DataCache
from djinn.data.calendar import align_to_calendar
from djinn.data.market_data import MarketData
from djinn.data.provider import DataProvider
from djinn.data.schema import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_HIGH,
    COL_IS_SUSPENDED,
    COL_LOW,
    COL_OPEN,
    COL_VOLUME,
    Adjust,
    Market,
)
from djinn.utils.exceptions import DataError, ProviderError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

_TS_MAP = {
    "trade_date": "date",
    "open": COL_OPEN,
    "high": COL_HIGH,
    "low": COL_LOW,
    "close": COL_CLOSE,
    "vol": COL_VOLUME,
    "amount": COL_AMOUNT,
}


def _has_tushare() -> bool:
    try:
        import tushare  # noqa: F401

        return bool(os.environ.get("TUSHARE_TOKEN"))
    except ImportError:
        return False


def _normalize_ts_code(symbol: str) -> str:
    """``000300`` → ``000300.SH``(tushare 需要后缀)。"""
    code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    if not code.isdigit():
        return symbol
    if code.startswith(("60", "68", "9", "11", "13")):
        return f"{code}.SH"
    return f"{code}.SZ"


class TushareProvider(DataProvider):
    """Tushare 数据提供器(A 股,需 ``TUSHARE_TOKEN``)。"""

    name = "tushare"
    market = Market.CN

    def __init__(
        self, cache: DataCache | None = None, token: str | None = None
    ) -> None:
        self.cache = cache or DataCache()
        self.token = token or os.environ.get("TUSHARE_TOKEN")

    def supports(self, symbol: str, market: Market | None = None) -> bool:
        if not _has_tushare():
            return False
        if market is Market.US:
            return False
        code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        return code.isdigit() and len(code) == 6

    def get_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: Adjust = Adjust.BACKWARD,
    ) -> MarketData:
        if not self.token:
            raise ProviderError("Tushare 缺少 TUSHARE_TOKEN")
        try:
            import tushare as ts
        except ImportError as e:  # pragma: no cover
            raise ProviderError("tushare 未安装(pip install djinn[tushare])") from e
        ts.set_token(self.token)
        pro = ts.pro_api()
        ts_code = _normalize_ts_code(symbol)
        _log.info("tushare 拉取 %s [%s ~ %s]", ts_code, start, end)
        try:
            raw = pro.daily(
                ts_code=ts_code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
        except Exception as e:
            raise ProviderError(f"tushare 拉取 {symbol} 失败: {e}") from e
        if raw is None or len(raw) == 0:
            raise DataError(f"tushare 返回空: {symbol}")
        df = raw.rename(columns=_TS_MAP)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = ensure_adjust_columns(df)
        df[COL_IS_SUSPENDED] = df[COL_VOLUME] == 0
        df = align_to_calendar(df, Market.CN, start, end)
        df = apply_adjust(df, adjust)
        df = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if len(df) == 0:
            raise DataError(f"tushare {symbol} 在 [{start}, {end}] 无数据")
        return MarketData(symbol=symbol, market=Market.CN, df=df, adjust=adjust)

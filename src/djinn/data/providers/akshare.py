"""AkShare 数据提供器(A 股 / 渓股,免费免 key)。

Phase 1 数据层:A 股日线通过 ``akshare.stock_zh_a_hist`` 拉取,含复权与停牌。
依赖为可选(``pip install djinn[akshare]``),缺失时 :meth:`supports` 返回 False。
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
)
from djinn.utils.exceptions import DataError, ProviderError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

# akshare 返回列名 → 规范化。
_AK_MAP = {
    "日期": "date",
    "开盘": COL_OPEN,
    "最高": COL_HIGH,
    "最低": COL_LOW,
    "收盘": COL_CLOSE,
    "成交量": COL_VOLUME,
    "成交额": COL_AMOUNT,
}

_AK_HAS_AKSHARE: bool | None = None


def _has_akshare() -> bool:
    global _AK_HAS_AKSHARE
    if _AK_HAS_AKSHARE is None:
        try:
            import akshare  # noqa: F401

            _AK_HAS_AKSHARE = True
        except ImportError:
            _AK_HAS_AKSHARE = False
    return _AK_HAS_AKSHARE


def _normalize_ak_code(symbol: str) -> str:
    """``000300.SH`` → ``000300``(akshare 用纯代码 + period 参数区分市场)。"""
    return symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")


class AkShareProvider(DataProvider):
    """AkShare 数据提供器(A 股)。"""

    name = "akshare"
    market = Market.CN

    def __init__(
        self, cache: DataCache | None = None, rate_limit_sec: float = 0.5
    ) -> None:
        self.cache = cache or DataCache()
        self.rate_limit_sec = rate_limit_sec
        self._last_request = 0.0

    def supports(self, symbol: str, market: Market | None = None) -> bool:
        if not _has_akshare():
            return False
        if market is Market.US:
            return False
        # 6 位数字代码(含 .SH/.SZ/.BJ)或 5 位港股代码
        code = _normalize_ak_code(symbol)
        if code.isdigit() and len(code) in (6,):
            return True
        return False

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
            new = self._fetch(symbol, start, end, adjust)
            df = self.cache.merge(self.name, symbol, adjust, new)
            df = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if len(df) == 0:
            raise DataError(f"AkShare {symbol} 在 [{start}, {end}] 无数据")
        df = ensure_adjust_columns(df)
        df = align_to_calendar(df, Market.CN, start, end)
        df = apply_adjust(df, adjust)
        df = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return MarketData(symbol=symbol, market=Market.CN, df=df, adjust=adjust)

    def _fetch(
        self, symbol: str, start: date, end: date, adjust: Adjust
    ) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError("akshare 未安装(pip install djinn[akshare])") from e
        if self.rate_limit_sec > 0:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.rate_limit_sec:
                time.sleep(self.rate_limit_sec - elapsed)
            self._last_request = time.monotonic()
        code = _normalize_ak_code(symbol)
        ak_adjust = {Adjust.NONE: "", Adjust.FORWARD: "qfq", Adjust.BACKWARD: "hfq"}[
            adjust
        ]
        _log.info("akshare 拉取 %s [%s ~ %s] adjust=%s", code, start, end, ak_adjust)
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust=ak_adjust,
            )
        except Exception as e:
            raise ProviderError(f"akshare 拉取 {symbol} 失败: {e}") from e
        if raw is None or len(raw) == 0:
            raise DataError(f"akshare 返回空: {symbol}")
        return self._normalize(raw)

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns=_AK_MAP).copy()
        if "date" not in df.columns:
            raise DataError("akshare 返回缺少日期列")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # akshare 已按 adjust 返回复权价;raw_close/adj_factor 信息有限,留默认
        df[COL_RAW_CLOSE] = df.get(COL_CLOSE, df[COL_CLOSE])
        df[COL_ADJ_FACTOR] = 1.0
        df[COL_DIVIDEND] = 0.0
        df[COL_SPLIT_RATIO] = 1.0
        df[COL_IS_SUSPENDED] = df[COL_VOLUME] == 0
        keep = [
            c
            for c in (
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
            if c in df.columns
        ]
        return df[keep]

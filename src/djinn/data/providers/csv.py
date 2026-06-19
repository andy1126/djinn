"""CSV 数据提供器:从本地 CSV 加载行情,用于离线回测与测试。

期望 CSV 列:``date,open,high,low,close,volume``(可选 ``amount,raw_close,
adj_factor,dividend,split_ratio``)。``date`` 列接受 ISO 字符串,索引转为
``DatetimeIndex``。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from djinn.data.adjust import apply_adjust, ensure_adjust_columns
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
from djinn.utils.exceptions import DataError

# CSV 原始列名 → 规范化列名(兼容常见命名)。
_COL_MAP = {
    "date": "date",
    "datetime": "date",
    "time": "date",
    "timestamp": "date",
    "open": COL_OPEN,
    "high": COL_HIGH,
    "low": COL_LOW,
    "close": COL_CLOSE,
    "volume": COL_VOLUME,
    "vol": COL_VOLUME,
    "amount": COL_AMOUNT,
    "turnover": COL_AMOUNT,
    "raw_close": COL_RAW_CLOSE,
    "adj_factor": COL_ADJ_FACTOR,
    "dividend": COL_DIVIDEND,
    "split": COL_SPLIT_RATIO,
    "split_ratio": COL_SPLIT_RATIO,
}

_NUMERIC_COLS = (
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


class CSVProvider(DataProvider):
    """本地 CSV 数据提供器。

    Args:
        base_dir: CSV 文件根目录,文件名规则 ``{symbol}.csv``(symbol 中的
            ``/`` ``^`` ``:`` 替换为 ``_``)。
        default_market: 当代码无法自动推断市场时使用。
    """

    name = "csv"

    def __init__(
        self, base_dir: str | Path, default_market: Market = Market.US
    ) -> None:
        self.base_dir = Path(base_dir)
        self.default_market = default_market
        self.market = default_market

    def _path_for(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("^", "_").replace(":", "_")
        return self.base_dir / f"{safe}.csv"

    def supports(self, symbol: str, market: Market | None = None) -> bool:
        return self._path_for(symbol).exists()

    def get_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: Adjust = Adjust.BACKWARD,
    ) -> MarketData:
        path = self._path_for(symbol)
        if not path.exists():
            raise DataError(f"CSV 文件不存在: {path}")
        df = self._read(path)
        market = detect_market(symbol)
        df = ensure_adjust_columns(df)
        df = align_to_calendar(df, market, start, end)
        df = apply_adjust(df, adjust)
        df = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if len(df) == 0:
            raise DataError(f"CSV {symbol} 在 [{start}, {end}] 区间无数据")
        return MarketData(symbol=symbol, market=market, df=df, adjust=adjust)

    def _read(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        raw.columns = [c.strip().lower() for c in raw.columns]
        rename = {c: _COL_MAP[c] for c in raw.columns if c in _COL_MAP}
        raw = raw.rename(columns=rename)
        if "date" not in raw.columns:
            raise DataError(f"CSV 缺少 date 列: {path}")
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date").sort_index()
        for c in (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME):
            if c not in raw.columns:
                raise DataError(f"CSV 缺少列 {c!r}: {path}")
        if COL_IS_SUSPENDED not in raw.columns:
            raw[COL_IS_SUSPENDED] = raw[COL_VOLUME] == 0
        # 补充可选列(缺失由 ensure_adjust_columns 填默认,此处仅做类型归一)
        raw = ensure_adjust_columns(raw)
        for c in _NUMERIC_COLS:
            if c in raw.columns:
                raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
        # raw_close 缺失(全 0)时回退到 close
        if raw[COL_RAW_CLOSE].eq(0).all():
            raw[COL_RAW_CLOSE] = raw[COL_CLOSE]
        return raw

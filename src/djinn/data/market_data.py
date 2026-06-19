"""MarketData:规范化行情数据容器。

封装单个标的的 OHLCV DataFrame + 元数据,提供复权切片与 Bar 访问。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import pandas as pd

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
    Bar,
    Market,
)


@dataclass
class MarketData:
    """单标的规范化行情数据。

    Attributes:
        symbol: 标的代码(provider 原始口径,如 ``AAPL`` / ``000300.SH``)。
        market: 所属市场。
        df: 行情表,``DatetimeIndex``(naive 交易日),列见 :mod:`djinn.data.schema`。
        adjust: 当前 df 已应用的复权方式。
    """

    symbol: str
    market: Market
    df: pd.DataFrame
    adjust: Adjust = Adjust.BACKWARD

    def __post_init__(self) -> None:
        # 校验索引为 DatetimeIndex
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError(
                f"MarketData.df 索引必须是 DatetimeIndex,实际为 {type(self.df.index).__name__}"
            )
        # 确保核心列存在
        missing = [
            c
            for c in (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME)
            if c not in self.df.columns
        ]
        if missing:
            raise ValueError(f"MarketData 缺少核心列: {missing}")

    # ── 基本访问 ────────────────────────────────────────
    def to_frame(self) -> pd.DataFrame:
        """返回行情 DataFrame(副本)。"""
        return self.df.copy()

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.df.index)

    @property
    def start_date(self) -> date:
        return pd.Timestamp(self.df.index[0]).date()

    @property
    def end_date(self) -> date:
        return pd.Timestamp(self.df.index[-1]).date()

    def __len__(self) -> int:
        return len(self.df)

    # ── 切片 ───────────────────────────────────────────
    def up_to(self, when: date) -> pd.DataFrame:
        """返回 ``<= when`` 的历史切片(防未来函数核心入口)。"""
        ts = pd.Timestamp(when)
        return self.df.loc[:ts]

    def slice(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """返回 [start, end] 闭区间切片。"""
        lo = pd.Timestamp(start) if start else None
        hi = pd.Timestamp(end) if end else None
        return self.df.loc[lo:hi]

    # ── Bar 迭代 ───────────────────────────────────────
    def bars(self) -> Iterator[Bar]:
        """按时间顺序产出 :class:`Bar`。"""
        for ts, row in self.df.iterrows():
            yield self._row_to_bar(pd.Timestamp(ts), row)  # type: ignore[arg-type]

    def bar_at(self, when: date) -> Bar | None:
        """返回 ``when`` 当日的 Bar;无该日则返回 None。"""
        ts = pd.Timestamp(when)
        if ts not in self.df.index:
            return None
        row = self.df.loc[ts]
        return self._row_to_bar(ts, row)  # type: ignore[arg-type]

    def _row_to_bar(self, ts: pd.Timestamp, row: pd.Series) -> Bar:
        def _g(col: str, default: float) -> float:
            v = row.get(col, default)
            if pd.isna(v):
                return default
            return float(v)

        return Bar(
            timestamp=ts.date(),
            symbol=self.symbol,
            market=self.market,
            open=_g(COL_OPEN, 0.0),
            high=_g(COL_HIGH, 0.0),
            low=_g(COL_LOW, 0.0),
            close=_g(COL_CLOSE, 0.0),
            volume=_g(COL_VOLUME, 0.0),
            amount=_g(COL_AMOUNT, 0.0),
            raw_close=_g(COL_RAW_CLOSE, _g(COL_CLOSE, 0.0)),
            adj_factor=_g(COL_ADJ_FACTOR, 1.0),
            dividend=_g(COL_DIVIDEND, 0.0),
            split_ratio=_g(COL_SPLIT_RATIO, 1.0),
            is_suspended=bool(_g(COL_IS_SUSPENDED, 0.0) != 0.0),
        )

    # ── 元信息 ─────────────────────────────────────────
    def info(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "market": self.market.value,
            "adjust": self.adjust.value,
            "rows": len(self.df),
            "start": str(self.start_date),
            "end": str(self.end_date),
        }

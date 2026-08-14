"""因子引擎:universe × 区间 → 因子宽表面板。

流程:拉取 universe 行情 → 组 ``date × symbol`` 宽表(价格 / 量 / 额)→
(可选)经 :class:`~djinn.data.fundamentals.FundamentalsSource` 组 point-in-time
基本面宽表 → 逐因子 :meth:`Factor.compute` → :class:`FactorPanel`。

面板全部为 ``float64`` pandas 宽表;基本面字段经 announce_date asof 对齐到交易日,
杜绝未来函数(见 :func:`_asof_field_panel`)。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from djinn.data.fundamentals import FundamentalsSource
from djinn.data.market_data import MarketData
from djinn.data.provider import ProviderRegistry
from djinn.data.schema import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_PB,
    COL_PE,
    COL_PS,
    COL_VOLUME,
    Adjust,
    Market,
)
from djinn.factor.base import Factor, Panel, PanelDict
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

# 引擎默认组装的基本面字段(值字段;行业单独由 industry_map 提供)。
DEFAULT_FUNDAMENTAL_FIELDS: tuple[str, ...] = (
    "market_cap",
    "float_cap",
    "pe",
    "pb",
    "ps",
    "roe",
    "gross_margin",
    "net_profit",
    "revenue",
    "ocf",
    "total_assets",
    "revenue_yoy",
    "profit_yoy",
)

_HISTORY_LOOKBACK_DAYS = 400

# 估值类字段:优先走 provider 的日频估值序列(``get_daily_valuation``),消除
# "用今日快照给历史打分"的前视;无日频估值时退化为快照口径(见 ``_asof_field_panel``)。
VALUATION_FIELDS: tuple[str, ...] = (COL_PE, COL_PB, COL_PS)


@dataclass
class FactorPanel:
    """因子面板容器:``{因子名 → DataFrame(date × symbol)}``。"""

    data: dict[str, Panel] = field(default_factory=dict)

    @property
    def factor_names(self) -> list[str]:
        return list(self.data)

    @property
    def dates(self) -> pd.DatetimeIndex:
        if not self.data:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(next(iter(self.data.values())).index)

    @property
    def symbols(self) -> list[str]:
        if not self.data:
            return []
        return list(next(iter(self.data.values())).columns)

    def factor(self, name: str) -> Panel:
        if name not in self.data:
            raise KeyError(f"无因子 {name!r},可用: {self.factor_names}")
        return self.data[name]

    def cross_section(self, when: date) -> pd.DataFrame:
        """``when`` 当日截面(index=symbol、columns=因子名)。"""
        ts = pd.Timestamp(when)
        out: dict[str, pd.Series] = {}
        for name, df in self.data.items():
            if ts in df.index:
                row = df.loc[ts]
                out[name] = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            else:
                out[name] = pd.Series(dtype=float)
        return pd.DataFrame(out)

    def __len__(self) -> int:
        return len(self.data)


class FactorEngine:
    """因子计算引擎。"""

    def compute(
        self,
        factors: list[Factor],
        universe: list[str],
        start: date,
        end: date,
        registry: ProviderRegistry,
        *,
        market: Market | None = None,
        adjust: Adjust = Adjust.BACKWARD,
        fundamentals_source: FundamentalsSource | None = None,
        fundamental_fields: tuple[str, ...] = DEFAULT_FUNDAMENTAL_FIELDS,
    ) -> FactorPanel:
        """对 universe 在 [start, end] 计算一组因子,返回 :class:`FactorPanel`。"""
        prices, ohlcv = self._ohlcv_panels(
            universe, start, end, registry, market, adjust
        )
        fundamentals: PanelDict = {}
        if fundamentals_source is not None:
            fundamentals = self._fundamental_panels(
                fundamental_fields,
                universe,
                pd.DatetimeIndex(prices.index),
                start,
                end,
                fundamentals_source,
                market,
            )
        data: dict[str, Panel] = {}
        for f in factors:
            _log.info("计算因子 %s(%d 标的)", f.name, len(universe))
            f.validate_inputs(fundamentals, ohlcv)
            data[f.name] = f.compute(prices, ohlcv, fundamentals)
        return FactorPanel(data=data)

    # ── 行情面板 ────────────────────────────────────────
    def _ohlcv_panels(
        self,
        universe: list[str],
        start: date,
        end: date,
        registry: ProviderRegistry,
        market: Market | None,
        adjust: Adjust,
    ) -> tuple[Panel, PanelDict]:
        closes: dict[str, pd.Series] = {}
        fields: dict[str, dict[str, pd.Series]] = {
            c: {} for c in (COL_OPEN, COL_HIGH, COL_LOW, COL_VOLUME, COL_AMOUNT)
        }

        def _fetch_one(sym: str) -> tuple[str, MarketData | None, str | None]:
            try:
                md = registry.get_ohlcv(sym, start, end, adjust, market=market)
                return sym, md, None
            except Exception as e:
                return sym, None, str(e)

        # D7:IO 密集拉取用线程池并发(E1 已保证 DataCache 线程安全)
        workers = int(os.environ.get("DJINN_FETCH_WORKERS", "8"))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_fetch_one, universe))
        for sym, md, err in results:
            if md is None:
                _log.warning("拉取 %s 失败,跳过: %s", sym, err)
                continue
            df = md.df
            closes[sym] = df[COL_CLOSE]
            for c in fields:
                if c in df.columns:
                    fields[c][sym] = df[c]
        if not closes:
            raise ValueError("universe 无可用行情,无法计算因子")
        close_panel = pd.DataFrame(closes).sort_index()
        ohlcv: PanelDict = {
            c: pd.DataFrame(s).reindex(close_panel.index) for c, s in fields.items()
        }
        return close_panel, ohlcv

    # ── 基本面面板(point-in-time)────────────────────────
    def _fundamental_panels(
        self,
        fields: tuple[str, ...],
        universe: list[str],
        trading_index: pd.DatetimeIndex,
        start: date,
        end: date,
        source: FundamentalsSource,
        market: Market | None,
    ) -> PanelDict:
        return {
            f: self._asof_field_panel(
                f, universe, trading_index, start, end, source, market
            )
            for f in fields
        }

    def _asof_field_panel(
        self,
        field: str,
        universe: list[str],
        trading_index: pd.DatetimeIndex,
        start: date,
        end: date,
        source: FundamentalsSource,
        market: Market | None,
    ) -> Panel:
        """单基本面字段的 point-in-time 宽表。

        优先用财报时序按 ``announce_date`` asof 到交易日(真正 PIT);
        无该字段时序(如估值快照)退化为区间末日常数(明确标注的近似)。
        """
        cols: dict[str, pd.Series] = {}
        hist_start = start - timedelta(days=_HISTORY_LOOKBACK_DAYS)
        for sym in universe:
            series: pd.Series | None = None
            # 估值类字段:优先日频估值序列(真正 point-in-time,无前视)
            if field in VALUATION_FIELDS:
                try:
                    daily = source.get_daily_valuation(sym, start, end, market)
                    if daily is not None and len(daily) and field in daily.columns:
                        series = (
                            pd.to_numeric(daily[field], errors="coerce")
                            .reindex(trading_index)
                            .ffill()
                        )
                except Exception as e:
                    _log.debug("%s 字段 %s 日频估值不可用: %s", sym, field, e)
            if series is None:
                try:
                    hist = source.get_history(sym, hist_start, end, market)
                    if field in hist.columns and "announce_date" in hist.columns:
                        series = _asof_series(
                            hist[field], hist["announce_date"], trading_index
                        )
                except Exception as e:
                    _log.debug("%s 字段 %s 时序不可用: %s", sym, field, e)
            if series is None:
                # 退化:用 when=end 的快照常数填充(估值类近似,非严格 PIT)
                try:
                    snap = source.get_snapshot([sym], end, market)
                    val = float(pd.to_numeric(snap[field], errors="coerce").iloc[0])
                except Exception:
                    val = float("nan")
                series = pd.Series(val, index=trading_index)
            cols[sym] = series.reindex(trading_index)
        return pd.DataFrame(cols)


def _asof_series(
    values: pd.Series, announce: pd.Series, trading_index: pd.DatetimeIndex
) -> pd.Series:
    """把"按公告日生效"的字段 asof 对齐到交易日(merge_asof,backward)。"""
    right = pd.DataFrame(
        {
            "announce_date": pd.to_datetime(announce).to_numpy(),
            "value": pd.to_numeric(values, errors="coerce").to_numpy(),
        }
    ).dropna(subset=["announce_date"])
    right = right.sort_values("announce_date")
    if len(right) == 0:
        return pd.Series(float("nan"), index=trading_index)
    left = pd.DataFrame({"date": trading_index})
    merged = pd.merge_asof(
        left, right, left_on="date", right_on="announce_date", direction="backward"
    )
    return pd.Series(merged["value"].to_numpy(), index=trading_index)

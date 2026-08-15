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
from typing import Any

import pandas as pd

from djinn.data.fundamentals import FundamentalsSource
from djinn.data.market_data import MarketData
from djinn.data.provider import ProviderRegistry
from djinn.data.schema import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_DIVIDEND,
    COL_FLOAT_CAP,
    COL_HIGH,
    COL_LOW,
    COL_MARKET_CAP,
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
    COL_DIVIDEND,
)

_HISTORY_LOOKBACK_DAYS = 400

# 估值类字段:优先走 provider 的日频估值序列(``get_daily_valuation``),消除
# "用今日快照给历史打分"的前视;无日频估值时退化为快照口径(见 ``_asof_field_panel``)。
VALUATION_FIELDS: tuple[str, ...] = (COL_PE, COL_PB, COL_PS)

# 市值类字段:无历史时序时用「收盘价 × 股本」合成(C2 市值 PIT 近似)。
_CAP_APPROX_FIELDS: frozenset[str] = frozenset({COL_MARKET_CAP, COL_FLOAT_CAP})


@dataclass
class FactorPanel:
    """因子面板容器:``{因子名 → DataFrame(date × symbol)}``。"""

    data: dict[str, Panel] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

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

    def __init__(self) -> None:
        self._warned_fields: set[str] = set()  # C3:退化告警按字段去重
        self._approx_fields: set[str] = set()  # C2:市值类字段走"收盘×股本"近似的字段

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
                close_panel=prices,
            )
        # C6:为声明了 benchmark 的因子预拉基准日收益(经 __benchmark__ 键注入)
        for f in factors:
            bench = getattr(f, "benchmark", None)
            if bench:
                bench_rets = self._benchmark_returns(
                    bench, start, end, registry, market, adjust
                )
                if len(bench_rets):
                    # PanelDict 值类型为 DataFrame,基准为 Series → 经宽松 dict 注入
                    ohlcv_any: dict[str, Any] = ohlcv
                    ohlcv_any["__benchmark__"] = bench_rets
                break

        data: dict[str, Panel] = {}
        for f in factors:
            _log.info("计算因子 %s(%d 标的)", f.name, len(universe))
            f.validate_inputs(fundamentals, ohlcv)
            data[f.name] = f.compute(prices, ohlcv, fundamentals)
        panel = FactorPanel(data=data)
        if self._approx_fields:
            panel.meta["market_cap_approx"] = sorted(self._approx_fields)
        return panel

    def caveats(self) -> list[str]:
        """C3:数据口径告警(供报告层标注非 point-in-time 字段)。

        - ``_warned_fields``:无历史/日频估值、退化为快照常数(pe/pb/ps 等);
        - ``_approx_fields``:市值类字段由「收盘价×股本」近似(股本未做 PIT)。
        """
        out: list[str] = []
        if self._warned_fields:
            out.append(
                "以下字段为快照口径(非 point-in-time,IC 可能高估): "
                + ", ".join(sorted(self._warned_fields))
            )
        if self._approx_fields:
            out.append(
                "市值/流通市值由收盘价 x 股本近似(股本未做 PIT): "
                + ", ".join(sorted(self._approx_fields))
            )
        return out

    def _benchmark_returns(
        self,
        benchmark: str,
        start: date,
        end: date,
        registry: ProviderRegistry,
        market: Market | None,
        adjust: Adjust,
    ) -> pd.Series:
        """拉取基准日收益序列;失败返回空 Series(因子内部退化为等权代理)。"""
        try:
            md = registry.get_ohlcv(benchmark, start, end, adjust, market=market)
            return md.df[COL_CLOSE].pct_change()
        except Exception as e:
            _log.warning("基准 %s 拉取失败,beta 退化为等权代理: %s", benchmark, e)
            return pd.Series(dtype=float)

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
        *,
        close_panel: Panel | None = None,
    ) -> PanelDict:
        # C15:按标的各取一次财报时序 + 日频估值 + 分红事件,再按字段 asof 对齐;
        # 消除"按字段循环对每个标的重复拉取"的 ×N 冗余(财报时序被拉 9 遍)。
        hist_start = start - timedelta(days=_HISTORY_LOOKBACK_DAYS)
        histories: dict[str, pd.DataFrame] = {}
        valuations: dict[str, pd.DataFrame] = {}
        dividends: dict[str, pd.DataFrame] = {}
        for sym in universe:
            histories[sym] = _safe_frame(
                source.get_history, sym, hist_start, end, market
            )
            valuations[sym] = _safe_frame(
                source.get_daily_valuation, sym, start, end, market
            )
            dividends[sym] = _safe_frame(
                source.get_daily_dividends, sym, hist_start, end, market
            )
        return {
            f: self._asof_field_panel(
                f,
                universe,
                trading_index,
                end,
                source,
                market,
                histories,
                valuations,
                dividends,
                close_panel=close_panel,
            )
            for f in fields
        }

    def _asof_field_panel(
        self,
        field: str,
        universe: list[str],
        trading_index: pd.DatetimeIndex,
        end: date,
        source: FundamentalsSource,
        market: Market | None,
        histories: dict[str, pd.DataFrame],
        valuations: dict[str, pd.DataFrame],
        dividends: dict[str, pd.DataFrame],
        *,
        close_panel: Panel | None = None,
    ) -> Panel:
        """单基本面字段的 point-in-time 宽表(输入为已预取的 per-symbol 时序)。

        优先用财报时序按 ``announce_date`` asof 到交易日(真正 PIT);
        无该字段时序(如估值快照)退化为区间末日常数(明确标注的近似)。
        """
        cols: dict[str, pd.Series] = {}
        for sym in universe:
            series: pd.Series | None = None
            # 估值类字段:优先日频估值序列(真正 point-in-time,无前视)
            daily = valuations.get(sym)
            if (
                field in VALUATION_FIELDS
                and daily is not None
                and len(daily)
                and field in daily.columns
            ):
                series = (
                    pd.to_numeric(daily[field], errors="coerce")
                    .reindex(trading_index)
                    .ffill()
                )
            # 分红字段:事件型序列(除息日 → 每股现金),reindex 到交易日,缺日填 0
            div = dividends.get(sym)
            if (
                field == COL_DIVIDEND
                and div is not None
                and len(div)
                and field in div.columns
            ):
                cash = pd.to_numeric(div[field], errors="coerce")
                if cash.index.has_duplicates:
                    cash = cash.groupby(level=0).sum()
                series = cash.reindex(trading_index).fillna(0.0)
            if series is None:
                hist = histories.get(sym)
                if (
                    hist is not None
                    and field in hist.columns
                    and "announce_date" in hist.columns
                ):
                    series = _asof_series(
                        hist[field], hist["announce_date"], trading_index
                    )
            if (
                series is None
                and field in _CAP_APPROX_FIELDS
                and close_panel is not None
            ):
                # C2:市值类字段无历史时序时,用「收盘价 × 股本」合成——股本取最新快照
                # 反推(market_cap/close),价格逐日变动 → 消除"今日市值给历史打分"的前视;
                # 股本未做 PIT(漂移不频繁,接受并在 meta 标注)。
                series = self._cap_approx_series(
                    field, sym, trading_index, end, source, market, close_panel
                )
            if series is None:
                # 退化:用 when=end 的快照常数填充(估值类近似,非严格 PIT)
                if field not in self._warned_fields:
                    _log.warning(
                        "字段 %s 无历史/日频估值,使用 %s 快照常数填充全历史(非 PIT)",
                        field,
                        end,
                    )
                    self._warned_fields.add(field)
                try:
                    snap = source.get_snapshot([sym], end, market)
                    val = float(pd.to_numeric(snap[field], errors="coerce").iloc[0])
                except Exception:
                    val = float("nan")
                series = pd.Series(val, index=trading_index)
            cols[sym] = series.reindex(trading_index)
        return pd.DataFrame(cols)

    def _cap_approx_series(
        self,
        field: str,
        sym: str,
        trading_index: pd.DatetimeIndex,
        end: date,
        source: FundamentalsSource,
        market: Market | None,
        close_panel: Panel,
    ) -> pd.Series | None:
        """市值近似:收盘价逐日 × 股本(最新快照 market_cap/close 反推)。

        成功时记录到 ``_approx_fields`` 供 ``FactorPanel.meta`` 标注;失败返回
        None(调用方继续走快照常数退化)。
        """
        close_series = close_panel.get(sym)
        if close_series is None or close_series.dropna().empty:
            return None
        try:
            snap = source.get_snapshot([sym], end, market)
            price_now = float(pd.to_numeric(snap["close"], errors="coerce").iloc[0])
            cap_now = float(pd.to_numeric(snap[field], errors="coerce").iloc[0])
        except Exception:
            return None
        if not (price_now > 0) or not (cap_now == cap_now and cap_now > 0):
            return None
        shares = cap_now / price_now
        self._approx_fields.add(field)
        return close_series.reindex(trading_index).astype(float) * shares


def _safe_frame(fn: Any, *args: Any) -> pd.DataFrame:
    """调用 ``fn(*args)`` 返回 DataFrame;异常 / 非 DataFrame → 空帧。"""
    try:
        out = fn(*args)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _asof_series(
    values: pd.Series, announce: pd.Series, trading_index: pd.DatetimeIndex
) -> pd.Series:
    """把"按公告日生效"的字段 asof 对齐到交易日(merge_asof,backward)。

    两侧键统一为 ``datetime64[ns]``:yahoo 财报的 announce_date 可能是 us 精度,
    与行情交易日(``ns``)不一致时 ``merge_asof`` 会抛 MergeError。
    """
    idx_ns = pd.DatetimeIndex(trading_index).astype("datetime64[ns]")
    ann_ns = pd.to_datetime(announce).dt.tz_localize(None).astype("datetime64[ns]")
    right = pd.DataFrame(
        {
            "announce_date": ann_ns.to_numpy(),
            "value": pd.to_numeric(values, errors="coerce").to_numpy(),
        }
    ).dropna(subset=["announce_date"])
    right = right.sort_values("announce_date")
    if len(right) == 0:
        return pd.Series(float("nan"), index=idx_ns)
    left = pd.DataFrame({"date": idx_ns})
    merged = pd.merge_asof(
        left, right, left_on="date", right_on="announce_date", direction="backward"
    )
    return pd.Series(merged["value"].to_numpy(), index=idx_ns)

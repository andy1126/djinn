"""Yahoo Finance 数据提供器(美股 / 部分港股 / 指数)。

通过 ``yfinance`` 拉取日线,统一规范化列名,应用复权与日历对齐。
内置 Parquet + 内存缓存,命中完整区间时直接返回。
指数成分(HSI / SP500 / NASDAQ100 / DOWJONES)来自 yfiua.github.io 免费 CSV,见 :meth:`get_index_components`。

yfinance 易发网络抖动 / 偶发空返回(尤其短时间多次请求后),
故 :meth:`_fetch` 内置指数退避重试(见 CLAUDE.md "yfinance 易发网络抖动")。
"""

from __future__ import annotations

import math
import random
import threading
import time
from datetime import date
from typing import Any, cast

import pandas as pd

from djinn.data.adjust import apply_adjust, ensure_adjust_columns
from djinn.data.cache import DataCache
from djinn.data.calendar import align_to_calendar
from djinn.data.market_data import MarketData
from djinn.data.provider import DataProvider
from djinn.data.schema import (
    COL_ADJ_FACTOR,
    COL_AMOUNT,
    COL_ANNOUNCE_DATE,
    COL_CLOSE,
    COL_DIVIDEND,
    COL_FLOAT_CAP,
    COL_GROSS_MARGIN,
    COL_HIGH,
    COL_IS_SUSPENDED,
    COL_LOW,
    COL_MARKET_CAP,
    COL_NET_PROFIT,
    COL_OCF,
    COL_OPEN,
    COL_PB,
    COL_PE,
    COL_PROFIT_YOY,
    COL_PS,
    COL_RAW_CLOSE,
    COL_REPORT_DATE,
    COL_REVENUE,
    COL_REVENUE_YOY,
    COL_ROE,
    COL_SPLIT_RATIO,
    COL_TOTAL_ASSETS,
    COL_VOLUME,
    Adjust,
    Market,
    detect_market,
)
from djinn.data.universe import INDEX_COMPONENTS_TTL_DAYS, UNIVERSE_INDEX_MAP
from djinn.utils.exceptions import DataError, ProviderError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

# D6:基本面 history 缓存 30 天过期(财报时序静态,过期重拉足够)
_FIN_HIST_TTL_DAYS = 30

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

    # D11:info 缓存 TTL(秒)。同一 provider 实例(registry 单例)内把详情端点多次
    # ``Ticker.info`` 合并为一次网络拉取;registry 单例保证进程内实际全局共享。
    _INFO_TTL_SEC = 300.0

    def __init__(
        self,
        cache: DataCache | None = None,
        rate_limit_sec: float = 0.3,
        max_retries: int = 3,
    ) -> None:
        self.cache = cache or DataCache()
        self.rate_limit_sec = rate_limit_sec
        self.max_retries = max(1, max_retries)
        self._last_request = 0.0
        self._rate_lock = threading.Lock()  # 限速临界区(跨线程串行)
        # D11:info TTL 缓存(key=symbol,value=(monotonic_ts, info_dict))
        self._info_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._info_cache_lock = threading.Lock()

    def _get_info_cached(self, symbol: str) -> dict[str, Any]:
        """``Ticker(symbol).info`` 的进程内 TTL 缓存;异常向上抛(调用方决定降级)。"""
        now = time.monotonic()
        with self._info_cache_lock:
            hit = self._info_cache.get(symbol)
            if hit is not None and now - hit[0] < self._INFO_TTL_SEC:
                return hit[1]
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise ProviderError("yfinance 未安装") from e
        self._throttle()
        info = yf.Ticker(symbol).info or {}
        with self._info_cache_lock:
            self._info_cache[symbol] = (now, info)
        return info

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
        ysym = self._yf_symbol(symbol)
        cached = self.cache.get(self.name, ysym, adjust)
        if DataCache.covers_soft(cached, start, end):
            assert cached is not None
            df = cached.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        else:
            new = self._fetch(ysym, start, end)
            df = self.cache.merge(self.name, ysym, adjust, new)
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

    def _yf_symbol(self, symbol: str) -> str:
        """djinn 符号 → yfinance 符号:美股带点代码(如 ``BRK.B``)需改连字符。

        yfinance 对 ``BRK.B`` / ``BF.B`` 不识别(抛 delisted),连字符形式
        ``BRK-B`` 才有效;``.HK`` / A 股后缀不在此列,原样返回。
        """
        if symbol.count(".") == 1 and not symbol.upper().endswith(
            (".HK", ".SH", ".SZ", ".BJ")
        ):
            return symbol.replace(".", "-")
        return symbol

    def _fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise ProviderError("yfinance 未安装") from e
        # 限流
        self._throttle()
        _log.info("yfinance 拉取 %s [%s ~ %s]", symbol, start, end)
        # yfinance 易发网络抖动 / 偶发空返回;指数退避重试。
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                raw = ticker.history(
                    start=start.isoformat(), end=end.isoformat(), auto_adjust=False
                )
            except Exception as e:  # yfinance 抛各种网络/解析错误
                last_exc = e
                _log.warning(
                    "yfinance 拉取 %s 第 %d/%d 次异常: %s",
                    symbol,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                self._sleep_backoff(attempt)
                continue
            if raw is None or len(raw) == 0:
                # 空返回在 yfinance 抖动里很常见;也可能是真无数据。
                # 重试覆盖抖动,真无数据时重试也是空,代价小。
                _log.warning(
                    "yfinance 拉取 %s 第 %d/%d 次返回空",
                    symbol,
                    attempt + 1,
                    self.max_retries,
                )
                last_exc = DataError(f"yfinance 返回空: {symbol}")
                self._sleep_backoff(attempt)
                continue
            return self._normalize(raw)
        # 重试用尽:若是网络异常,抛 ProviderError;若是空返回,抛 DataError。
        if isinstance(last_exc, DataError):
            raise last_exc
        raise ProviderError(
            f"yfinance 拉取 {symbol} 失败(重试 {self.max_retries} 次): {last_exc}"
        ) from last_exc

    def _sleep_backoff(self, attempt: int) -> None:
        """指数退避 + 抖动:``base * (0.5 + uniform(0,1))``,打散并发重试节奏(E14)。"""
        delay = 0.5 * (3**attempt) * (0.5 + random.random())
        time.sleep(delay)

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

    # ── 基本面(美/港兜底,Phase 0 扩展)─────────────────────
    def get_fundamentals(self, symbols: list[str], when: date) -> pd.DataFrame:
        """截面基本面快照(``Ticker.info``,实时口径,美/港兜底)。

        覆盖全部 11 个 FUNDAMENTAL_VALUE_COLUMNS(估值 + 财务 + 成长);
        ``when`` 仅作记录(info 为最新值,非历史 PIT)。网络抖动时按单标的降级,
        不整体失败。
        """
        rows: dict[str, dict[str, float]] = {}
        for sym in symbols:
            cache_symbol = f"info_{sym}"
            cached = self.cache.get_fundamentals(self.name, cache_symbol)
            # 旧缓存只有估值 4 列(缺 COL_REVENUE),视为 miss 重拉自愈(同 index_cons 缺 name 列)
            if cached is not None and len(cached) and COL_REVENUE in cached.columns:
                rows[sym] = {c: float(cached.iloc[0][c]) for c in cached.columns}
                continue
            try:
                info = self._get_info_cached(sym)
            except Exception as e:
                _log.warning("yfinance %s info 拉取失败,跳过: %s", sym, e)
                continue
            row = {
                COL_MARKET_CAP: _fnum(info.get("marketCap")),
                COL_FLOAT_CAP: _float_cap(info),
                COL_PE: _fnum(info.get("trailingPE")),
                COL_PB: _fnum(info.get("priceToBook")),
                COL_PS: _fnum(info.get("priceToSalesTrailing12Months")),
                COL_ROE: _pct(info.get("returnOnEquity")),
                COL_GROSS_MARGIN: _pct(info.get("grossMargins")),
                COL_REVENUE: _fnum(info.get("totalRevenue")),
                COL_NET_PROFIT: _fnum(info.get("netIncomeToCommon")),
                COL_REVENUE_YOY: _pct(info.get("revenueGrowth")),
                COL_PROFIT_YOY: _pct(info.get("earningsGrowth")),
            }
            self.cache.put_fundamentals(self.name, cache_symbol, pd.DataFrame([row]))
            rows[sym] = row
        if not rows:
            raise DataError(f"yfinance 基本面快照为空: {symbols}")
        return pd.DataFrame.from_dict(rows, orient="index")

    def get_fundamentals_history(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        """单标的财务时序(income_stmt + balance_sheet + cashflow 年度报表)。

        yfinance 无公告日,报表列即财报期末;此处以 ``report_date + 45 天``
        近似公告日(同 A 股 akshare 口径,标注为近似)。Yahoo 硬性上限 ~4 年度,
        同比/ROE 序列较稀疏但非空。
        """
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise ProviderError("yfinance 未安装") from e
        cache_symbol = f"finhist_{symbol}"
        cached = self.cache.get_fundamentals(
            self.name, cache_symbol, max_age_days=_FIN_HIST_TTL_DAYS
        )
        if cached is not None and len(cached):
            return cached
        self._throttle()
        try:
            t = yf.Ticker(symbol)
            ist = t.income_stmt
            bs = t.balance_sheet
            cf = t.cashflow
        except Exception as e:
            raise ProviderError(f"yfinance 拉取 {symbol} 财务报表失败: {e}") from e
        if ist is None or len(ist) == 0:
            raise DataError(f"yfinance {symbol} 利润表为空")
        df = self._normalize_fin_history(ist, bs, cf)
        if df is None or len(df) == 0:
            raise DataError(f"yfinance {symbol} 财务时序为空")
        self.cache.put_fundamentals(self.name, cache_symbol, df)
        return df

    @staticmethod
    def _normalize_fin_history(
        ist: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """income_stmt / balance_sheet / cashflow → 规范化财务时序。

        经营现金流(OCF)取自 cashflow 表的 ``Operating Cash Flow``;缺失时该列
        置空(不 fail,由因子层 ``required_fundamentals`` 兜底)。
        """

        def row_at(frm: pd.DataFrame, label: str) -> pd.Series:
            if frm is None or label not in frm.index:
                return pd.Series(dtype=float)
            return pd.to_numeric(cast(pd.Series, frm.loc[label]), errors="coerce")

        revenue = row_at(ist, "Total Revenue")
        net_income = row_at(ist, "Net Income")
        gross_profit = row_at(ist, "Gross Profit")
        equity = row_at(bs, "Stockholders Equity")
        if len(revenue) == 0:
            return pd.DataFrame()

        periods = pd.DatetimeIndex(revenue.index).astype("datetime64[ns]")
        out = pd.DataFrame(index=pd.DatetimeIndex(periods, name="date"))
        out[COL_ROE] = _safe_div(net_income, equity).reindex(periods) * 100.0
        out[COL_GROSS_MARGIN] = (
            _safe_div(gross_profit, revenue).reindex(periods) * 100.0
        )
        out[COL_REVENUE] = revenue.reindex(periods)
        out[COL_NET_PROFIT] = net_income.reindex(periods)
        total_assets = row_at(bs, "Total Assets")
        out[COL_TOTAL_ASSETS] = total_assets.reindex(periods)
        ocf = (
            row_at(cf, "Operating Cash Flow")
            if cf is not None
            else pd.Series(dtype=float)
        )
        out[COL_OCF] = ocf.reindex(periods)
        # 同比:与上一会计年度比较(升序后 shift 1)
        rev_asc = revenue.reindex(periods).sort_index()
        ni_asc = net_income.reindex(periods).sort_index()
        rev_yoy = (_safe_div(rev_asc, rev_asc.shift(1)) - 1.0) * 100.0
        ni_yoy = (_safe_div(ni_asc, ni_asc.shift(1)) - 1.0) * 100.0
        out[COL_REVENUE_YOY] = rev_yoy.reindex(periods)
        out[COL_PROFIT_YOY] = ni_yoy.reindex(periods)
        out[COL_REPORT_DATE] = pd.DatetimeIndex(periods)
        out[COL_ANNOUNCE_DATE] = pd.DatetimeIndex(periods) + pd.Timedelta(days=45)
        return out.sort_index()

    def get_daily_dividends(
        self, symbol: str, start: date, end: date, market: Market | None = None
    ) -> pd.DataFrame:
        """单标的每股现金分红事件序列(index=除息日,``dividend`` 列)。

        用 ``yf.Ticker(symbol).dividends``(美股 / 港股 ``.HK`` 均可);分红为静态
        历史,整帧落盘缓存后按区间切片。空历史返回空帧(由调用方退化)。
        """
        cache_symbol = f"dividends_{symbol}"
        cached = self.cache.get_fundamentals(self.name, cache_symbol)
        if cached is not None and len(cached):
            cached.index = pd.to_datetime(cached.index)
            return cached.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise ProviderError("yfinance 未安装") from e
        self._throttle()
        try:
            div = yf.Ticker(symbol).dividends
        except Exception as e:
            raise ProviderError(f"yfinance 拉取 {symbol} 分红失败: {e}") from e
        if div is None or len(div) == 0:
            return pd.DataFrame()
        cash = pd.to_numeric(div, errors="coerce")
        cash.index = pd.to_datetime(cash.index).tz_localize(None)
        out = pd.DataFrame({COL_DIVIDEND: cash}).sort_index()
        self.cache.put_fundamentals(self.name, cache_symbol, out)
        return out.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    def _throttle(self) -> None:
        if self.rate_limit_sec <= 0:
            return
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.rate_limit_sec:
                time.sleep(self.rate_limit_sec - elapsed)
            self._last_request = time.monotonic()

    # ── 指数成分(HSI / SP500 / NASDAQ100 / DOWJONES,来自 yfiua 免费 CSV)────
    # 仅处理 UNIVERSE_INDEX_MAP 里带 ``yahoo`` 键的指数(美 / 港宽基);
    # 其余(如 A 股宽基)抛 NotImplementedError 交给更前序的 provider(akshare)。
    def get_index_components(self, index: str) -> list[str]:
        meta = UNIVERSE_INDEX_MAP.get(index)
        if meta is None or "yahoo" not in meta:
            raise NotImplementedError(
                f"yahoo 不提供指数 {index} 成分(仅支持带 yahoo 键的美/港宽基)"
            )
        cache_name = f"index_cons_{index.lower()}"
        cached = self.cache.get_universe(
            self.name, cache_name, max_age_days=INDEX_COMPONENTS_TTL_DAYS
        )
        if cached is not None and len(cached):
            if "name" not in cached.columns:
                # 旧格式缓存(只有 symbol 列):视为 miss,重拉一次重写为新格式
                _log.info("yahoo 指数 %s 缓存缺 name 列,重新拉取", index)
                cached = None
            else:
                return [str(s) for s in cached["symbol"].tolist()]
        # 恒生科技等无 yfiua 覆盖的指数:用追踪 ETF 持仓代理(yfinance 仅暴露前十大)
        if meta.get("etf"):
            symbols, names = self._etf_holdings(str(meta["etf"]))
            self.cache.put_universe(
                self.name, cache_name, pd.DataFrame({"symbol": symbols, "name": names})
            )
            return symbols
        url = f"https://yfiua.github.io/index-constituents/constituents-{index.lower()}.csv"
        self._throttle()
        _log.info("yahoo 拉取指数 %s 成分: %s", index, url)
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "djinn/0.1"})
            raw = None
            for attempt in range(2):  # E14:一次重试 + UA 头
                try:
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        raw = pd.read_csv(resp)
                    break
                except Exception:
                    if attempt == 0:
                        time.sleep(1.0)
                        continue
                    raise
        except Exception as e:
            raise ProviderError(f"yahoo 拉取指数 {index} 成分失败: {e}") from e
        if raw is None:  # pragma: no cover - 两次失败必抛,防御性兜底
            raise ProviderError(f"yahoo 拉取指数 {index} 成分失败")
        if "Symbol" not in raw.columns or len(raw) == 0:
            raise DataError(f"yahoo 指数 {index} 成分 CSV 缺少 Symbol 列或为空")
        # 并行提取 (symbol, name),保持去重保序;Name 列缺失时名称置空串
        raw_names = raw["Name"].tolist() if "Name" in raw.columns else [""] * len(raw)
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for i, s in enumerate(raw["Symbol"].tolist()):
            sym = str(s).strip()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            n = raw_names[i] if i < len(raw_names) else None
            name = str(n).strip() if n is not None and str(n).strip() else ""
            pairs.append((sym, name))
        symbols = [p[0] for p in pairs]
        names = [p[1] for p in pairs]
        self.cache.put_universe(
            self.name, cache_name, pd.DataFrame({"symbol": symbols, "name": names})
        )
        return symbols

    def _etf_holdings(self, etf_symbol: str) -> tuple[list[str], list[str]]:
        """用追踪 ETF 的前十大持仓当指数成分(恒生科技代理)。

        yfinance 的 ``funds_data`` 只暴露 ``top_holdings``(前十大),拿不到完整
        持仓;对 ~30 只的恒生科技这是近似(覆盖权重最大的前 10 只)。
        """
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise ProviderError("yfinance 未安装") from e
        self._throttle()
        try:
            th = yf.Ticker(etf_symbol).funds_data.top_holdings
        except Exception as e:
            raise ProviderError(f"yfinance 拉取 {etf_symbol} 持仓失败: {e}") from e
        if th is None or len(th) == 0:
            raise DataError(f"yfinance {etf_symbol} 持仓为空")
        symbols: list[str] = []
        names: list[str] = []
        for sym, row in th.iterrows():
            s = _hk_symbol(str(sym).strip())
            if not s:
                continue
            symbols.append(s)
            names.append(str(row.get("Name", "") or "").strip())
        return symbols, names

    def get_index_component_names(self, index: str) -> dict[str, str]:
        """指数成分 symbol → 名称映射(与 :meth:`get_index_components` 同源)。"""
        meta = UNIVERSE_INDEX_MAP.get(index)
        if meta is None or "yahoo" not in meta:
            raise NotImplementedError(
                f"yahoo 不提供指数 {index} 成分名称(仅支持带 yahoo 键的美/港宽基)"
            )
        cache_name = f"index_cons_{index.lower()}"
        cached = self.cache.get_universe(
            self.name, cache_name, max_age_days=INDEX_COMPONENTS_TTL_DAYS
        )
        if cached is None or len(cached) == 0 or "name" not in cached.columns:
            self.get_index_components(index)  # 未拉取 / 旧格式 / 超龄 → 拉取或刷新
            cached = self.cache.get_universe(
                self.name, cache_name, max_age_days=INDEX_COMPONENTS_TTL_DAYS
            )
        if cached is None or len(cached) == 0 or "name" not in cached.columns:
            return {}
        return {
            str(s): str(n)
            for s, n in zip(
                cached["symbol"].tolist(), cached["name"].tolist(), strict=False
            )
        }

    def search_symbols(
        self, query: str, market: Market | None = None
    ) -> list[tuple[str, str]]:
        """按代码联想美 / 港标的(``yf.Search``),返回 ``(symbol, name)``。

        yfinance 搜索按代码 / 英文名匹配(A 股不在此列,交给 akshare)。
        """
        if market is Market.CN:
            return []
        q = query.strip()
        if not q:
            return []
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise ProviderError("yfinance 未安装") from e
        try:
            res = yf.Search(q, max_results=20)
            quotes = list(res.quotes or [])
        except Exception as e:
            _log.warning("yfinance 搜索 %s 失败: %s", q, e)
            return []
        out: list[tuple[str, str]] = []
        for qq in quotes:
            sym = str(qq.get("symbol", "")).strip()
            if not sym:
                continue
            name = str(qq.get("shortname") or qq.get("longname") or "").strip()
            out.append((sym, name))
        return out

    def get_stock_name(self, symbol: str, market: Market | None = None) -> str:
        if market is Market.CN:
            raise NotImplementedError("yahoo 不支持 A 股名称")
        try:
            info = self._get_info_cached(symbol)
        except Exception as e:
            _log.warning("yfinance %s name 拉取失败: %s", symbol, e)
            return ""
        return str(info.get("longName") or info.get("shortName") or "")

    def get_stock_price(self, symbol: str, market: Market | None = None) -> float:
        if market is Market.CN:
            raise NotImplementedError("yahoo 不支持 A 股价格")
        try:
            info = self._get_info_cached(symbol)
        except Exception as e:
            _log.warning("yfinance %s price 拉取失败: %s", symbol, e)
            raise DataError(f"yfinance 无 {symbol} 价格") from e
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            raise DataError(f"yfinance 无 {symbol} 价格")
        f = float(price)
        if not math.isfinite(f):
            raise DataError(f"yfinance {symbol} 价格非法")
        return f

    def get_profile(self, symbol: str, market: Market | None = None) -> dict[str, Any]:
        """单标的扩展档案(估值扩展/盈利质量/财务健康/行情/分析师/公司概况)。

        从 ``Ticker.info`` 抽取,百分比字段转百分数;A 股不在此列。
        """
        if market is Market.CN:
            raise NotImplementedError("yahoo 不支持 A 股 profile")
        try:
            info = self._get_info_cached(symbol)
        except Exception as e:
            _log.warning("yfinance %s profile 拉取失败: %s", symbol, e)
            return {}
        out: dict[str, Any] = {}
        for src, dst in _PROFILE_NUM.items():
            out[dst] = _fnum(info.get(src))
        for src, dst in _PROFILE_PCT.items():
            out[dst] = _pct(info.get(src))
        for src, dst in _PROFILE_STR.items():
            v = info.get(src)
            out[dst] = str(v).strip() if v not in (None, "") else None
        # 缺失数值字段转 None(JSON 不接受 NaN/Inf,见 CLAUDE.md 序列化约定)
        return {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in out.items()
        }


def _fnum(v: object) -> float:
    """yfinance info 数值字段 → float(None / 非法 → nan)。"""
    try:
        if v is None:
            return float("nan")
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _pct(v: object) -> float:
    """yfinance 比例字段(0.15)→ 百分数(15.0),与 A 股新浪源口径一致。"""
    f = _fnum(v)
    return float("nan") if math.isnan(f) else f * 100.0


def _float_cap(info: dict[str, Any]) -> float:
    """流通市值 = floatShares × 现价(无字段 → nan)。"""
    shares = _fnum(info.get("floatShares"))
    price = _fnum(info.get("currentPrice") or info.get("regularMarketPrice"))
    if math.isnan(shares) or math.isnan(price):
        return float("nan")
    return shares * price


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """逐项除法,除零 / 非法 → NaN(抑制 inf)。"""
    return (num / den).replace([float("inf"), float("-inf")], float("nan"))


def _hk_symbol(sym: str) -> str:
    """ETF 持仓里的港股符号 → 标准 ``.HK`` 后缀。

    yfinance ``top_holdings`` 的港股符号不一致:多数带 ``.HK``(如 ``0700.HK``),
    少数是纯数字 5 位代码(如 ``01211``)。统一为 Yahoo 4 位代码 + ``.HK``。
    """
    s = sym.strip()
    if not s:
        return ""
    if s.upper().endswith(".HK"):
        return s
    if s.isdigit():
        code = s.lstrip("0") or "0"
        return f"{code.zfill(4)}.HK"
    return s


# 股票详情扩展字段:info 键 → 规范化字段名(百分比字段用 _pct,其余用 _fnum)。
_PROFILE_NUM = {
    # 估值扩展
    "forwardPE": "forward_pe",
    "trailingEps": "eps_ttm",
    "forwardEps": "forward_eps",
    "pegRatio": "peg_ratio",
    "bookValue": "book_value",
    "enterpriseValue": "enterprise_value",
    "enterpriseToEbitda": "ev_to_ebitda",
    "beta": "beta",
    # 财务健康
    "currentRatio": "current_ratio",
    "quickRatio": "quick_ratio",
    "debtToEquity": "debt_to_equity",
    "totalCash": "total_cash",
    "totalDebt": "total_debt",
    "freeCashflow": "free_cashflow",
    # 行情动量
    "fiftyTwoWeekHigh": "fifty_two_week_high",
    "fiftyTwoWeekLow": "fifty_two_week_low",
    "fiftyDayAverage": "fifty_day_avg",
    "twoHundredDayAverage": "two_hundred_day_avg",
    # 分析师
    "targetMeanPrice": "target_mean_price",
    "targetHighPrice": "target_high_price",
    "targetLowPrice": "target_low_price",
    "numberOfAnalystOpinions": "number_of_analysts",
    # 分红
    "dividendRate": "dividend_rate",
}

# 百分比字段(用 _pct 而非 _fnum)。
_PROFILE_PCT = {
    "operatingMargins": "operating_margin",
    "profitMargins": "profit_margin",
    "returnOnAssets": "return_on_assets",
    "trailingAnnualDividendYield": "dividend_yield",
}

# 字符串字段(直接透传)。
_PROFILE_STR = {
    "sector": "sector",
    "industry": "industry",
    "recommendationKey": "recommendation",
    "website": "website",
    "longBusinessSummary": "summary",
}

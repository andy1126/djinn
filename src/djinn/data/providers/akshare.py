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
    COL_FLOAT_CAP,
    COL_GROSS_MARGIN,
    COL_HIGH,
    COL_IS_SUSPENDED,
    COL_LOW,
    COL_MARKET_CAP,
    COL_OPEN,
    COL_PB,
    COL_PE,
    COL_PROFIT_YOY,
    COL_RAW_CLOSE,
    COL_REVENUE_YOY,
    COL_ROE,
    COL_SPLIT_RATIO,
    COL_VOLUME,
    Adjust,
    Market,
)
from djinn.data.universe import UNIVERSE_INDEX_MAP, normalize_cn_symbol
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

    # ── 限流 ────────────────────────────────────────────
    def _throttle(self) -> None:
        if self.rate_limit_sec > 0:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.rate_limit_sec:
                time.sleep(self.rate_limit_sec - elapsed)
            self._last_request = time.monotonic()

    # ── 股票池 / 行业 / 基本面(Phase 0 扩展)─────────────────
    def _spot_df(self) -> pd.DataFrame:
        """全 A 股实时快照(``stock_zh_a_spot_em``),universe 缓存整帧复用。"""
        cached = self.cache.get_universe(self.name, "spot_a")
        if cached is not None and len(cached):
            return cached
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError("akshare 未安装") from e
        self._throttle()
        _log.info("akshare 拉取全 A 股快照 stock_zh_a_spot_em")
        try:
            raw = ak.stock_zh_a_spot_em()
        except Exception as e:
            raise ProviderError(f"akshare 全 A 股快照失败: {e}") from e
        if raw is None or len(raw) == 0:
            raise DataError("akshare 全 A 股快照返回空")
        df = self._normalize_spot(raw)
        self.cache.put_universe(self.name, "spot_a", df)
        return df

    @staticmethod
    def _normalize_spot(raw: pd.DataFrame) -> pd.DataFrame:
        """``stock_zh_a_spot_em`` 原始列 → 规范化(symbol 后缀 + 估值字段)。"""
        df = raw.copy()
        # 代码 → 标准后缀 symbol,作为索引
        df["symbol"] = df["代码"].astype(str).map(normalize_cn_symbol)
        df = df.set_index("symbol")
        out = pd.DataFrame(index=df.index)
        out["name"] = df.get("名称", "")
        out["market"] = Market.CN.value
        # 估值字段(东财快照口径;缺失列降级为 NaN)
        num = {
            COL_PE: "市盈率-动态",
            COL_PB: "市净率",
            COL_MARKET_CAP: "总市值",
            COL_FLOAT_CAP: "流通市值",
        }
        for dst, src in num.items():
            out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df else pd.NA
        return out

    def get_stock_list(self, market: Market | None = None) -> pd.DataFrame:
        if market is not None and market is not Market.CN:
            raise NotImplementedError("akshare get_stock_list 仅支持 A 股")
        df = self._spot_df()
        return df[["name", "market"]].copy()

    def get_index_components(self, index: str) -> list[str]:
        # index 既可为 UNIVERSE_INDEX_MAP 键(如 CSI300),也可为 akshare 纯代码
        meta = UNIVERSE_INDEX_MAP.get(index)
        if meta is not None:
            if meta.get("market") is not Market.CN or "akshare" not in meta:
                raise NotImplementedError(
                    f"akshare 不提供指数 {index} 成分(仅支持 A 股宽基)"
                )
            code = str(meta["akshare"])
        else:
            code = _normalize_ak_code(index)
        cache_name = f"index_cons_{code}"
        cached = self.cache.get_universe(self.name, cache_name)
        if cached is not None and len(cached):
            return [str(s) for s in cached["symbol"].tolist()]
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError("akshare 未安装") from e
        self._throttle()
        _log.info("akshare 拉取指数 %s 成分 index_stock_cons", code)
        try:
            raw = ak.index_stock_cons(symbol=code)
        except Exception as e:
            raise ProviderError(f"akshare 拉取指数 {code} 成分失败: {e}") from e
        if raw is None or len(raw) == 0:
            raise DataError(f"akshare 指数 {code} 成分返回空")
        # 兼容多种返回列名:品种代码 / con_code / 成分券代码
        col = next(
            (
                c
                for c in ("品种代码", "成分券代码", "con_code", "code")
                if c in raw.columns
            ),
            None,
        )
        if col is None:
            raise DataError(f"akshare 指数成分返回缺少代码列: {list(raw.columns)}")
        symbols = [normalize_cn_symbol(str(c)) for c in raw[col].tolist()]
        self.cache.put_universe(
            self.name, cache_name, pd.DataFrame({"symbol": symbols})
        )
        return symbols

    def get_industry_map(self, symbols: list[str]) -> dict[str, str]:
        rev = self._industry_reverse_map()
        return {s: rev[s] for s in symbols if s in rev}

    def _industry_reverse_map(self) -> dict[str, str]:
        """symbol → 行业名(东财行业板块成分反向索引,universe 缓存)。"""
        cached = self.cache.get_universe(self.name, "industry_map")
        if cached is not None and len(cached):
            return dict(
                zip(
                    cached["symbol"].astype(str),
                    cached["industry"].astype(str),
                    strict=True,
                )
            )
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError("akshare 未安装") from e
        self._throttle()
        _log.info("akshare 拉取行业板块列表 stock_board_industry_name_em")
        try:
            boards = ak.stock_board_industry_name_em()
        except Exception as e:
            raise ProviderError(f"akshare 行业板块列表失败: {e}") from e
        if boards is None or len(boards) == 0:
            raise DataError("akshare 行业板块列表返回空")
        name_col = next(
            (c for c in ("板块名称", "行业名称", "name") if c in boards.columns), None
        )
        if name_col is None:
            raise DataError(f"akshare 行业板块缺少名称列: {list(boards.columns)}")
        records: list[tuple[str, str]] = []
        for industry in boards[name_col].astype(str).tolist():
            self._throttle()
            try:
                cons = ak.stock_board_industry_cons_em(symbol=industry)
            except Exception as e:  # 单个板块失败不致命,跳过
                _log.warning("akshare 行业 %s 成分拉取失败,跳过: %s", industry, e)
                continue
            if cons is None or len(cons) == 0 or "代码" not in cons.columns:
                continue
            for c in cons["代码"].astype(str).tolist():
                records.append((normalize_cn_symbol(c), industry))
        if not records:
            raise DataError("akshare 行业成分反查为空")
        df = pd.DataFrame(records, columns=["symbol", "industry"]).drop_duplicates(
            "symbol"
        )
        self.cache.put_universe(self.name, "industry_map", df)
        return dict(zip(df["symbol"], df["industry"], strict=True))

    def get_fundamentals(self, symbols: list[str], when: date) -> pd.DataFrame:
        """截面估值快照(PE/PB/市值,东财实时口径)。

        注意:``stock_zh_a_spot_em`` 为实时快照,非历史 point-in-time;估值类因子
        在回测中取"最新可见"口径。ROE / 毛利率等财务字段由
        :meth:`get_fundamentals_history` 提供(带 announce_date,真正 PIT)。
        """
        spot = self._spot_df()
        sub = spot.reindex([s for s in symbols if s in spot.index])
        out = pd.DataFrame(index=sub.index)
        for col in (COL_MARKET_CAP, COL_FLOAT_CAP, COL_PE, COL_PB):
            out[col] = pd.to_numeric(sub[col], errors="coerce")
        return out

    def get_fundamentals_history(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        """单标的财务指标时序(``stock_financial_analysis_indicator``)。

        akshare 该接口按报告期返回,不含公告日;此处以 ``report_date + 45 天``
        近似公告日(A 股季报 / 年报法定披露滞后上限内),明确标注为近似值。
        精确 PIT 请用 tushare ``fina_indicator``(带 ann_date)。
        """
        code = _normalize_ak_code(symbol)
        cache_symbol = f"finhist_{code}"
        cached = self.cache.get_fundamentals(self.name, cache_symbol)
        if cached is not None and len(cached):
            return cached
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError("akshare 未安装") from e
        self._throttle()
        _log.info("akshare 拉取 %s 财务指标 stock_financial_analysis_indicator", code)
        try:
            raw = ak.stock_financial_analysis_indicator(
                symbol=code, start_year=str(start.year)
            )
        except Exception as e:
            raise ProviderError(f"akshare 拉取 {symbol} 财务指标失败: {e}") from e
        if raw is None or len(raw) == 0:
            raise DataError(f"akshare {symbol} 财务指标返回空")
        df = self._normalize_fin_history(raw)
        self.cache.put_fundamentals(self.name, cache_symbol, df)
        return df

    @staticmethod
    def _normalize_fin_history(raw: pd.DataFrame) -> pd.DataFrame:
        """财务指标原始列 → 规范化(含近似 announce_date)。"""
        from djinn.data.schema import COL_ANNOUNCE_DATE, COL_REPORT_DATE

        df = raw.copy()
        date_col = next(
            (c for c in ("日期", "报告期", "date") if c in df.columns), None
        )
        if date_col is None:
            raise DataError(f"akshare 财务指标缺少日期列: {list(df.columns)}")
        rep = pd.to_datetime(df[date_col])
        out = pd.DataFrame(index=pd.DatetimeIndex(rep, name="date"))
        col_map = {
            COL_ROE: ("净资产收益率(%)", "净资产收益率"),
            COL_GROSS_MARGIN: ("销售毛利率(%)", "销售毛利率"),
            COL_REVENUE_YOY: ("主营业务收入增长率(%)", "主营业务收入增长率"),
            COL_PROFIT_YOY: ("净利润增长率(%)", "净利润增长率"),
        }
        for dst, candidates in col_map.items():
            src = next((c for c in candidates if c in df.columns), None)
            out[dst] = (
                pd.to_numeric(df[src], errors="coerce").to_numpy()
                if src is not None
                else float("nan")
            )
        out[COL_REPORT_DATE] = rep
        out[COL_ANNOUNCE_DATE] = rep + pd.Timedelta(days=45)  # 近似公告日
        return out.sort_index()

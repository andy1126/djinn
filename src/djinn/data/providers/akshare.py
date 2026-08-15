"""AkShare 数据提供器(A 股 / 渓股,免费免 key)。

Phase 1 数据层:A 股日线通过 ``akshare.stock_zh_a_hist`` 拉取,含复权与停牌。
依赖为可选(``pip install djinn[akshare]``),缺失时 :meth:`supports` 返回 False。
"""

from __future__ import annotations

import math
import threading
import time
from datetime import date, timedelta

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
    COL_NET_PROFIT,
    COL_OCF,
    COL_OPEN,
    COL_PB,
    COL_PE,
    COL_PROFIT_YOY,
    COL_PS,
    COL_RAW_CLOSE,
    COL_REVENUE,
    COL_REVENUE_YOY,
    COL_ROE,
    COL_SPLIT_RATIO,
    COL_TOTAL_ASSETS,
    COL_VOLUME,
    Adjust,
    Market,
)
from djinn.data.universe import (
    INDEX_COMPONENTS_TTL_DAYS,
    UNIVERSE_INDEX_MAP,
    normalize_cn_symbol,
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


def _sina_symbol(code: str) -> str:
    """纯 6 位代码 → 新浪源符号(``600519`` → ``sh600519``)。

    新浪 ``stock_zh_a_daily`` 用 ``sh/sz/bj`` 前缀区分交易所。
    """
    if code.startswith(("60", "68", "9", "11", "13")):
        return f"sh{code}"
    if code.startswith(("43", "83", "87", "88")):
        return f"bj{code}"
    return f"sz{code}"


def _normalize_name(s: str) -> str:
    """新浪名称清洗:全角→半角(``Ａ``→``A``)、去空白(``万  科Ａ``→``万科A``)。

    新浪 ``stock_info_a_code_name`` 的名称带全角字母与内部空格,直接子串匹配
    会漏搜(如搜「万科」匹配不到「万  科Ａ」)。
    """
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:  # 全角 ASCII 区 → 半角
            ch = chr(code - 0xFEE0)
        elif ch.isspace():
            continue  # 去掉内部空白(含全角空格)
        out.append(ch)
    return "".join(out)


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
        self._rate_lock = threading.Lock()  # 限速临界区(跨线程串行)

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
        if DataCache.covers_soft(cached, start, end, slack_days=12):
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
            raise ProviderError(
                "akshare 未安装,请执行 uv pip install -e '.[akshare]'"
            ) from e
        self._throttle()
        code = _normalize_ak_code(symbol)
        ak_adjust = {Adjust.NONE: "", Adjust.FORWARD: "qfq", Adjust.BACKWARD: "hfq"}[
            adjust
        ]
        _log.info("akshare 拉取 %s [%s ~ %s] adjust=%s", code, start, end, ak_adjust)
        try:
            raw = ak.stock_zh_a_daily(
                symbol=_sina_symbol(code),
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
        if self.rate_limit_sec <= 0:
            return
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.rate_limit_sec:
                time.sleep(self.rate_limit_sec - elapsed)
            self._last_request = time.monotonic()

    # ── 股票池 / 行业 / 基本面(Phase 0 扩展)─────────────────
    def _spot_df(self) -> pd.DataFrame:
        """全 A 股实时快照(``stock_zh_a_spot_em``),universe 缓存整帧复用。"""
        cached = self.cache.get_universe(self.name, "spot_a", max_age_days=7)
        if cached is not None and len(cached):
            return cached
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "akshare 未安装,请执行 uv pip install -e '.[akshare]'"
            ) from e
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
        if "最新价" in df.columns:
            price = pd.to_numeric(df["最新价"], errors="coerce").to_numpy()
        else:
            price = pd.Series(float("nan"), index=df.index).to_numpy()
        out["price"] = price
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

    def _code_name_df(self) -> pd.DataFrame:
        """全 A 股代码 + 名称(新浪 ``stock_info_a_code_name``),universe 缓存。

        东财 ``stock_zh_a_spot_em`` 在当前网络不可达时,搜索 / 名称 / 列表
        改走新浪源(仅 code + name,无估值字段)。整帧缓存,按月刷新。
        """
        cached = self.cache.get_universe(self.name, "code_name_sina", max_age_days=7)
        if cached is not None and len(cached):
            return cached
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "akshare 未安装,请执行 uv pip install -e '.[akshare]'"
            ) from e
        _log.info("akshare 拉取全 A 股代码名称 stock_info_a_code_name")
        try:
            raw = ak.stock_info_a_code_name()
        except Exception as e:
            raise ProviderError(f"akshare 全 A 股代码名称失败: {e}") from e
        if raw is None or len(raw) == 0:
            raise DataError("akshare 全 A 股代码名称返回空")
        symbols = [normalize_cn_symbol(str(c)) for c in raw["code"].tolist()]
        names = [_normalize_name(str(n)) for n in raw["name"].tolist()]
        out = pd.DataFrame({"name": names}, index=pd.Index(symbols, name="symbol"))
        out["market"] = Market.CN.value
        self.cache.put_universe(self.name, "code_name_sina", out)
        return out

    def get_stock_list(self, market: Market | None = None) -> pd.DataFrame:
        if market is not None and market is not Market.CN:
            raise NotImplementedError("akshare get_stock_list 仅支持 A 股")
        df = self._code_name_df()
        return df[["name", "market"]].copy()

    def _index_cache_name(self, index: str) -> str:
        """校验 akshare 支持性并返回指数成分的 universe 缓存键。

        index 既可为 UNIVERSE_INDEX_MAP 键(如 CSI300),也可为 akshare 纯代码。
        """
        meta = UNIVERSE_INDEX_MAP.get(index)
        if meta is not None:
            if meta.get("market") is not Market.CN or "akshare" not in meta:
                raise NotImplementedError(
                    f"akshare 不提供指数 {index} 成分(仅支持 A 股宽基)"
                )
            return f"index_cons_{meta['akshare']}"
        return f"index_cons_{_normalize_ak_code(index)}"

    def get_index_components(self, index: str) -> list[str]:
        cache_name = self._index_cache_name(index)
        cached = self.cache.get_universe(
            self.name, cache_name, max_age_days=INDEX_COMPONENTS_TTL_DAYS
        )
        if cached is not None and len(cached):
            if "name" not in cached.columns:
                # 旧格式缓存(只有 symbol 列):视为 miss,重拉一次重写为新格式
                _log.info("akshare 指数 %s 缓存缺 name 列,重新拉取", index)
                cached = None
            else:
                return [str(s) for s in cached["symbol"].tolist()]
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "akshare 未安装,请执行 uv pip install -e '.[akshare]'"
            ) from e
        code = cache_name.removeprefix("index_cons_")
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
        # 并行提取名称(兼容多列名),缺失置空串
        names = [""] * len(symbols)
        name_col = next(
            (
                c
                for c in ("品种名称", "成分券名称", "con_name", "name")
                if c in raw.columns
            ),
            None,
        )
        if name_col is not None:
            raw_name = raw[name_col].tolist()
            for i in range(len(symbols)):
                n = raw_name[i] if i < len(raw_name) else None
                names[i] = str(n).strip() if n is not None and str(n).strip() else ""
        self.cache.put_universe(
            self.name, cache_name, pd.DataFrame({"symbol": symbols, "name": names})
        )
        return symbols

    def get_index_component_names(self, index: str) -> dict[str, str]:
        """指数成分 symbol → 名称映射(与 :meth:`get_index_components` 同源)。"""
        cache_name = self._index_cache_name(index)  # 非 A 股抛 NotImplementedError
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

    def get_industry_map(self, symbols: list[str]) -> dict[str, str]:
        rev = self._industry_reverse_map()
        return {s: rev[s] for s in symbols if s in rev}

    def _industry_reverse_map(self) -> dict[str, str]:
        """symbol → 行业名(东财行业板块成分反向索引,universe 缓存)。"""
        cached = self.cache.get_universe(self.name, "industry_map", max_age_days=7)
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
            raise ProviderError(
                "akshare 未安装,请执行 uv pip install -e '.[akshare]'"
            ) from e
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
            raise ProviderError(
                "akshare 未安装,请执行 uv pip install -e '.[akshare]'"
            ) from e
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
            COL_REVENUE: ("主营业务收入", "营业收入"),
            COL_NET_PROFIT: ("净利润", "归母净利润"),
            COL_OCF: ("经营活动产生的现金流量净额", "经营现金流"),
            COL_TOTAL_ASSETS: ("总资产", "资产总计"),
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

    # ── 日频估值(根治 EP/BP/SP 前视)─────────────────────
    def get_daily_valuation(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """单标的日频估值时序(``stock_a_indicator_lg`` 乐咕),index=交易日。

        覆盖 pe/pb/ps(无市值);估值是**日频行情衍生序列**,随收盘价每日更新,
        天然 point-in-time,消除"用今日估值给三年前打分"的前视。整帧缓存,
        覆盖不足时重拉(``DataCache.covers`` 判定)。
        """
        code = _normalize_ak_code(symbol)
        cache_symbol = f"valuation_{code}"
        cached = self.cache.get_fundamentals(self.name, cache_symbol)
        if cached is not None and len(cached):
            cached.index = pd.to_datetime(cached.index)
            if DataCache.covers(cached, start, end):
                return cached.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "akshare 未安装,请执行 uv pip install -e '.[akshare]'"
            ) from e
        self._throttle()
        _log.info("akshare 拉取 %s 日频估值 stock_a_indicator_lg", code)
        try:
            raw = ak.stock_a_indicator_lg(
                symbol=code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
        except Exception as e:
            raise ProviderError(f"akshare 拉取 {symbol} 日频估值失败: {e}") from e
        if raw is None or len(raw) == 0:
            raise DataError(f"akshare {symbol} 日频估值返回空")
        df = self._normalize_valuation(raw)
        self.cache.put_fundamentals(self.name, cache_symbol, df)
        return df

    @staticmethod
    def _normalize_valuation(raw: pd.DataFrame) -> pd.DataFrame:
        """``stock_a_indicator_lg`` 原始列 → 规范化(pe/pb/ps,index=交易日)。"""
        df = raw.copy()
        date_col = next(
            (c for c in ("trade_date", "日期", "date") if c in df.columns), None
        )
        if date_col is None:
            raise DataError(f"akshare 日频估值缺少日期列: {list(df.columns)}")
        out = pd.DataFrame(index=pd.to_datetime(df[date_col]))
        col_map = {
            COL_PE: ("pe", "pe_ttm", "市盈率", "市盈率TTM"),
            COL_PB: ("pb", "市净率"),
            COL_PS: ("ps", "ps_ttm", "市销率", "市销率TTM"),
        }
        for dst, candidates in col_map.items():
            src = next((c for c in candidates if c in df.columns), None)
            out[dst] = (
                pd.to_numeric(df[src], errors="coerce").to_numpy()
                if src is not None
                else float("nan")
            )
        return out.sort_index()

    # ── 每股现金分红(股息率因子)─────────────────────────
    def get_daily_dividends(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """单标的每股现金分红事件序列(index=除息日,``dividend`` 列)。

        新浪源(与项目 A 股行情源一致,东财不可达):``stock_history_dividend_detail``
        一次返回全部历史分红,整帧缓存后按区间切片(分红为静态历史,复用 ``finhist_``
        式的落盘缓存)。
        """
        code = _normalize_ak_code(symbol)
        cache_symbol = f"dividends_{code}"
        cached = self.cache.get_fundamentals(self.name, cache_symbol)
        if cached is not None and len(cached):
            cached.index = pd.to_datetime(cached.index)
            return cached.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "akshare 未安装,请执行 uv pip install -e '.[akshare]'"
            ) from e
        self._throttle()
        _log.info("akshare 拉取 %s 历史分红 stock_history_dividend_detail", code)
        try:
            raw = ak.stock_history_dividend_detail(
                symbol=code, indicator="分红", date=""
            )
        except Exception as e:
            raise ProviderError(f"akshare 拉取 {symbol} 分红失败: {e}") from e
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        df = self._normalize_dividends(raw)
        if len(df):
            self.cache.put_fundamentals(self.name, cache_symbol, df)
        return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    @staticmethod
    def _normalize_dividends(raw: pd.DataFrame) -> pd.DataFrame:
        """``stock_history_dividend_detail`` → 规范化(index=除息日,``dividend`` 列)。

        仅保留「实施」且除权除息日有效的记录;新浪 ``派息(税前)`` 是**每 10 股**
        口径,需 ÷10 得每股现金分红(元);同日多笔(罕见)求和去重。
        """
        df = raw.copy()
        if "进度" in df.columns:
            df = df[df["进度"] == "实施"]
        if "除权除息日" not in df.columns or "派息" not in df.columns:
            return pd.DataFrame()
        ex = pd.to_datetime(df["除权除息日"], errors="coerce")
        cash = pd.to_numeric(df["派息"], errors="coerce") / 10.0
        out = pd.DataFrame({COL_DIVIDEND: cash.to_numpy()}, index=ex)
        out = out.dropna(subset=[COL_DIVIDEND]).sort_index()
        if out.index.has_duplicates:
            out = out.groupby(level=0).sum()
        return out

    def search_symbols(
        self, query: str, market: Market | None = None
    ) -> list[tuple[str, str]]:
        """按代码 / 名称子串匹配 A 股(全 A 股快照),返回 ``(symbol, name)``。"""
        if market is not None and market is not Market.CN:
            return []
        q = _normalize_name(query).upper()
        if not q:
            return []
        df = self._code_name_df()
        out: list[tuple[str, str]] = []
        for sym, row in df.iterrows():
            name = str(row.get("name", "") or "")
            if q in str(sym).upper() or q in name.upper():
                out.append((str(sym), name))
            if len(out) >= 20:
                break
        return out

    def get_stock_name(self, symbol: str, market: Market | None = None) -> str:
        if market is not None and market is not Market.CN:
            raise NotImplementedError("akshare 仅支持 A 股名称")
        df = self._code_name_df()
        return str(df.loc[symbol, "name"]) if symbol in df.index else ""

    def get_stock_price(self, symbol: str, market: Market | None = None) -> float:
        if market is not None and market is not Market.CN:
            raise NotImplementedError("akshare 仅支持 A 股价格")
        # 新浪日线最新收盘价(东财 _spot_df 不可达;新浪日线覆盖全板含科创/创业)
        code = _normalize_ak_code(symbol)
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "akshare 未安装,请执行 uv pip install -e '.[akshare]'"
            ) from e
        self._throttle()
        end = date.today()
        start = end - timedelta(days=30)
        try:
            raw = ak.stock_zh_a_daily(
                symbol=_sina_symbol(code),
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="",
            )
        except Exception as e:
            raise DataError(f"akshare 拉取 {symbol} 价格失败: {e}") from e
        if raw is None or len(raw) == 0:
            raise DataError(f"akshare 无 {symbol} 价格")
        f = float(pd.to_numeric(raw["close"].iloc[-1], errors="coerce"))
        if not math.isfinite(f):
            raise DataError(f"akshare {symbol} 价格非法")
        return f

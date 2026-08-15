"""基本面路由器:按市场把截面 / 时序基本面请求分发给最合适的 provider。

路由规则(与行情 ``ProviderRegistry`` 一致的 ``supports()`` 优先级):
- A 股(CN) → AkShare 优先,Tushare 补充;
- 港股 / 美股(HK / US) → Yahoo 兜底。

职责:
- 字段口径归一化(provider 输出已是规范化 ``COL_*`` 列,此处仅做合并对齐);
- **point-in-time 不变量**:截面快照的财务字段(ROE / 毛利率 / 成长)逐标的经
  :func:`asof_snapshot` 按 ``announce_date <= when`` 取最新一期,杜绝未来函数;
  估值字段(PE/PB/市值)取 provider 快照口径。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from djinn.data.fundamentals import FundamentalsSource, asof_snapshot
from djinn.data.provider import DataProvider
from djinn.data.schema import (
    COL_MARKET_CAP,
    FUNDAMENTAL_VALUE_COLUMNS,
    Market,
    detect_market,
)
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

# 财务字段需要回看的最大窗口(覆盖季报 / 年报披露周期)。
_HISTORY_LOOKBACK_DAYS = 400


class FundamentalsRouter(FundamentalsSource):
    """按市场路由基本面请求的 :class:`FundamentalsSource` 实现。"""

    name = "router"

    def __init__(self, providers: list[DataProvider]) -> None:
        self._providers = list(providers)

    # ── 路由 ────────────────────────────────────────────
    def _resolve(
        self, symbol: str, market: Market | None = None
    ) -> DataProvider | None:
        m = market or detect_market(symbol)
        for p in self._providers:
            try:
                if p.supports(symbol, m):
                    return p
            except Exception:  # provider supports 异常不致命
                continue
        return None

    def _group_by_provider(
        self, symbols: list[str], market: Market | None
    ) -> dict[DataProvider, list[str]]:
        groups: dict[DataProvider, list[str]] = {}
        for s in symbols:
            p = self._resolve(s, market)
            if p is None:
                _log.warning("无 provider 支持 %s 基本面,跳过", s)
                continue
            groups.setdefault(p, []).append(s)
        return groups

    # ── 截面快照 ────────────────────────────────────────
    def get_snapshot(
        self, symbols: list[str], when: date, market: Market | None = None
    ) -> pd.DataFrame:
        """``when`` 当日截面快照(point-in-time)。

        Returns:
            index=symbol,columns 覆盖 ``FUNDAMENTAL_VALUE_COLUMNS``(缺源列为 NaN)。
        """
        frames: list[pd.DataFrame] = []
        for provider, group in self._group_by_provider(symbols, market).items():
            # 估值字段(快照口径)
            try:
                val = provider.get_fundamentals(group, when)
            except NotImplementedError:
                val = pd.DataFrame(index=group)
            except Exception as e:
                _log.warning("%s 估值快照失败: %s", provider.name, e)
                val = pd.DataFrame(index=group)
            # 财务字段(PIT:逐标的 asof)
            fin = self._financial_snapshot(provider, group, when)
            # 财务比率(roe/毛利率/同比)以 PIT 时序为准,丢弃快照中的同名列
            overlap = [c for c in fin.columns if c in val.columns]
            if overlap:
                val = val.drop(columns=overlap)
            merged = val.join(fin, how="outer")
            frames.append(merged)
        if frames:
            out = pd.concat(frames)
        else:
            out = pd.DataFrame(index=list(symbols))
        out = out.reindex(symbols)
        for col in FUNDAMENTAL_VALUE_COLUMNS:
            if col not in out.columns:
                out[col] = float("nan")
        return out

    def _financial_snapshot(
        self, provider: DataProvider, symbols: list[str], when: date
    ) -> pd.DataFrame:
        """逐标的取财报时序并按 announce_date asof,得到 point-in-time 财务字段。"""
        rows: dict[str, pd.Series] = {}
        start = when - timedelta(days=_HISTORY_LOOKBACK_DAYS)
        for s in symbols:
            try:
                hist = provider.get_fundamentals_history(s, start, when)
            except NotImplementedError:
                break  # 该 provider 不支持时序,整组跳过
            except Exception as e:
                _log.debug("%s %s 财务时序失败: %s", provider.name, s, e)
                continue
            latest = asof_snapshot(hist, when)
            if latest is not None:
                rows[s] = latest.drop(
                    labels=["report_date", "announce_date"], errors="ignore"
                )
        if not rows:
            return pd.DataFrame(index=symbols)
        return pd.DataFrame.from_dict(rows, orient="index")

    # ── 时序 ────────────────────────────────────────────
    def get_history(
        self, symbol: str, start: date, end: date, market: Market | None = None
    ) -> pd.DataFrame:
        """单标的财报时序(含 announce_date/report_date),供成长/质量因子回看。"""
        provider = self._resolve(symbol, market)
        if provider is None:
            raise ValueError(f"无 provider 支持 {symbol} 基本面时序")
        return provider.get_fundamentals_history(symbol, start, end)

    def get_daily_valuation(
        self, symbol: str, start: date, end: date, market: Market | None = None
    ) -> pd.DataFrame:
        """单标的日频估值时序(按 provider 优先级路由)。

        依注册顺序遍历 provider(tushare 有 token 优先、akshare 兜底),首个
        ``supports`` 且返回非空日频估值者命中;全部不支持返回空 DataFrame(由
        调用方退化为快照口径)。
        """
        m = market or detect_market(symbol)
        for provider in self._providers:
            try:
                if not provider.supports(symbol, m):
                    continue
            except Exception:
                continue
            try:
                daily = provider.get_daily_valuation(symbol, start, end)
            except NotImplementedError:
                continue
            except Exception as e:
                _log.debug("%s 日频估值 %s 失败: %s", provider.name, symbol, e)
                continue
            if daily is not None and len(daily):
                return daily
        return pd.DataFrame()

    def get_daily_dividends(
        self, symbol: str, start: date, end: date, market: Market | None = None
    ) -> pd.DataFrame:
        """单标的每股现金分红事件序列(按 provider 优先级路由,同估值)。"""
        m = market or detect_market(symbol)
        for provider in self._providers:
            try:
                if not provider.supports(symbol, m):
                    continue
            except Exception:
                continue
            try:
                daily = provider.get_daily_dividends(symbol, start, end)
            except NotImplementedError:
                continue
            except Exception as e:
                _log.debug("%s 分红 %s 失败: %s", provider.name, symbol, e)
                continue
            if daily is not None and len(daily):
                return daily
        return pd.DataFrame()

    # ── 便捷:行业 / 市值代理 ─────────────────────────────
    def market_cap_snapshot(
        self, symbols: list[str], when: date, market: Market | None = None
    ) -> pd.Series:
        """``when`` 当日市值 Series(供中性化的对数市值自变量)。"""
        snap = self.get_snapshot(symbols, when, market)
        return pd.to_numeric(snap[COL_MARKET_CAP], errors="coerce")

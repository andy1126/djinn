"""股票路由:代码搜索联想 + 单股详情。

搜索:委托给 provider 的 ``search_symbols``(A 股 akshare 全表,美 / 港 yf.Search)。
详情:复用 ``FundamentalsRouter.get_snapshot`` 聚合估值 + 财务(PIT),再补 name/price。
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from djinn.api.deps import get_registry
from djinn.api.schemas import (
    StockDetail,
    StockProfile,
    SymbolSearchResponse,
    SymbolSearchResult,
)
from djinn.data.provider import ProviderRegistry
from djinn.data.providers.fundamentals_router import FundamentalsRouter
from djinn.data.schema import Market
from djinn.utils.exceptions import SymbolNotFoundError
from djinn.utils.logging import get_logger

router = APIRouter(prefix="/stocks", tags=["stocks"])

_log = get_logger(__name__)


def _market_from(value: str | None) -> Market | None:
    try:
        return Market(value) if value else None
    except ValueError:
        return None


@router.get("/search", response_model=SymbolSearchResponse)
async def search_symbols(
    q: str,
    market: str | None = None,
    registry: ProviderRegistry = Depends(get_registry),
) -> SymbolSearchResponse:
    """按代码 / 名称搜索标的(三市场,返回联想建议)。"""
    m = _market_from(market)

    def _do_search() -> list[SymbolSearchResult]:
        results: list[SymbolSearchResult] = []
        for p in registry.providers:
            try:
                pairs = p.search_symbols(q, m)
            except NotImplementedError:
                continue
            except Exception as e:
                _log.warning("provider %s 搜索 %s 失败: %s", p.name, q, e)
                continue
            for sym, name in pairs:
                results.append(
                    SymbolSearchResult(
                        symbol=sym, market=str(m.value if m else "auto"), name=name
                    )
                )
            if results:
                break  # 首个支持搜索的 provider 返回结果
        return results

    # E2:provider 网络调用卸载到线程,不阻塞事件循环
    results = await asyncio.to_thread(_do_search)
    return SymbolSearchResponse(query=q, results=results)


@router.get("/{symbol}", response_model=StockDetail)
async def stock_detail(
    symbol: str,
    market: str | None = None,
    registry: ProviderRegistry = Depends(get_registry),
) -> StockDetail:
    """单只股票详情(估值 + 财务 + 价格,字段按数据源能力降级)。"""
    m = _market_from(market)
    try:
        # E2:provider 网络调用(快照 / 名称 / 价格 / 档案)卸载到线程
        return await asyncio.to_thread(_build_detail, registry, symbol, m)
    except SymbolNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def _build_detail(
    registry: ProviderRegistry, symbol: str, m: Market | None
) -> StockDetail:
    """同步构建单股详情(供 to_thread 卸载;含网络调用)。"""
    # 解析 provider(symbol 属于哪个市场),拿行情兜底
    provider = registry.resolve(symbol, m)

    # 估值 + 财务快照(聚合各 provider,字段缺失为 NaN)
    snap = FundamentalsRouter(registry.providers).get_snapshot(
        [symbol], date.today(), m
    )
    row = snap.loc[symbol] if symbol in snap.index else None

    def _f(col: str) -> float | None:
        if row is None or col not in row.index:
            return None
        v: Any = row[col]
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if _finite(f) else None

    # 名称 / 价格 / 扩展档案:优先从 provider 自身能力取
    name = _provider_name(provider, symbol, m)
    price = _provider_price(provider, symbol, m)
    profile = _provider_profile(provider, symbol, m)

    return StockDetail(
        symbol=symbol,
        market=str(m.value if m else "auto"),
        name=name or "",
        price=price,
        pe=_f("pe"),
        pb=_f("pb"),
        ps=_f("ps"),
        market_cap=_f("market_cap"),
        float_cap=_f("float_cap"),
        roe=_f("roe"),
        gross_margin=_f("gross_margin"),
        revenue=_f("revenue"),
        net_profit=_f("net_profit"),
        revenue_yoy=_f("revenue_yoy"),
        profit_yoy=_f("profit_yoy"),
        profile=profile,
    )


def _finite(f: float) -> bool:
    import math

    return math.isfinite(f)


def _provider_name(provider: Any, symbol: str, market: Market | None) -> str:
    """取标的名称(provider 各自实现,失败返回空串)。"""
    try:
        name = provider.get_stock_name(symbol, market)
        return str(name or "")
    except Exception:
        return ""


def _provider_price(provider: Any, symbol: str, market: Market | None) -> float | None:
    try:
        p = provider.get_stock_price(symbol, market)
        return float(p) if p is not None else None
    except Exception:
        return None


def _provider_profile(
    provider: Any, symbol: str, market: Market | None
) -> StockProfile | None:
    """取标的扩展档案(仅部分 provider 支持,失败/不支持返回 None)。"""
    try:
        data = provider.get_profile(symbol, market)
        if not data:
            return None
        return StockProfile(**data)
    except NotImplementedError:
        return None
    except Exception:
        return None

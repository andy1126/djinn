"""股票池路由:全市场股票列表 / 宽基指数成分 / 行业分布。

均委托给 provider 的非抽象接口(见 :mod:`djinn.data.provider`),按 ``supports``
优先级路由;成分 / 行业属低频数据,provider 内部已按 universe 缓存键缓存。
"""

from __future__ import annotations

import asyncio
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Query

from djinn.api.deps import get_registry
from djinn.api.jobs import _index_components, _index_components_with_provider
from djinn.api.schemas import (
    IndexComponentsResponse,
    IndexInfo,
    IndexListResponse,
    IndustryCount,
    IndustryListResponse,
    UniverseStock,
    UniverseStockListResponse,
)
from djinn.data.provider import ProviderRegistry
from djinn.data.schema import Market
from djinn.data.universe import UNIVERSE_INDEX_MAP
from djinn.utils.logging import get_logger

router = APIRouter(prefix="/universe", tags=["universe"])

_log = get_logger(__name__)


@router.get("/stock-list", response_model=UniverseStockListResponse)
async def stock_list(
    market: str | None = None,
    registry: ProviderRegistry = Depends(get_registry),
) -> UniverseStockListResponse:
    """全市场股票列表(目前仅 A 股 provider 支持)。"""
    m = Market(market) if market else None

    def _do() -> UniverseStockListResponse:
        for p in registry.providers:
            try:
                df = p.get_stock_list(m)
            except NotImplementedError:
                continue
            except Exception as e:
                raise HTTPException(
                    status_code=502, detail=f"{p.name} 拉取股票列表失败: {e}"
                ) from e
            stocks = [
                UniverseStock(
                    symbol=str(sym),
                    name=str(row.get("name", "") or ""),
                    market=str(row.get("market", "") or ""),
                )
                for sym, row in df.iterrows()
            ]
            return UniverseStockListResponse(
                market=market, count=len(stocks), stocks=stocks
            )
        raise HTTPException(status_code=501, detail="无 provider 支持全市场股票列表")

    # E2:provider 网络调用卸载到线程
    return await asyncio.to_thread(_do)


@router.get("/indexes", response_model=IndexListResponse)
async def list_indexes() -> IndexListResponse:
    """列出内置宽基指数(UNIVERSE_INDEX_MAP)。"""
    items: list[IndexInfo] = []
    for key, meta in UNIVERSE_INDEX_MAP.items():
        m = meta.get("market", "")
        market_str = m.value if isinstance(m, Market) else str(m or "")
        items.append(
            IndexInfo(
                key=key,
                name=str(meta.get("name", key)),
                market=market_str,
            )
        )
    return IndexListResponse(indexes=items)


@router.get("/index-components/{index}", response_model=IndexComponentsResponse)
async def index_components(
    index: str, registry: ProviderRegistry = Depends(get_registry)
) -> IndexComponentsResponse:
    """指数成分股代码列表(附可选成分名称,与 symbols 位置对齐)。"""

    def _do() -> IndexComponentsResponse:
        symbols, provider, errors = _index_components_with_provider(registry, index)
        if not symbols:
            detail = f"无 provider 提供指数 {index} 成分"
            if errors:
                detail += ";" + "; ".join(errors)
            raise HTTPException(status_code=501, detail=detail)
        names: list[str] = []
        if provider is not None:
            try:
                mapping = provider.get_index_component_names(index)
            except NotImplementedError:
                mapping = {}
            except Exception as e:
                _log.warning(
                    "provider %s 取指数 %s 名称失败: %s", provider.name, index, e
                )
                mapping = {}
            names = [str(mapping.get(s, "")) for s in symbols]
        return IndexComponentsResponse(
            index=index, count=len(symbols), symbols=symbols, names=names
        )

    # E2:provider 网络调用卸载到线程
    return await asyncio.to_thread(_do)


@router.get("/industries", response_model=IndustryListResponse)
async def industries(
    index: str = Query(default="CSI300", description="行业统计范围(指数键)"),
    symbols: list[str] | None = Query(
        default=None, description="显式标的(优先于 index)"
    ),
    registry: ProviderRegistry = Depends(get_registry),
) -> IndustryListResponse:
    """行业分布(统计范围内各行业股票数,按数量降序)。"""

    def _do() -> IndustryListResponse:
        scope = (
            [str(s) for s in symbols] if symbols else _index_components(registry, index)
        )
        if not scope:
            raise HTTPException(status_code=501, detail="无法解析行业统计范围标的池")
        for p in registry.providers:
            try:
                mapping = p.get_industry_map(scope)
            except NotImplementedError:
                continue
            except Exception as e:
                raise HTTPException(
                    status_code=502, detail=f"{p.name} 拉取行业映射失败: {e}"
                ) from e
            if mapping:
                counts = Counter(str(v) for v in mapping.values())
                items = [
                    IndustryCount(name=name, count=c)
                    for name, c in sorted(
                        counts.items(), key=lambda kv: (-kv[1], kv[0])
                    )
                ]
                return IndustryListResponse(industries=items)
        raise HTTPException(status_code=501, detail="无 provider 支持行业映射")

    # E2:provider 网络调用(行业映射首建分钟级)卸载到线程
    return await asyncio.to_thread(_do)

"""数据路由:拉取/缓存/基准。"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException

from djinn.api.deps import get_cache
from djinn.api.schemas import CacheResponse, DataFetchRequest
from djinn.data import DataCache, default_registry
from djinn.data.schema import Adjust, Market

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/fetch")
async def fetch_data(req: DataFetchRequest, cache: DataCache = Depends(get_cache)):
    """拉取标的 OHLCV 数据。"""
    try:
        market = Market(req.market) if req.market else None
        adjust = Adjust(req.adjust)
        registry = default_registry(cache=cache)
        # 将 str 转为 date，避免底层 date vs str 比较错误
        start_date = date.fromisoformat(req.start)
        end_date = date.fromisoformat(req.end)
        results = []
        for symbol in req.symbols:
            md = registry.get_ohlcv(
                symbol, start_date, end_date, market=market, adjust=adjust
            )
            results.append(
                {"symbol": symbol, "rows": len(md), "start": req.start, "end": req.end}
            )
        return {"status": "ok", "fetched": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/cache")
async def list_cache(cache: DataCache = Depends(get_cache)):
    """列出缓存条目。"""
    return CacheResponse(entries=cache.list_entries())


@router.delete("/cache")
async def clear_cache(cache: DataCache = Depends(get_cache)):
    """清空缓存。"""
    cache.clear()
    return {"status": "cleared"}

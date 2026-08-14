"""数据路由:拉取/缓存/基准。"""

import asyncio
import math
from datetime import date, datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from djinn.api.deps import get_cache
from djinn.api.schemas import CacheEntry, CacheResponse, DataFetchRequest
from djinn.data import DataCache, default_registry
from djinn.data.schema import Adjust, Market

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/fetch")
async def fetch_data(
    req: DataFetchRequest, cache: DataCache = Depends(get_cache)
) -> dict[str, Any]:
    """拉取标的 OHLCV 数据。"""
    try:
        market = Market(req.market) if req.market else None
        adjust = Adjust(req.adjust)
        registry = default_registry(cache=cache)
        # 将 str 转为 date，避免底层 date vs str 比较错误
        start_date = date.fromisoformat(req.start)
        end_date = date.fromisoformat(req.end)

        def _fetch_one(symbol: str) -> dict[str, Any]:
            md = registry.get_ohlcv(
                symbol, start_date, end_date, market=market, adjust=adjust
            )
            return {
                "symbol": symbol,
                "rows": len(md),
                "start": req.start,
                "end": req.end,
            }

        # 阻塞的网络拉取放到线程池,避免阻塞事件循环;并发同键受 E1 单飞锁约束
        results = await asyncio.gather(
            *(asyncio.to_thread(_fetch_one, s) for s in req.symbols)
        )
        return {"status": "ok", "fetched": list(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _entry_from(e: dict[str, object]) -> CacheEntry:
    rows_raw = e.get("rows", 0)
    return CacheEntry(
        file=str(e.get("file", "")),
        rows=rows_raw if isinstance(rows_raw, int) else 0,
        start=str(e["start"]) if e.get("start") is not None else None,
        end=str(e["end"]) if e.get("end") is not None else None,
        error=bool(e.get("error", False)),
    )


@router.get("/cache", response_model=CacheResponse)
async def list_cache(cache: DataCache = Depends(get_cache)) -> CacheResponse:
    """列出缓存条目。"""
    return CacheResponse(entries=[_entry_from(e) for e in cache.list_entries()])


@router.delete("/cache")
async def clear_cache(cache: DataCache = Depends(get_cache)) -> dict[str, str]:
    """清空缓存。"""
    cache.clear()
    return {"status": "cleared"}


def _safe_scalar(v: Any) -> Any:
    """把缓存单元格转成 JSON 友好值(NaN/Inf/NaT → None,时间 → ISO 字符串)。"""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (bool, int, float)):
        return v
    return str(v)


def _preview(
    df: pd.DataFrame, file: str, head: int = 5, tail: int = 5
) -> dict[str, Any]:
    """缓存文件结构 + 首尾内容预览(JSON 友好,index 并入每行)。"""
    is_dt = isinstance(df.index, pd.DatetimeIndex)

    def _rows(sub: pd.DataFrame) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx_val, row in sub.iterrows():
            key = str(idx_val)
            if is_dt:
                key = key[:10]  # Timestamp 的 str 为 ISO 格式,取前 10 位即日期
            d: dict[str, Any] = {"_index": key}
            for col in sub.columns:
                d[str(col)] = _safe_scalar(row[col])
            out.append(d)
        return out

    return {
        "file": file,
        "rows": len(df),
        "index_type": "datetime" if is_dt else str(df.index.dtype),
        "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
        "head": _rows(df.head(head)),
        "tail": _rows(df.tail(tail)),
    }


@router.get("/cache/content")
async def cache_content(
    file: str, cache: DataCache = Depends(get_cache)
) -> dict[str, Any]:
    """查看缓存文件的字段(列名/类型)与内容预览(首尾各 5 行)。"""
    df = cache.inspect(file)
    if df is None:
        raise HTTPException(status_code=404, detail=f"缓存文件 {file} 不存在")
    return _preview(df, file)

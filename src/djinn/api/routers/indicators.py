"""指标路由:指标库 schema(GET /indicators)+ 用户自定义指标 CRUD。"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from djinn.api.deps import get_indicator_store
from djinn.api.schemas import (
    IndicatorInfo,
    IndicatorListResponse,
    UserIndicatorCreate,
    UserIndicatorResponse,
    UserIndicatorUpdate,
    UserIndicatorValidateResponse,
)
from djinn.indicators import __all__ as BUILTIN_INDICATORS
from djinn.indicators import indicator_specs
from djinn.indicators.store import IndicatorStore, UserIndicatorRecord
from djinn.indicators.user import compile_user_indicator
from djinn.utils.exceptions import StrategyError

router = APIRouter(prefix="/indicators", tags=["indicators"])


def _fmt_signature(name: str, func: Any) -> str:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return name
    parts: list[str] = []
    for p in sig.parameters.values():
        s = p.name
        if p.default is not inspect.Parameter.empty:
            s += f"={p.default!r}"
        parts.append(s)
    return f"{name}({', '.join(parts)})"


def _builtin_info() -> list[IndicatorInfo]:
    return [
        IndicatorInfo(**{**spec, "origin": "builtin"}) for spec in indicator_specs()
    ]


def _user_info(rec: UserIndicatorRecord) -> IndicatorInfo:
    desc = rec.description
    sig = ""
    try:
        func = compile_user_indicator(rec.name, rec.source_code)
        sig = _fmt_signature(rec.name, func)
        if not desc:
            doc = (func.__doc__ or "").strip().splitlines()
            desc = doc[0].strip() if doc else ""
    except StrategyError:
        pass
    return IndicatorInfo(
        name=rec.name,
        category="自定义",
        description=desc,
        doc=desc,
        signature=sig,
        params=[],
        source=rec.source_code,
        origin="user",
    )


def _to_response(rec: UserIndicatorRecord) -> UserIndicatorResponse:
    sig = ""
    try:
        func = compile_user_indicator(rec.name, rec.source_code)
        sig = _fmt_signature(rec.name, func)
    except StrategyError:
        pass
    return UserIndicatorResponse(
        indicator_id=rec.indicator_id,
        name=rec.name,
        source_code=rec.source_code,
        description=rec.description,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        signature=sig,
    )


@router.get("", response_model=IndicatorListResponse)
async def list_indicators(
    store: IndicatorStore = Depends(get_indicator_store),
) -> IndicatorListResponse:
    """列出全部指标(内置 + 用户自定义)及签名/源码。"""
    items = _builtin_info()
    # E2:用户指标编译(exec 用户代码)卸载到线程
    user_items = await asyncio.to_thread(
        lambda: [_user_info(r) for r in store.list_indicators()]
    )
    items += user_items
    return IndicatorListResponse(indicators=items)


@router.get("/user", response_model=list[UserIndicatorResponse])
async def list_user_indicators(
    store: IndicatorStore = Depends(get_indicator_store),
) -> list[UserIndicatorResponse]:
    """列出全部用户自定义指标(含源码)。"""
    # E2:编译用户代码卸载到线程
    return await asyncio.to_thread(
        lambda: [_to_response(r) for r in store.list_indicators()]
    )


@router.post("/user", response_model=UserIndicatorResponse, status_code=201)
async def create_user_indicator(
    req: UserIndicatorCreate,
    store: IndicatorStore = Depends(get_indicator_store),
) -> UserIndicatorResponse:
    """创建用户指标(先编译校验再落库)。"""
    if req.name in BUILTIN_INDICATORS:
        raise HTTPException(status_code=409, detail=f"名称 {req.name!r} 与内置指标冲突")
    try:
        # E2:exec 用户代码卸载到线程
        await asyncio.to_thread(compile_user_indicator, req.name, req.source_code)
    except StrategyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        rec = store.create(req.name, req.source_code, req.description)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _to_response(rec)


@router.post("/user/validate", response_model=UserIndicatorValidateResponse)
async def validate_user_indicator(
    req: UserIndicatorCreate,
) -> UserIndicatorValidateResponse:
    """仅编译校验(不落库),返回签名或错误。"""
    try:
        # E2:exec 用户代码卸载到线程
        func = await asyncio.to_thread(
            compile_user_indicator, req.name, req.source_code
        )
        return UserIndicatorValidateResponse(
            valid=True, signature=_fmt_signature(req.name, func)
        )
    except StrategyError as e:
        return UserIndicatorValidateResponse(valid=False, error=str(e))


@router.put("/user/{indicator_id}", response_model=UserIndicatorResponse)
async def update_user_indicator(
    indicator_id: str,
    req: UserIndicatorUpdate,
    store: IndicatorStore = Depends(get_indicator_store),
) -> UserIndicatorResponse:
    """更新用户指标(编译校验通过才落库)。"""
    existing = store.get(indicator_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"指标 {indicator_id} 不存在")
    new_name = req.name if req.name is not None else existing.name
    new_source = (
        req.source_code if req.source_code is not None else existing.source_code
    )
    if new_name in BUILTIN_INDICATORS:
        raise HTTPException(status_code=409, detail=f"名称 {new_name!r} 与内置指标冲突")
    try:
        # E2:exec 用户代码卸载到线程
        await asyncio.to_thread(compile_user_indicator, new_name, new_source)
    except StrategyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        rec = store.update(
            indicator_id,
            name=req.name,
            source_code=req.source_code,
            description=req.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if rec is None:
        raise HTTPException(status_code=404, detail=f"指标 {indicator_id} 不存在")
    return _to_response(rec)


@router.delete("/user/{indicator_id}", status_code=204)
async def delete_user_indicator(
    indicator_id: str,
    store: IndicatorStore = Depends(get_indicator_store),
) -> None:
    """删除用户指标。"""
    if not store.delete(indicator_id):
        raise HTTPException(status_code=404, detail=f"指标 {indicator_id} 不存在")


@router.get("/{name}", response_model=IndicatorInfo)
async def get_indicator(
    name: str, store: IndicatorStore = Depends(get_indicator_store)
) -> IndicatorInfo:
    """获取单个指标详情(内置或用户自定义)。"""
    if name in BUILTIN_INDICATORS:
        spec = next(s for s in indicator_specs() if s["name"] == name)
        return IndicatorInfo(**{**spec, "origin": "builtin"})
    rec = store.get_by_name(name)
    if rec is not None:
        return _user_info(rec)
    raise HTTPException(status_code=404, detail=f"指标 {name} 不存在")

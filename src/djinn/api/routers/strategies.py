"""策略路由:列表/schema + 用户自定义策略 CRUD。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from djinn.api.deps import get_strategy_store
from djinn.api.schemas import (
    StrategyInfo,
    StrategyListResponse,
    UserStrategyCreate,
    UserStrategyResponse,
    UserStrategyUpdate,
    UserStrategyValidateResponse,
)
from djinn.strategy.library import STRATEGY_REGISTRY
from djinn.strategy.parameter import param_schema
from djinn.strategy.store import (
    KIND_PINE,
    KIND_PYTHON,
    StrategyStore,
    UserStrategyRecord,
)
from djinn.strategy.user import compile_user_strategy
from djinn.utils.exceptions import StrategyError

router = APIRouter(prefix="/strategies", tags=["strategies"])

_KINDS = {KIND_PYTHON, KIND_PINE}


def _doc(cls: type) -> str:
    doc = (cls.__doc__ or "").strip().splitlines()
    return doc[0].strip() if doc else ""


def _params(cls: type) -> list[dict[str, Any]]:
    return [p.to_dict() for p in param_schema(cls)]


def _builtin_info(name: str) -> StrategyInfo:
    return StrategyInfo(
        name=name,
        description=_doc(STRATEGY_REGISTRY[name]),
        params=_params(STRATEGY_REGISTRY[name]),
    )


def _compiled(rec: UserStrategyRecord) -> tuple[list[dict[str, Any]], str]:
    """编译用户策略,返回 (params, error);失败时 params 为空。"""
    try:
        cls = compile_user_strategy(rec.name, rec.source_code, rec.kind)
        return _params(cls), ""
    except StrategyError as e:
        return [], str(e)


def _user_info(rec: UserStrategyRecord) -> StrategyInfo:
    params, _ = _compiled(rec)
    return StrategyInfo(
        name=rec.name, description=rec.description or rec.kind, params=params
    )


def _to_response(rec: UserStrategyRecord) -> UserStrategyResponse:
    params, _ = _compiled(rec)
    return UserStrategyResponse(
        strategy_id=rec.strategy_id,
        name=rec.name,
        kind=rec.kind,
        source_code=rec.source_code,
        description=rec.description,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        params=params,
    )


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise HTTPException(
            status_code=400, detail=f"未知 kind {kind!r},仅支持 {sorted(_KINDS)}"
        )


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    store: StrategyStore = Depends(get_strategy_store),
) -> StrategyListResponse:
    """列出全部策略(内置 + 用户自定义)及参数 schema。"""
    items = [_builtin_info(n) for n in STRATEGY_REGISTRY]
    items += [_user_info(r) for r in store.list_strategies()]
    return StrategyListResponse(strategies=items)


@router.get("/user", response_model=list[UserStrategyResponse])
async def list_user_strategies(
    store: StrategyStore = Depends(get_strategy_store),
) -> list[UserStrategyResponse]:
    """列出全部用户自定义策略(含源码)。"""
    return [_to_response(r) for r in store.list_strategies()]


@router.post("/user", response_model=UserStrategyResponse, status_code=201)
async def create_user_strategy(
    req: UserStrategyCreate,
    store: StrategyStore = Depends(get_strategy_store),
) -> UserStrategyResponse:
    """创建用户自定义策略(先编译校验再落库)。"""
    _check_kind(req.kind)
    if req.name in STRATEGY_REGISTRY:
        raise HTTPException(status_code=409, detail=f"名称 {req.name!r} 与内置策略冲突")
    try:
        compile_user_strategy(req.name, req.source_code, req.kind)
    except StrategyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        rec = store.create(
            req.name, req.source_code, kind=req.kind, description=req.description
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _to_response(rec)


@router.post("/user/validate", response_model=UserStrategyValidateResponse)
async def validate_user_strategy(
    req: UserStrategyCreate,
) -> UserStrategyValidateResponse:
    """仅编译校验(不落库),返回参数 schema 或错误。"""
    _check_kind(req.kind)
    try:
        cls = compile_user_strategy(req.name, req.source_code, req.kind)
        return UserStrategyValidateResponse(valid=True, params=_params(cls))
    except StrategyError as e:
        return UserStrategyValidateResponse(valid=False, error=str(e))


@router.put("/user/{strategy_id}", response_model=UserStrategyResponse)
async def update_user_strategy(
    strategy_id: str,
    req: UserStrategyUpdate,
    store: StrategyStore = Depends(get_strategy_store),
) -> UserStrategyResponse:
    """更新用户策略(编译校验通过才落库)。"""
    existing = store.get(strategy_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")
    new_name = req.name if req.name is not None else existing.name
    new_kind = req.kind if req.kind is not None else existing.kind
    new_source = (
        req.source_code if req.source_code is not None else existing.source_code
    )
    _check_kind(new_kind)
    if new_name in STRATEGY_REGISTRY:
        raise HTTPException(status_code=409, detail=f"名称 {new_name!r} 与内置策略冲突")
    try:
        compile_user_strategy(new_name, new_source, new_kind)
    except StrategyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        rec = store.update(
            strategy_id,
            name=req.name,
            source_code=req.source_code,
            kind=req.kind,
            description=req.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if rec is None:
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")
    return _to_response(rec)


@router.delete("/user/{strategy_id}", status_code=204)
async def delete_user_strategy(
    strategy_id: str,
    store: StrategyStore = Depends(get_strategy_store),
) -> None:
    """删除用户策略。"""
    if not store.delete(strategy_id):
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")


@router.get("/{name}", response_model=StrategyInfo)
async def get_strategy(
    name: str, store: StrategyStore = Depends(get_strategy_store)
) -> StrategyInfo:
    """获取单个策略详情(内置或用户自定义)。"""
    if name in STRATEGY_REGISTRY:
        return _builtin_info(name)
    rec = store.get_by_name(name)
    if rec is not None:
        return _user_info(rec)
    raise HTTPException(status_code=404, detail=f"策略 {name} 不存在")

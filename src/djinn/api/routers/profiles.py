"""标的 profile 路由:命名标的列表的增删改查。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from djinn.api.deps import get_profile_registry
from djinn.api.profiles import ProfileRegistry
from djinn.api.schemas import ProfileCreate, ProfileResponse, ProfileUpdate

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("", response_model=list[ProfileResponse])
async def list_profiles(
    registry: ProfileRegistry = Depends(get_profile_registry),
) -> list[ProfileResponse]:
    """列出全部 profile(按名称升序)。"""
    return [ProfileResponse(**p.to_dict()) for p in registry.list_profiles()]


@router.post("", response_model=ProfileResponse, status_code=201)
async def create_profile(
    req: ProfileCreate,
    registry: ProfileRegistry = Depends(get_profile_registry),
) -> ProfileResponse:
    """创建 profile;名称重复返回 409。"""
    try:
        rec = registry.create(name=req.name, symbols=req.symbols, market=req.market)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return ProfileResponse(**rec.to_dict())


@router.get("/{profile_id}", response_model=ProfileResponse)
async def get_profile(
    profile_id: str,
    registry: ProfileRegistry = Depends(get_profile_registry),
) -> ProfileResponse:
    """按 id 查询单个 profile。"""
    rec = registry.get(profile_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"profile {profile_id} 不存在")
    return ProfileResponse(**rec.to_dict())


@router.put("/{profile_id}", response_model=ProfileResponse)
async def update_profile(
    profile_id: str,
    req: ProfileUpdate,
    registry: ProfileRegistry = Depends(get_profile_registry),
) -> ProfileResponse:
    """更新 profile(仅更新请求中非 None 字段)。"""
    try:
        rec = registry.update(
            profile_id, name=req.name, symbols=req.symbols, market=req.market
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if rec is None:
        raise HTTPException(status_code=404, detail=f"profile {profile_id} 不存在")
    return ProfileResponse(**rec.to_dict())


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(
    profile_id: str,
    registry: ProfileRegistry = Depends(get_profile_registry),
) -> None:
    """删除 profile。"""
    if not registry.delete(profile_id):
        raise HTTPException(status_code=404, detail=f"profile {profile_id} 不存在")

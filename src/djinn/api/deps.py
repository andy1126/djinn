"""依赖注入:共享单例(JobRegistry / DataCache / ProviderRegistry)。"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from djinn.api.jobs import JobRegistry
from djinn.data import DataCache, default_registry
from djinn.data.provider import ProviderRegistry


@lru_cache(maxsize=1)
def get_job_registry() -> JobRegistry:
    return JobRegistry()


@lru_cache(maxsize=1)
def get_cache() -> DataCache:
    return DataCache()


def get_registry(cache: DataCache = Depends(get_cache)) -> ProviderRegistry:
    """构建默认 provider 注册表(共享缓存;测试可 override 注入 stub)。"""
    return default_registry(cache=cache)

"""依赖注入:共享单例(JobRegistry / DataCache / ProviderRegistry)。"""

from __future__ import annotations

import threading
from functools import lru_cache

from fastapi import Depends

from djinn.api.jobs import JobRegistry, JobScheduler
from djinn.api.profiles import ProfileRegistry
from djinn.data import DataCache, default_registry
from djinn.data.provider import ProviderRegistry
from djinn.indicators.store import get_indicator_store as get_indicator_store
from djinn.strategy.store import get_strategy_store as get_strategy_store


@lru_cache(maxsize=1)
def get_job_registry() -> JobRegistry:
    return JobRegistry()


@lru_cache(maxsize=1)
def get_profile_registry() -> ProfileRegistry:
    return ProfileRegistry()


@lru_cache(maxsize=1)
def get_cache() -> DataCache:
    return DataCache()


@lru_cache(maxsize=1)
def get_scheduler() -> JobScheduler:
    """任务调度器单例(并发上限 + FIFO 排队,E5)。"""
    return JobScheduler(get_job_registry())


# provider 注册表单例:跨请求共享 provider 实例,保留其限速状态(_last_request),
# 否则每个请求新建 provider → akshare/yahoo 跨请求限速形同虚设(E3)。
_registry_lock = threading.Lock()
_REGISTRY: ProviderRegistry | None = None


def get_registry(cache: DataCache = Depends(get_cache)) -> ProviderRegistry:
    """返回默认 provider 注册表单例(共享缓存;测试可 override 注入 stub)。"""
    global _REGISTRY
    with _registry_lock:
        if _REGISTRY is None:
            _REGISTRY = default_registry(cache=cache)
        return _REGISTRY

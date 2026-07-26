"""依赖注入:共享单例(JobRegistry / DataCache / ProviderRegistry)。"""

from __future__ import annotations

from functools import lru_cache

from djinn.api.jobs import JobRegistry
from djinn.data import DataCache


@lru_cache(maxsize=1)
def get_job_registry() -> JobRegistry:
    return JobRegistry()


@lru_cache(maxsize=1)
def get_cache() -> DataCache:
    return DataCache()

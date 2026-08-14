"""E2/E3 测试:registry 单例 + provider 限速线程安全。"""

from __future__ import annotations

import threading

from djinn.api.deps import get_registry
from djinn.data import DataCache


def test_registry_singleton(tmp_path) -> None:
    """两次调用 get_registry 返回同一 provider 注册表(跨请求共享限速状态)。"""
    cache = DataCache(cache_dir=tmp_path)
    r1 = get_registry(cache)
    r2 = get_registry(cache)
    assert r1 is r2


def test_throttle_threadsafe() -> None:
    """并发调用 provider._throttle 无异常(限速临界区加锁)。"""
    from djinn.data.providers.akshare import AkShareProvider

    p = AkShareProvider(rate_limit_sec=0.001)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(20):
                p._throttle()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors

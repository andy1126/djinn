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


def test_search_does_not_block_health(monkeypatch) -> None:
    """E2:慢搜索(to_thread 卸载)不阻塞 /health —— 防事件循环被网络调用占住。"""
    import asyncio
    import time

    import httpx

    from djinn.api import deps
    from djinn.api.main import app
    from djinn.data import ProviderRegistry

    class _SlowProvider:
        name = "slow"

        def supports(self, symbol, market=None):
            return True

        def search_symbols(self, q, market=None):
            time.sleep(0.3)  # 线程内阻塞(不应占事件循环)
            return [("AAPL", "Apple Inc.")]

    # 注入只含慢 provider 的注册表(覆盖 E3 单例)
    monkeypatch.setattr(deps, "_REGISTRY", ProviderRegistry([_SlowProvider()]))

    async def run() -> float:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            a = asyncio.create_task(client.get("/stocks/search", params={"q": "aapl"}))
            await asyncio.sleep(0)  # 让 A 进入请求处理(在 to_thread 线程中 sleep)
            t0 = time.time()
            b = await client.get("/health")
            latency = time.time() - t0
            assert b.status_code == 200
            await a  # A 300ms 后完成
            return latency

    latency = asyncio.run(run())
    assert latency < 0.15, f"/health 被慢搜索阻塞了 {latency:.2f}s(应 <150ms)"

"""DataCache 线程安全并发测试(E1:锁 + 原子写 + 单飞)。"""

from __future__ import annotations

import threading
import time
from datetime import date

import pandas as pd

from djinn.data.cache import DataCache
from djinn.data.market_data import MarketData
from djinn.data.provider import DataProvider, ProviderRegistry
from djinn.data.schema import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_VOLUME,
    Adjust,
    Market,
)


def _frame(n: int, value: float) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            COL_OPEN: [value] * n,
            COL_HIGH: [value] * n,
            COL_LOW: [value] * n,
            COL_CLOSE: [value] * n,
            COL_VOLUME: [1.0e6] * n,
        },
        index=idx,
    )


def test_concurrent_put_get(tmp_path) -> None:
    """8 线程 × 混合 put/get/merge 随机键,无异常。"""
    cache = DataCache(cache_dir=tmp_path)
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        try:
            for i in range(200):
                sym = f"S{(tid * 7 + i) % 5}"
                df = _frame(10, float(i))
                if i % 3 == 0:
                    cache.put("p", sym, Adjust.BACKWARD, df)
                elif i % 3 == 1:
                    cache.get("p", sym, Adjust.BACKWARD)
                else:
                    cache.merge("p", sym, Adjust.BACKWARD, df)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_concurrent_merge_no_segment_loss(tmp_path) -> None:
    """两线程并发 merge 同一键的不同区间,结果覆盖完整区间(不丢段)。"""
    cache = DataCache(cache_dir=tmp_path)

    def merge_segment(start_day: int, end_day: int) -> None:
        idx = pd.date_range(date(2024, 1, start_day), periods=end_day - start_day + 1)
        df = pd.DataFrame({"close": [1.0] * len(idx)}, index=idx)
        cache.merge("p", "S", Adjust.BACKWARD, df)

    t1 = threading.Thread(target=merge_segment, args=(1, 5))
    t2 = threading.Thread(target=merge_segment, args=(6, 10))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    merged = cache.get("p", "S", Adjust.BACKWARD)
    assert merged is not None
    assert merged.index.min().date() == date(2024, 1, 1)
    assert merged.index.max().date() == date(2024, 1, 10)
    assert len(merged) == 10


def test_atomic_write_no_partial(tmp_path) -> None:
    """写线程 + 读线程并发,读端永不读到截断 parquet。"""
    cache = DataCache(cache_dir=tmp_path)
    key = cache.make_key("p", "S", Adjust.BACKWARD, dtype="quote")
    path = cache._parquet_path(key)
    n = 100
    stop = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        for i in range(300):
            cache.put("p", "S", Adjust.BACKWARD, _frame(n, float(i)))

    def reader() -> None:
        while not stop.is_set():
            try:
                df = pd.read_parquet(path)
                assert len(df) == n, f"截断读取: {len(df)}"
            except FileNotFoundError:
                pass  # 尚未写入,合法
            except Exception as e:  # pragma: no cover
                errors.append(e)
                break

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    stop.set()
    t2.join()
    assert not errors, errors


def test_singleflight(tmp_path) -> None:
    """同键并发拉取 → 底层网络函数仅调用一次(单飞)。"""
    cache = DataCache(cache_dir=tmp_path)

    class _CountingProvider(DataProvider):
        name = "counting"
        market = Market.US

        def __init__(self) -> None:
            self.fetch_count = 0
            self._count_lock = threading.Lock()

        def supports(self, symbol: str, market: Market | None = None) -> bool:
            return True

        def get_ohlcv(
            self, symbol: str, start: date, end: date, adjust: Adjust = Adjust.BACKWARD
        ) -> MarketData:
            cached = self.cache.get(self.name, symbol, adjust)
            if DataCache.covers_soft(cached, start, end):
                assert cached is not None
                df = cached.loc[pd.Timestamp(start) : pd.Timestamp(end)]
            else:
                with self._count_lock:
                    self.fetch_count += 1
                time.sleep(0.05)  # 放大竞态窗口
                idx = pd.bdate_range(start, end)
                n = len(idx)
                new = pd.DataFrame(
                    {
                        COL_OPEN: [10.0] * n,
                        COL_HIGH: [10.0] * n,
                        COL_LOW: [10.0] * n,
                        COL_CLOSE: [10.0] * n,
                        COL_VOLUME: [1.0e6] * n,
                    },
                    index=idx,
                )
                df = self.cache.merge(self.name, symbol, adjust, new)
                df = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
            return MarketData(symbol=symbol, market=Market.US, df=df, adjust=adjust)

    provider = _CountingProvider()
    provider.cache = cache
    registry = ProviderRegistry([provider])
    start = date(2024, 1, 1)
    end = date(2024, 1, 31)
    barrier = threading.Barrier(2)

    def run() -> None:
        barrier.wait()
        registry.get_ohlcv("AAPL", start, end)

    t1 = threading.Thread(target=run)
    t2 = threading.Thread(target=run)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert provider.fetch_count == 1

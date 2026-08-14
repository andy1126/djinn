"""DataCache 覆盖 / LRU / TTL 单测(D6:LRU 可调 + covers 修复 + TTL)。"""

from __future__ import annotations

import os
import time
from datetime import date

import pandas as pd

from djinn.data.cache import DataCache
from djinn.data.schema import Adjust


def _frame(n: int, start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, periods=n)
    return pd.DataFrame({"close": [1.0] * n}, index=idx)


# ── covers / covers_soft ──────────────────────────────
def test_covers_weekend_hit() -> None:
    """缓存末日周五、请求 end 周日 → covers_soft 命中(软命中)。"""
    df = _frame(5, "2024-01-01", "2024-01-05")  # Mon–Fri
    assert DataCache.covers_soft(df, date(2024, 1, 1), date(2024, 1, 7)) is True


def test_covers_gap_miss() -> None:
    """缓存末日距 end 30 天 → 软命中也 miss(需增量拉取)。"""
    df = _frame(5, "2024-01-01", "2024-01-05")
    assert DataCache.covers_soft(df, date(2024, 1, 1), date(2024, 2, 5)) is False


def test_covers_clamps_future_end() -> None:
    """end 晚于今天(未来)→ 按今天截断,已有数据到最近交易日则命中。"""
    df = _frame(5, "2024-01-01", "2024-01-05")
    today = date(2024, 1, 4)  # 固定 today,避免依赖真实日期
    assert DataCache.covers(df, date(2024, 1, 1), date(2024, 1, 10), today=today)


def test_covers_empty_miss() -> None:
    assert DataCache.covers(None, date(2024, 1, 1), date(2024, 1, 5)) is False
    assert (
        DataCache.covers_soft(pd.DataFrame(), date(2024, 1, 1), date(2024, 1, 5))
        is False
    )


# ── LRU 容量 ──────────────────────────────────────────
def test_lru_size_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DJINN_CACHE_MEM_SIZE", "2")
    cache = DataCache(cache_dir=tmp_path)
    assert cache._mem_size == 2
    for i in range(3):
        df = _frame(1, "2024-01-01", "2024-01-01")
        cache.put("p", f"S{i}", Adjust.BACKWARD, df)
    assert len(cache._mem) <= 2
    key0 = cache.make_key("p", "S0", Adjust.BACKWARD, dtype="quote")
    assert cache._mem_get(key0) is None  # 最旧的已被逐出内存


def test_lru_size_param(tmp_path) -> None:
    cache = DataCache(cache_dir=tmp_path, mem_size=1)
    assert cache._mem_size == 1
    cache2 = DataCache(cache_dir=tmp_path, mem_size=0)  # 下限夹到 1
    assert cache2._mem_size == 1


def test_lru_size_env_invalid(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DJINN_CACHE_MEM_SIZE", "not-a-number")
    cache = DataCache(cache_dir=tmp_path)
    assert cache._mem_size == 128  # 回退默认


# ── universe TTL ──────────────────────────────────────
def test_universe_ttl(tmp_path) -> None:
    cache = DataCache(cache_dir=tmp_path)
    cache.put_universe("p", "idx", pd.DataFrame({"symbol": ["A"], "name": ["a"]}))
    assert cache.get_universe("p", "idx") is not None
    path = cache._parquet_path(cache.make_key("p", "idx", dtype="universe"))
    old = time.time() - 8 * 86400
    os.utime(path, (old, old))
    assert cache.get_universe("p", "idx", max_age_days=7) is None

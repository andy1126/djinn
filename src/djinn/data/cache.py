"""两级缓存:内存 LRU + 磁盘 Parquet(线程安全 + 原子写)。

缓存键 ``{provider}::{dtype}::{symbol_or_index}::{adjust}``:

- ``dtype`` 区分数据类型:``quote``(行情)/ ``fundamental``(基本面)/ ``universe``(股票池);
- 行情按 ``(provider, symbol, adjust)`` 命中,范围不足时由 provider 增量拉取并合并;
- 基本面 / 股票池不做区间合并,整帧覆盖写入(快照 / 低频数据)。

线程安全:内存 LRU 用实例级 ``RLock``;磁盘层用**按键锁**(``_key_lock``),配合
原子写(tmp + ``os.replace``)保证并发读不读到截断文件、并发同键 merge 不丢区间。
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd

from djinn.data.schema import Adjust
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

DEFAULT_CACHE_DIR: Final[str] = ".cache/djinn"
_DEFAULT_MEM_SIZE: Final[int] = 128


class DataCache:
    """两级缓存:内存 LRU + 磁盘 Parquet(线程安全)。"""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        mem_size: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._mem_size = self._resolve_mem_size(mem_size)
        # 内存 LRU 锁:所有 _mem 的读写(move_to_end/popitem/赋值/删除)。
        self._mem_lock = threading.RLock()
        # 磁盘按键锁:同键的"读旧帧→合并→写回"串行化(即"单飞"的简化实现)。
        self._file_locks: dict[str, threading.RLock] = {}
        self._file_locks_guard = threading.Lock()

    @staticmethod
    def _resolve_mem_size(mem_size: int | None) -> int:
        if mem_size is not None:
            return max(1, int(mem_size))
        raw = os.environ.get("DJINN_CACHE_MEM_SIZE")
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                _log.warning(
                    "DJINN_CACHE_MEM_SIZE 非法 %r,回退默认 %d", raw, _DEFAULT_MEM_SIZE
                )
        return _DEFAULT_MEM_SIZE

    def _key_lock(self, key: str) -> threading.RLock:
        with self._file_locks_guard:
            return self._file_locks.setdefault(key, threading.RLock())

    # ── 键 ──────────────────────────────────────────────
    @staticmethod
    def make_key(
        provider: str,
        symbol: str,
        adjust: Adjust = Adjust.NONE,
        dtype: str = "quote",
    ) -> str:
        """构造缓存键 ``{provider}::{dtype}::{symbol}::{adjust}``。

        行情(dtype=quote)需带 ``adjust``;基本面 / 股票池(dtype=fundamental/universe)
        与复权无关,统一用 ``Adjust.NONE`` 占位。
        """
        return f"{provider}::{dtype}::{symbol}::{adjust.value}"

    def _parquet_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}.parquet"

    # ── 内存 LRU(受 _mem_lock 保护)──────────────────────
    def _mem_get(self, key: str) -> pd.DataFrame | None:
        with self._mem_lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                return self._mem[key]
        return None

    def _put_mem(self, key: str, df: pd.DataFrame) -> None:
        with self._mem_lock:
            self._mem[key] = df
            self._mem.move_to_end(key)
            while len(self._mem) > self._mem_size:
                self._mem.popitem(last=False)

    def _mem_pop(self, key: str) -> None:
        with self._mem_lock:
            self._mem.pop(key, None)

    # ── 磁盘读写(按键锁 + 原子写)────────────────────────
    def _atomic_write_parquet(self, df: pd.DataFrame, path: Path) -> None:
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            df.to_parquet(tmp)
            os.replace(tmp, path)  # POSIX 原子 rename:读端永不看到截断文件
        except Exception as e:
            _log.warning("缓存写入失败 %s: %s", path, e)
            with contextlib.suppress(OSError):
                tmp.unlink()

    def _read_parquet(self, key: str, *, datetime_index: bool) -> pd.DataFrame | None:
        path = self._parquet_path(key)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if datetime_index:
                df.index = pd.to_datetime(df.index)
            self._put_mem(key, df)
            return df
        except Exception as e:
            _log.warning("缓存读取失败 %s: %s", path, e)
            return None

    def _write(self, key: str, df: pd.DataFrame) -> None:
        self._put_mem(key, df)
        self._atomic_write_parquet(df, self._parquet_path(key))

    # ── 行情 ────────────────────────────────────────────
    def get(self, provider: str, symbol: str, adjust: Adjust) -> pd.DataFrame | None:
        """返回完整行情缓存(不按区间截断);无则 None。"""
        key = self.make_key(provider, symbol, adjust, dtype="quote")
        df = self._mem_get(key)
        if df is not None:
            return df
        with self._key_lock(key):
            df = self._mem_get(key)  # 等待键锁期间其他线程可能已写入
            if df is not None:
                return df
            return self._read_parquet(key, datetime_index=True)

    def put(self, provider: str, symbol: str, adjust: Adjust, df: pd.DataFrame) -> None:
        key = self.make_key(provider, symbol, adjust, dtype="quote")
        with self._key_lock(key):
            self._write(key, df)

    # ── 基本面 / 股票池(整帧,不做区间合并)──────────────
    def put_fundamentals(self, provider: str, symbol: str, df: pd.DataFrame) -> None:
        key = self.make_key(provider, symbol, dtype="fundamental")
        with self._key_lock(key):
            self._write(key, df)

    def get_fundamentals(
        self, provider: str, symbol: str, max_age_days: float | None = None
    ) -> pd.DataFrame | None:
        """读基本面缓存(整帧);``max_age_days`` 给定时超龄视为 miss(D6)。

        与 ``get_universe`` 同语义:用磁盘 parquet mtime 判龄,超龄丢弃内存
        拷贝返回 None 由调用方重拉(基本面 history 建议 30 天)。
        """
        key = self.make_key(provider, symbol, dtype="fundamental")
        with self._key_lock(key):
            if max_age_days is not None:
                path = self._parquet_path(key)
                if path.exists():
                    age_sec = time.time() - path.stat().st_mtime
                    if age_sec > max_age_days * 86400:
                        self._mem_pop(key)
                        return None
            df = self._mem_get(key)
            if df is not None:
                return df
            return self._read_parquet(key, datetime_index=False)

    def put_universe(self, provider: str, name: str, df: pd.DataFrame) -> None:
        key = self.make_key(provider, name, dtype="universe")
        with self._key_lock(key):
            self._write(key, df)

    def get_universe(
        self,
        provider: str,
        name: str,
        max_age_days: float | None = None,
    ) -> pd.DataFrame | None:
        """读股票池缓存(整帧);``max_age_days`` 给定时超龄视为 miss。

        用磁盘 parquet 文件的 mtime 作为写入时间(``_write`` 总先落盘):
        即使内存 LRU 命中,也先检查磁盘 mtime,超龄则丢弃内存拷贝返回
        None,由调用方重拉重写。磁盘不存在(刚写入未落盘)视为新鲜。
        """
        key = self.make_key(provider, name, dtype="universe")
        with self._key_lock(key):
            if max_age_days is not None:
                path = self._parquet_path(key)
                if path.exists():
                    age_sec = time.time() - path.stat().st_mtime
                    if age_sec > max_age_days * 86400:
                        self._mem_pop(key)  # 超龄:丢弃内存拷贝
                        return None
            df = self._mem_get(key)
            if df is not None:
                return df
            return self._read_parquet(key, datetime_index=False)

    # ── 合并 ────────────────────────────────────────────
    def merge(
        self,
        provider: str,
        symbol: str,
        adjust: Adjust,
        new: pd.DataFrame,
    ) -> pd.DataFrame:
        """将新数据与已有缓存合并去重(按索引),写回并返回完整 df。

        "读旧帧 → 合并 → 写回"整体持键锁,并发同 symbol 拉取不互相覆盖。
        """
        key = self.make_key(provider, symbol, adjust, dtype="quote")
        with self._key_lock(key):
            existing = self._mem_get(key)
            if existing is None:
                existing = self._read_parquet(key, datetime_index=True)
            if existing is not None and len(existing):
                combined = pd.concat([existing, new])
                combined = combined[
                    ~combined.index.duplicated(keep="last")
                ].sort_index()
            else:
                combined = new.sort_index()
            self._write(key, combined)
            return combined

    # ── 区间覆盖 ────────────────────────────────────────
    @staticmethod
    def covers(
        df: pd.DataFrame | None,
        start: date,
        end: date,
        *,
        today: date | None = None,
    ) -> bool:
        """缓存是否完整覆盖 [start, end]。

        ``end`` 晚于今天(未来)时按今天截断——未来不可能有数据,否则永远 miss。
        """
        if df is None or len(df) == 0:
            return False
        today = today or date.today()
        effective_end = min(end, today)
        return bool(
            df.index[0].date() <= start and df.index[-1].date() >= effective_end
        )

    @staticmethod
    def covers_soft(
        df: pd.DataFrame | None,
        start: date,
        end: date,
        *,
        slack_days: int = 7,
        today: date | None = None,
    ) -> bool:
        """覆盖或"接近覆盖"(末日距 end ≤ ``slack_days`` 自然日)视为命中。

        用于周末 / 节假日请求 ``end`` 无数据导致的永久 miss:缓存末日(周五)距
        周日 ≤ slack 时视为命中,避免每次重复打网络。A 股长假更长,调用方传
        ``slack_days=12``。
        """
        if df is None or len(df) == 0:
            return False
        today = today or date.today()
        effective_end = min(end, today)
        if df.index[0].date() > start:
            return False
        last = df.index[-1].date()
        if last >= effective_end:
            return True
        return bool((effective_end - last).days <= slack_days)

    def clear(self) -> None:
        self._mem_pop_all()
        # 不主动删磁盘文件,避免误删;提供 list 供 CLI 管理
        for p in self.cache_dir.glob("*.parquet"):
            with contextlib.suppress(OSError):
                p.unlink()

    def _mem_pop_all(self) -> None:
        with self._mem_lock:
            self._mem.clear()

    def list_entries(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for p in self.cache_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(p)
                entry: dict[str, object] = {"file": p.name, "rows": len(df)}
                # E9:仅 quote 帧(index=交易日)解析日期;universe/fundamental 帧
                # index 为 symbol 字符串,强制 to_datetime 会抛异常误报 error。
                if "::quote::" in p.name:
                    idx = pd.to_datetime(df.index)
                    entry["start"] = str(idx.min().date())
                    entry["end"] = str(idx.max().date())
                else:
                    entry["start"] = None
                    entry["end"] = None
                out.append(entry)
            except Exception:
                out.append({"file": p.name, "rows": -1, "error": True})
        return out

    def inspect(self, file: str) -> pd.DataFrame | None:
        """按文件名读取缓存文件(仅允许文件名,防路径穿越);不存在 / 读取失败返回 None。"""
        p = self.cache_dir / file
        if p.parent != self.cache_dir or not p.is_file():
            return None
        try:
            return pd.read_parquet(p)
        except Exception as e:
            _log.warning("缓存读取失败 %s: %s", p, e)
            return None


def env_cache_dir() -> str:
    """从 env 读取缓存目录(默认 ``.cache/djinn``)。"""
    return os.environ.get("DJINN_CACHE_DIR", DEFAULT_CACHE_DIR)

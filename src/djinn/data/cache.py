"""两级缓存:内存 LRU + 磁盘 Parquet。

缓存键 ``{provider}::{dtype}::{symbol_or_index}::{adjust}``:

- ``dtype`` 区分数据类型:``quote``(行情)/ ``fundamental``(基本面)/ ``universe``(股票池);
- 行情按 ``(provider, symbol, adjust)`` 命中,范围不足时由 provider 增量拉取并合并;
- 基本面 / 股票池不做区间合并,整帧覆盖写入(快照 / 低频数据)。
"""

from __future__ import annotations

import contextlib
import os
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
_MEMORY_LRU_SIZE: Final[int] = 32


class DataCache:
    """两级缓存:内存 LRU + 磁盘 Parquet。"""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        memory_size: int = _MEMORY_LRU_SIZE,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._mem_size = memory_size

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

    # ── 读 ──────────────────────────────────────────────
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

    def get(self, provider: str, symbol: str, adjust: Adjust) -> pd.DataFrame | None:
        """返回完整行情缓存(不按区间截断);无则 None。"""
        key = self.make_key(provider, symbol, adjust, dtype="quote")
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        return self._read_parquet(key, datetime_index=True)

    def _put_mem(self, key: str, df: pd.DataFrame) -> None:
        self._mem[key] = df
        self._mem.move_to_end(key)
        while len(self._mem) > self._mem_size:
            self._mem.popitem(last=False)

    def _write(self, key: str, df: pd.DataFrame) -> None:
        self._put_mem(key, df)
        path = self._parquet_path(key)
        try:
            df.to_parquet(path)
        except Exception as e:
            _log.warning("缓存写入失败 %s: %s", path, e)

    # ── 写 ──────────────────────────────────────────────
    def put(self, provider: str, symbol: str, adjust: Adjust, df: pd.DataFrame) -> None:
        key = self.make_key(provider, symbol, adjust, dtype="quote")
        self._write(key, df)

    # ── 基本面 / 股票池(整帧,不做区间合并)──────────────
    def put_fundamentals(self, provider: str, symbol: str, df: pd.DataFrame) -> None:
        self._write(self.make_key(provider, symbol, dtype="fundamental"), df)

    def get_fundamentals(self, provider: str, symbol: str) -> pd.DataFrame | None:
        key = self.make_key(provider, symbol, dtype="fundamental")
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        return self._read_parquet(key, datetime_index=False)

    def put_universe(self, provider: str, name: str, df: pd.DataFrame) -> None:
        self._write(self.make_key(provider, name, dtype="universe"), df)

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
        if max_age_days is not None:
            path = self._parquet_path(key)
            if path.exists():
                age_sec = time.time() - path.stat().st_mtime
                if age_sec > max_age_days * 86400:
                    self._mem.pop(key, None)  # 超龄:丢弃内存拷贝
                    return None
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        return self._read_parquet(key, datetime_index=False)

    # ── 合并 ────────────────────────────────────────────
    def merge(
        self,
        provider: str,
        symbol: str,
        adjust: Adjust,
        new: pd.DataFrame,
    ) -> pd.DataFrame:
        """将新数据与已有缓存合并去重(按索引),写回并返回完整 df。"""
        existing = self.get(provider, symbol, adjust)
        if existing is not None and len(existing):
            combined = pd.concat([existing, new])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new.sort_index()
        self.put(provider, symbol, adjust, combined)
        return combined

    # ── 区间覆盖 ────────────────────────────────────────
    @staticmethod
    def covers(df: pd.DataFrame | None, start: date, end: date) -> bool:
        """缓存是否完整覆盖 [start, end]。"""
        if df is None or len(df) == 0:
            return False
        return (
            df.index[0].date() <= start and df.index[-1].date() >= end and len(df) > 0
        )

    def clear(self) -> None:
        self._mem.clear()
        # 不主动删磁盘文件,避免误删;提供 list 供 CLI 管理
        for p in self.cache_dir.glob("*.parquet"):
            with contextlib.suppress(OSError):
                p.unlink()

    def list_entries(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for p in self.cache_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(p)
                out.append(
                    {
                        "file": p.name,
                        "rows": len(df),
                        "start": str(pd.to_datetime(df.index).min().date()),
                        "end": str(pd.to_datetime(df.index).max().date()),
                    }
                )
            except Exception:
                out.append({"file": p.name, "rows": -1, "error": True})
        return out


def env_cache_dir() -> str:
    """从 env 读取缓存目录(默认 ``.cache/djinn``)。"""
    return os.environ.get("DJINN_CACHE_DIR", DEFAULT_CACHE_DIR)

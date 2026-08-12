"""标的 Profile 存储:SQLite 持久化的命名标的列表。

用户可创建常用的股票/ETF 列表(profile),在回测/组合/数据等页面一键载入。
存储层仿 :class:`djinn.api.jobs.JobRegistry`(原生 sqlite3 + 线程锁 + JSON 列),
symbols 以 JSON 字符串落库。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from djinn.utils.logging import get_logger

_log = get_logger(__name__)

DEFAULT_DB_PATH = ".cache/djinn_profiles.db"


def _clean_symbols(symbols: list[str]) -> list[str]:
    """strip + 去空串 + 保序去重。"""
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        s = raw.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


@dataclass
class ProfileRecord:
    """一条 profile 记录。"""

    profile_id: str
    name: str
    symbols: list[str]
    market: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "symbols": self.symbols,
            "market": self.market,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ProfileRegistry:
    """SQLite profile 注册表(线程安全)。"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    profile_id TEXT PRIMARY KEY,
                    name       TEXT NOT NULL UNIQUE,
                    market     TEXT,
                    symbols    TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            c.commit()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def create(
        self, name: str, symbols: list[str], market: str | None = None
    ) -> ProfileRecord:
        """新建 profile;``name`` 冲突抛 ValueError。"""
        name = name.strip()
        if not name:
            raise ValueError("profile 名称不能为空")
        cleaned = _clean_symbols(symbols)
        if not cleaned:
            raise ValueError("profile 至少需要一个标的")
        rec = ProfileRecord(
            profile_id=uuid.uuid4().hex[:12],
            name=name,
            symbols=cleaned,
            market=market or None,
            created_at=self._now(),
            updated_at=self._now(),
        )
        with self._lock, self._conn() as c:
            try:
                c.execute(
                    "INSERT INTO profiles VALUES (?,?,?,?,?,?)",
                    (
                        rec.profile_id,
                        rec.name,
                        rec.market,
                        json.dumps(rec.symbols),
                        rec.created_at,
                        rec.updated_at,
                    ),
                )
                c.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"profile 名称已存在: {name}") from e
        return rec

    def get(self, profile_id: str) -> ProfileRecord | None:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT * FROM profiles WHERE profile_id=?", (profile_id,)
            ).fetchone()
            return self._row_to_rec(row) if row else None

    def list_profiles(self) -> list[ProfileRecord]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT * FROM profiles ORDER BY name ASC").fetchall()
            return [self._row_to_rec(r) for r in rows]

    def update(
        self,
        profile_id: str,
        *,
        name: str | None = None,
        symbols: list[str] | None = None,
        market: str | None = None,
    ) -> ProfileRecord | None:
        """更新 profile;``name`` 传 None 表示不更新(无法清空 market)。"""
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT * FROM profiles WHERE profile_id=?", (profile_id,)
            ).fetchone()
            if row is None:
                return None
            rec = self._row_to_rec(row)
            if name is not None:
                name = name.strip()
                if not name:
                    raise ValueError("profile 名称不能为空")
                rec.name = name
            if symbols is not None:
                cleaned = _clean_symbols(symbols)
                if not cleaned:
                    raise ValueError("profile 至少需要一个标的")
                rec.symbols = cleaned
            if market is not None:
                rec.market = market or None
            rec.updated_at = self._now()
            try:
                c.execute(
                    "UPDATE profiles SET name=?, market=?, symbols=?, updated_at=? "
                    "WHERE profile_id=?",
                    (
                        rec.name,
                        rec.market,
                        json.dumps(rec.symbols),
                        rec.updated_at,
                        rec.profile_id,
                    ),
                )
                c.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"profile 名称已存在: {rec.name}") from e
        return rec

    def delete(self, profile_id: str) -> bool:
        with self._lock, self._conn() as c:
            cur = c.execute("DELETE FROM profiles WHERE profile_id=?", (profile_id,))
            c.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_rec(row: sqlite3.Row) -> ProfileRecord:
        symbols_raw = row["symbols"]
        symbols = json.loads(symbols_raw) if symbols_raw else []
        return ProfileRecord(
            profile_id=row["profile_id"],
            name=row["name"],
            symbols=symbols if isinstance(symbols, list) else [],
            market=row["market"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

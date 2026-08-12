"""用户自定义策略存储:SQLite 持久化的命名策略源码。

仿 :class:`djinn.api.profiles.ProfileRegistry`(原生 sqlite3 + 线程锁)。
存储的是「源码」,运行时由 :mod:`djinn.strategy.user` 动态编译成 Strategy 子类。
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from djinn.utils.logging import get_logger

_log = get_logger(__name__)

DEFAULT_DB_PATH = ".cache/djinn_strategies.db"

KIND_PYTHON = "python"
KIND_PINE = "pine"


@dataclass
class UserStrategyRecord:
    """一条用户策略记录。"""

    strategy_id: str
    name: str
    kind: str  # "python" / "pine"
    source_code: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "kind": self.kind,
            "source_code": self.source_code,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class StrategyStore:
    """SQLite 用户策略注册表(线程安全)。"""

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
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    name        TEXT NOT NULL UNIQUE,
                    kind        TEXT NOT NULL,
                    source_code TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )
            c.commit()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def create(
        self,
        name: str,
        source_code: str,
        *,
        kind: str = KIND_PYTHON,
        description: str = "",
    ) -> UserStrategyRecord:
        name = name.strip()
        if not name:
            raise ValueError("策略名称不能为空")
        if not source_code.strip():
            raise ValueError("策略源码不能为空")
        rec = UserStrategyRecord(
            strategy_id=uuid.uuid4().hex[:12],
            name=name,
            kind=kind,
            source_code=source_code,
            description=description,
            created_at=self._now(),
            updated_at=self._now(),
        )
        with self._lock, self._conn() as c:
            try:
                c.execute(
                    "INSERT INTO strategies VALUES (?,?,?,?,?,?,?)",
                    (
                        rec.strategy_id,
                        rec.name,
                        rec.kind,
                        rec.source_code,
                        rec.description,
                        rec.created_at,
                        rec.updated_at,
                    ),
                )
                c.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"策略名称已存在: {name}") from e
        return rec

    def get(self, strategy_id: str) -> UserStrategyRecord | None:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT * FROM strategies WHERE strategy_id=?", (strategy_id,)
            ).fetchone()
            return self._row_to_rec(row) if row else None

    def get_by_name(self, name: str) -> UserStrategyRecord | None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT * FROM strategies WHERE name=?", (name,)).fetchone()
            return self._row_to_rec(row) if row else None

    def list_strategies(self) -> list[UserStrategyRecord]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT * FROM strategies ORDER BY name ASC").fetchall()
            return [self._row_to_rec(r) for r in rows]

    def update(
        self,
        strategy_id: str,
        *,
        name: str | None = None,
        source_code: str | None = None,
        kind: str | None = None,
        description: str | None = None,
    ) -> UserStrategyRecord | None:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT * FROM strategies WHERE strategy_id=?", (strategy_id,)
            ).fetchone()
            if row is None:
                return None
            rec = self._row_to_rec(row)
            if name is not None:
                name = name.strip()
                if not name:
                    raise ValueError("策略名称不能为空")
                rec.name = name
            if source_code is not None:
                if not source_code.strip():
                    raise ValueError("策略源码不能为空")
                rec.source_code = source_code
            if kind is not None:
                rec.kind = kind
            if description is not None:
                rec.description = description
            rec.updated_at = self._now()
            try:
                c.execute(
                    "UPDATE strategies SET name=?, kind=?, source_code=?, "
                    "description=?, updated_at=? WHERE strategy_id=?",
                    (
                        rec.name,
                        rec.kind,
                        rec.source_code,
                        rec.description,
                        rec.updated_at,
                        rec.strategy_id,
                    ),
                )
                c.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"策略名称已存在: {rec.name}") from e
        return rec

    def delete(self, strategy_id: str) -> bool:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "DELETE FROM strategies WHERE strategy_id=?", (strategy_id,)
            )
            c.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_rec(row: sqlite3.Row) -> UserStrategyRecord:
        return UserStrategyRecord(
            strategy_id=row["strategy_id"],
            name=row["name"],
            kind=row["kind"],
            source_code=row["source_code"],
            description=row["description"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@lru_cache(maxsize=1)
def get_strategy_store() -> StrategyStore:
    """进程内单例策略存储(CLI / API / 后台 job 共享同一实例)。"""
    return StrategyStore()

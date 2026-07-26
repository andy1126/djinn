"""异步任务注册表:SQLite 持久化 job 状态 + 进度回调。

回测/扫描为长任务,在后台线程跑;进度通过 :class:`ProgressCallback` 上报,
供 WebSocket 推送与 GET 轮询。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from djinn.utils.logging import get_logger

_log = get_logger(__name__)

DEFAULT_DB_PATH = ".cache/djinn_jobs.db"


def make_title(config: dict[str, Any], *, kind: str, target: str | None = None) -> str:
    """从 BacktestConfig dict 派生可读任务标题。

    回测:  MACrossover · NVDA,AAPL · 2023-01-01~2024-12-31
    扫描:  参数扫描 MACrossover · NVDA,AAPL · 2023-01-01~2024-12-31 · 目标=sharpe

    config 来自 ``req.config.model_dump(mode="json")``,``period.start/end`` 已是 ISO
    字符串。标的超过 3 个时截断为前 3 + ``+N``。
    """
    strat = (config.get("strategy") or {}).get("name", "?")
    syms: list[str] = (config.get("universe") or {}).get("symbols") or []
    if len(syms) <= 3:
        sym_str = ",".join(syms)
    else:
        sym_str = f"{','.join(syms[:3])},+{len(syms) - 3}"
    period = config.get("period") or {}
    start = period.get("start", "")
    end = period.get("end", "")
    base = f"{strat} · {sym_str} · {start}~{end}"
    if kind == "sweep":
        return f"参数扫描 {base} · 目标={target or 'sharpe'}"
    return base


@dataclass
class JobRecord:
    """任务记录。"""

    job_id: str
    kind: str  # "backtest" / "sweep"
    status: str = "pending"  # pending / running / done / error
    progress: float = 0.0
    stage: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        meta = (self.result or {}).get("__meta__", {}) or {}
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "error": self.error,
            "result": self.result,
            "title": meta.get("title", ""),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ProgressCallback:
    """进度回调(线程安全,供后台任务上报)。"""

    job_id: str
    registry: JobRegistry
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, progress: float, stage: str = "") -> None:
        with self.lock:
            self.registry.update(self.job_id, progress=progress, stage=stage)


class JobRegistry:
    """SQLite 任务注册表(线程安全)。"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Callable[[JobRecord], None]]] = {}
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    stage TEXT NOT NULL,
                    error TEXT,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)
            c.commit()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def create(self, kind: str, meta: dict[str, Any] | None = None) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        rec = JobRecord(
            job_id=job_id,
            kind=kind,
            status="pending",
            result={"__meta__": meta or {}},
            created_at=self._now(),
            updated_at=self._now(),
        )
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    rec.job_id,
                    rec.kind,
                    rec.status,
                    rec.progress,
                    rec.stage,
                    rec.error,
                    json.dumps(rec.result) if rec.result else None,
                    rec.created_at,
                    rec.updated_at,
                ),
            )
            c.commit()
        return rec

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        stage: str | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> JobRecord | None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            rec = self._row_to_rec(row)
            if status is not None:
                rec.status = status
            if progress is not None:
                rec.progress = progress
            if stage is not None:
                rec.stage = stage
            if error is not None:
                rec.error = error
            if result is not None:
                rec.result = result
            rec.updated_at = self._now()
            c.execute(
                "UPDATE jobs SET status=?, progress=?, stage=?, error=?, result=?, updated_at=? WHERE job_id=?",
                (
                    rec.status,
                    rec.progress,
                    rec.stage,
                    rec.error,
                    json.dumps(rec.result) if rec.result else None,
                    rec.updated_at,
                    rec.job_id,
                ),
            )
            c.commit()
        self._notify(rec)
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return self._row_to_rec(row) if row else None

    def list(self, limit: int = 50, kind: str | None = None) -> list[JobRecord]:
        with self._lock, self._conn() as c:
            if kind:
                rows = c.execute(
                    "SELECT * FROM jobs WHERE kind=? ORDER BY updated_at DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [self._row_to_rec(r) for r in rows]

    def subscribe(self, job_id: str, cb: Callable[[JobRecord], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(cb)

    def unsubscribe(self, job_id: str, cb: Callable[[JobRecord], None]) -> None:
        with self._lock:
            if job_id in self._subscribers:
                with suppress(ValueError):
                    self._subscribers[job_id].remove(cb)

    def _notify(self, rec: JobRecord) -> None:
        cbs = self._subscribers.get(rec.job_id, [])
        for cb in cbs:
            with suppress(Exception):
                cb(rec)

    @staticmethod
    def _row_to_rec(row: sqlite3.Row) -> JobRecord:
        result_raw = row["result"]
        result = json.loads(result_raw) if result_raw else None
        return JobRecord(
            job_id=row["job_id"],
            kind=row["kind"],
            status=row["status"],
            progress=row["progress"],
            stage=row["stage"],
            error=row["error"],
            result=result,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# ── 后台任务执行 ──────────────────────────────────────
def run_backtest_job(
    registry: JobRegistry,
    job_id: str,
    csv_dir: str | None = None,
) -> None:
    """在后台线程执行回测任务(更新 job 状态与结果)。"""
    from djinn.cli.runner import run_backtest
    from djinn.config import load_config

    job = registry.get(job_id)
    meta = (job.result or {}).get("__meta__", {}) if job and job.result else {}
    config_dict = meta.get("config", {})

    try:
        registry.update(job_id, status="running", progress=0.1, stage="加载配置")
        cfg = load_config(data=config_dict)
        registry.update(job_id, progress=0.2, stage="拉取数据")
        result = run_backtest(cfg, csv_dir=csv_dir)
        registry.update(job_id, progress=0.8, stage="生成报告")
        report = result.report
        # 结果摘要(完整曲线通过单独端点按需取)
        summary = report.summary()
        registry.update(
            job_id,
            status="done",
            progress=1.0,
            stage="完成",
            result={"__meta__": meta, "summary": summary, "symbols": report.symbols},
        )
    except Exception as e:
        _log.exception("回测任务 %s 失败", job_id)
        registry.update(job_id, status="error", error=f"{type(e).__name__}: {e}")


def run_sweep_job(
    registry: JobRegistry,
    job_id: str,
    csv_dir: str | None = None,
) -> None:
    """在后台线程执行参数扫描。"""
    from djinn.cli.sweep import _run_one
    from djinn.config import load_config
    from djinn.data import DataCache, default_registry
    from djinn.utils.logging import get_logger

    job = registry.get(job_id)
    meta = (job.result or {}).get("__meta__", {}) if job and job.result else {}
    config_dict = meta.get("config", {})
    grid = meta.get("grid", {})
    target = meta.get("target", "sharpe")

    log = get_logger(__name__)
    try:
        registry.update(job_id, status="running", progress=0.1, stage="加载配置")
        cfg = load_config(data=config_dict)
        cache = DataCache()
        registry_obj = default_registry(csv_dir=csv_dir, cache=cache)
        market = cfg.resolved_market()
        # 预拉取数据
        for sym in cfg.universe.symbols:
            registry_obj.get_ohlcv(
                sym, cfg.period.start, cfg.period.end, cfg.adjust, market=market
            )
        from djinn.cli.sweep import _expand_grid

        combos = _expand_grid(grid)
        n = len(combos) or 1
        results: list[dict[str, Any]] = []
        for i, c in enumerate(combos):
            results.append(_run_one(cfg, registry_obj, c, target))
            registry.update(
                job_id, progress=0.1 + 0.85 * (i + 1) / n, stage=f"扫描 {i + 1}/{n}"
            )
        results.sort(key=lambda r: r.get(target, 0.0), reverse=True)
        registry.update(
            job_id,
            status="done",
            progress=1.0,
            stage="完成",
            result={"__meta__": meta, "results": results, "target": target},
        )
    except Exception as e:
        log.exception("扫描任务 %s 失败", job_id)
        registry.update(job_id, status="error", error=f"{type(e).__name__}: {e}")

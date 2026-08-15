"""异步任务注册表:SQLite 持久化 job 状态 + 进度回调。

回测/扫描为长任务,在后台线程跑;进度通过 :class:`ProgressCallback` 上报,
供 WebSocket 推送与 GET 轮询。
"""

from __future__ import annotations

import builtins
import json
import math
import os
import queue
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from djinn.data.provider import DataProvider, ProviderRegistry
from djinn.utils.exceptions import BacktestCancelled
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
    if kind == "walk-forward":
        wf = config.get("walk_forward") or {}
        return (
            f"Walk-Forward {base} · "
            f"is={wf.get('is_days', '?')}/oos={wf.get('oos_days', '?')} · "
            f"目标={target or 'sharpe'}"
        )
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
    """进度回调(线程安全,供后台任务上报)。

    E12:内置节流——高频 ``update`` 每 ``min_interval_sec`` 秒才写一次 DB
    (降低进度写放大);``force=True`` 或终态跳过节流。
    """

    job_id: str
    registry: JobRegistry
    lock: threading.Lock = field(default_factory=threading.Lock)
    min_interval_sec: float = 0.5

    def __post_init__(self) -> None:
        self._last_write = 0.0

    def update(self, progress: float, stage: str = "", *, force: bool = False) -> None:
        now = time.monotonic()
        with self.lock:
            if not force and now - self._last_write < self.min_interval_sec:
                return
            self._last_write = now
            self.registry.update(self.job_id, progress=progress, stage=stage)


class JobRegistry:
    """SQLite 任务注册表(线程安全)。"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Callable[[JobRecord], None]]] = {}
        self._cancel_flags: set[str] = set()  # 请求取消的任务 id(E4)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """
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
                """
            )
            c.commit()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def create(self, kind: str, meta: dict[str, Any] | None = None) -> JobRecord:
        # E12:uuid4 hex[:12] 撞车即重试(否则 500);连续冲突 3 次才报错。
        created = self._now()
        for _ in range(3):
            job_id = uuid.uuid4().hex[:12]
            rec = JobRecord(
                job_id=job_id,
                kind=kind,
                status="pending",
                result={"__meta__": meta or {}},
                created_at=created,
                updated_at=created,
            )
            try:
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
            except sqlite3.IntegrityError:
                continue  # 主键冲突,换新 id 重试
            return rec
        raise RuntimeError("生成任务 ID 失败(连续主键冲突)")

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

    # ── 取消(E4)────────────────────────────────────────
    def request_cancel(self, job_id: str) -> bool:
        """请求取消任务(仅 pending / running 可取消),返回是否受理。"""
        job = self.get(job_id)
        if job is None or job.status not in ("pending", "running"):
            return False
        with self._lock:
            self._cancel_flags.add(job_id)
        return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel_flags

    def clear_cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancel_flags.discard(job_id)

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

    def list_by_status(self, statuses: builtins.list[str]) -> builtins.list[JobRecord]:
        """按状态集查询全部记录(无 limit)——孤儿恢复用,避免漏掉更老的任务(E7)。"""
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._lock, self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM jobs WHERE status IN ({placeholders})",
                tuple(statuses),
            ).fetchall()
            return [self._row_to_rec(r) for r in rows]

    def purge_older_than(self, days: int, keep_kinds: tuple[str, ...] = ()) -> int:
        """删除 ``updated_at`` 早于 ``days`` 天且已终态的 job 记录及其产物(E6)。

        仅清理 ``done / error / cancelled``(不碰 running / pending);同步删除报告
        缓存与 exports 目录。返回删除条数。
        """
        import shutil

        from djinn.api.report_store import delete as delete_report

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT job_id, kind FROM jobs "
                "WHERE status IN ('done','error','cancelled') AND updated_at < ?",
                (cutoff,),
            ).fetchall()
        ids = [str(r["job_id"]) for r in rows if r["kind"] not in keep_kinds]
        exports_root = Path(".cache/exports")
        for job_id in ids:
            with self._lock, self._conn() as c:
                c.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
                c.commit()
            delete_report(job_id)
            with suppress(OSError):
                shutil.rmtree(exports_root / job_id, ignore_errors=True)
        if ids:
            _log.info("清理 %d 个过期任务", len(ids))
        return len(ids)

    def subscribe(self, job_id: str, cb: Callable[[JobRecord], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(cb)

    def unsubscribe(self, job_id: str, cb: Callable[[JobRecord], None]) -> None:
        with self._lock:
            if job_id in self._subscribers:
                with suppress(ValueError):
                    self._subscribers[job_id].remove(cb)

    def _notify(self, rec: JobRecord) -> None:
        # E12:锁内取订阅者快照,锁外执行回调(避免与 subscribe/unsubscribe 竞态)
        with self._lock:
            cbs = list(self._subscribers.get(rec.job_id, []))
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
    provider_registry: ProviderRegistry | None = None,
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
        # Web 报告 / 导出端点从 report_store 读缓存,故后台任务一次性算好归因落盘,
        # 避免读端点重跑回测(参见 api/report_store.py)。
        result = run_backtest(
            cfg,
            csv_dir=csv_dir,
            registry=provider_registry,
            with_attribution=True,
            should_stop=lambda: registry.is_cancel_requested(job_id),
        )
        registry.update(job_id, progress=0.8, stage="生成报告")
        report = result.report
        summary = report.summary()
        # 落盘完整序列化报告(含归因)供 /report 与 /export 读端点复用
        from djinn.api.report_store import save, serialize_report

        save(job_id, serialize_report(report))
        registry.update(
            job_id,
            status="done",
            progress=1.0,
            stage="完成",
            result={"__meta__": meta, "summary": summary, "symbols": report.symbols},
        )
        registry.clear_cancel(job_id)
    except BacktestCancelled:
        _log.info("回测任务 %s 已取消", job_id)
        registry.update(job_id, status="cancelled", stage="已取消")
    except Exception as e:
        _log.exception("回测任务 %s 失败", job_id)
        registry.update(job_id, status="error", error=f"{type(e).__name__}: {e}")


def run_sweep_job(
    registry: JobRegistry,
    job_id: str,
    csv_dir: str | None = None,
    provider_registry: ProviderRegistry | None = None,
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
        # 注入的 provider_registry 优先(测试 / API 复用单例缓存);否则自建。
        if provider_registry is not None:
            registry_obj = provider_registry
        else:
            registry_obj = default_registry(csv_dir=csv_dir, cache=DataCache())
        market = cfg.resolved_market()
        # 预拉数据:base symbols ∪ 所有扫到的 universe.index 的成分
        # (扫 index 时各组合成分股不同,这里统一预拉缓存,_run_one 内按需命中)
        from djinn.cli.sweep import _expand_grid, _index_symbols

        combos = _expand_grid(grid)
        all_symbols: set[str] = set(cfg.universe.symbols)
        for idx in grid.get("universe.index", []) or []:
            all_symbols.update(_index_symbols(str(idx), registry_obj))
        for sym in all_symbols:
            registry_obj.get_ohlcv(
                sym, cfg.period.start, cfg.period.end, cfg.adjust, market=market
            )
        n = len(combos) or 1
        # D7:并行扫描(线程共享缓存;``SweepRequest.parallel`` 控制)
        parallel = bool(meta.get("parallel", False))
        if parallel and n > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            _workers = int(os.environ.get("DJINN_FETCH_WORKERS", "8"))
            ex = ThreadPoolExecutor(max_workers=_workers)
            results = []
            try:
                futs = [
                    ex.submit(_run_one, cfg, registry_obj, c, target) for c in combos
                ]
                for done, fut in enumerate(as_completed(futs), 1):
                    if registry.is_cancel_requested(job_id):
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise BacktestCancelled(f"扫描已取消 @{done}/{n}")
                    results.append(fut.result())
                    registry.update(
                        job_id,
                        progress=0.1 + 0.85 * done / n,
                        stage=f"扫描 {done}/{n}",
                    )
            finally:
                ex.shutdown(wait=True)
        else:
            results = []
            for i, c in enumerate(combos):
                if registry.is_cancel_requested(job_id):
                    raise BacktestCancelled(f"扫描已取消 @组合 {i + 1}/{n}")
                results.append(_run_one(cfg, registry_obj, c, target))
                registry.update(
                    job_id,
                    progress=0.1 + 0.85 * (i + 1) / n,
                    stage=f"扫描 {i + 1}/{n}",
                )
        # 排序:max_drawdown 等越小越好的目标需升序(reversed=False)。
        from djinn.cli.sweep import REVERSE_MIN_TARGETS

        reverse_sort = target not in REVERSE_MIN_TARGETS
        nan_val = float("-inf") if reverse_sort else float("inf")

        def _key(r: dict[str, Any]) -> float:
            v = r.get(target)
            if v is None:
                return nan_val
            try:
                f = float(v)
            except (TypeError, ValueError):
                return nan_val
            return f if math.isfinite(f) else nan_val

        results.sort(key=_key, reverse=reverse_sort)
        registry.update(
            job_id,
            status="done",
            progress=1.0,
            stage="完成",
            result={"__meta__": meta, "results": results, "target": target},
        )
        registry.clear_cancel(job_id)
    except BacktestCancelled:
        log.info("扫描任务 %s 已取消", job_id)
        registry.update(job_id, status="cancelled", stage="已取消")
    except Exception as e:
        log.exception("扫描任务 %s 失败", job_id)
        registry.update(job_id, status="error", error=f"{type(e).__name__}: {e}")


def run_walk_forward_job(
    registry: JobRegistry,
    job_id: str,
    csv_dir: str | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> None:
    """在后台线程执行 walk-forward 分析(逐窗口 IS 独立选参 + OOS 评估)。

    依赖 ``__meta__`` 约定:首行从 ``result["__meta__"]`` 重建 config/grid/target,
    故孤儿恢复只需 ``(registry, job_id)``。完成结果保留 ``__meta__``。
    """
    from djinn.cli.walk_forward import walk_forward
    from djinn.config import load_config
    from djinn.data import DataCache, default_registry

    job = registry.get(job_id)
    meta = (job.result or {}).get("__meta__", {}) if job and job.result else {}
    config_dict = meta.get("config", {})
    grid = meta.get("grid") or None
    target = meta.get("target") or None
    parallel = bool(meta.get("parallel", False))

    log = get_logger(__name__)
    try:
        registry.update(job_id, status="running", progress=0.05, stage="加载配置")
        cfg = load_config(data=config_dict)
        # 注入的 provider_registry 优先(测试 / API 复用单例缓存);否则自建。
        if provider_registry is not None:
            registry_obj = provider_registry
        else:
            registry_obj = default_registry(csv_dir=csv_dir, cache=DataCache())

        def _progress(done: int, total: int) -> None:
            registry.update(
                job_id,
                progress=0.1 + 0.85 * done / max(1, total),
                stage=f"窗口 {done}/{total}",
            )

        report = walk_forward(
            cfg,
            registry=registry_obj,
            grid=grid,
            target=target,
            parallel=parallel,
            on_progress=_progress,
            should_stop=lambda: registry.is_cancel_requested(job_id),
        )
        registry.update(
            job_id,
            status="done",
            progress=1.0,
            stage="完成",
            result={"__meta__": meta, "report": report.to_dict()},
        )
        registry.clear_cancel(job_id)
    except BacktestCancelled:
        log.info("walk-forward 任务 %s 已取消", job_id)
        registry.update(job_id, status="cancelled", stage="已取消")
    except Exception as e:
        log.exception("walk-forward 任务 %s 失败", job_id)
        registry.update(job_id, status="error", error=f"{type(e).__name__}: {e}")


# ── 横截面 alpha 任务(因子分析 / 选股)────────────────────
def _json_scalar(v: Any) -> Any:
    """标量 JSON 友好化:numpy 标量 → python,NaN/Inf → None。"""
    if v is None or isinstance(v, (str, bool, int)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f if math.isfinite(f) else None


def _index_components_with_provider(
    registry: ProviderRegistry, index: str
) -> tuple[list[str], DataProvider | None, list[str]]:
    """取首个成功 provider 的成分股,并返回该 provider 与各 provider 的失败原因。

    返回 ``(symbols, provider, errors)``:``errors`` 形如 ``["akshare: akshare 未安装"]``,
    供路由在全部失败时透出底层原因(而非笼统的「无 provider 提供指数」)。
    """
    errors: list[str] = []
    for p in registry.providers:
        try:
            comps = p.get_index_components(index)
        except NotImplementedError:
            continue
        except Exception as e:
            errors.append(f"{p.name}: {e}")
            _log.warning("provider %s 取指数 %s 成分失败: %s", p.name, index, e)
            continue
        if comps:
            return [str(s) for s in comps], p, errors
    return [], None, errors


def _index_components(registry: ProviderRegistry, index: str) -> list[str]:
    """从首个支持指数成分的 provider 取成分股(全部失败返回 [])。"""
    symbols, _, _ = _index_components_with_provider(registry, index)
    return symbols


def _resolve_universe(meta: dict[str, Any], registry: ProviderRegistry) -> list[str]:
    """从任务 meta 解析候选标的池:显式 symbols 优先,否则 index 成分。"""
    symbols = [str(s) for s in (meta.get("symbols") or [])]
    if not symbols and meta.get("index"):
        symbols = _index_components(registry, str(meta["index"]))
    return list(dict.fromkeys(symbols))


def _industry_map(registry: ProviderRegistry, symbols: list[str]) -> dict[str, str]:
    """取 symbol → 行业映射(全部 provider 失败返回 {})。"""
    for p in registry.providers:
        try:
            m = p.get_industry_map(symbols)
        except NotImplementedError:
            continue
        except Exception as e:
            _log.warning("provider %s 取行业映射失败: %s", p.name, e)
            continue
        if m:
            return {str(k): str(v) for k, v in m.items()}
    return {}


def _build_fundamental_panels(
    symbols: list[str],
    prices_index: Any,
    start: date,
    end: date,
    registry: ProviderRegistry,
    market: Any,
    *,
    engine: Any = None,
) -> dict[str, Any]:
    """组装 point-in-time 基本面宽表(供估值 / 质量 / 成长类因子)。

    ``engine`` 复用时,C3 的口径告警(``engine.caveats()``)会在组装过程中被
    填充——调用方需用与 ``_ohlcv_panels`` 同一引擎实例,以便把 caveats 透出报告。
    """
    import pandas as pd

    from djinn.data.providers.fundamentals_router import FundamentalsRouter
    from djinn.factor.engine import DEFAULT_FUNDAMENTAL_FIELDS, FactorEngine

    eng = engine if engine is not None else FactorEngine()
    return eng._fundamental_panels(
        DEFAULT_FUNDAMENTAL_FIELDS,
        symbols,
        pd.DatetimeIndex(prices_index),
        start,
        end,
        FundamentalsRouter(registry.providers),
        market,
    )


def run_factor_analysis_job(
    registry: JobRegistry,
    job_id: str,
    provider_registry: ProviderRegistry | None = None,
) -> None:
    """后台执行单因子分析:universe × 区间 → IC / 分层 / 衰减 / 换手报告。"""

    from djinn.data import default_registry
    from djinn.data.schema import Adjust, Market
    from djinn.factor import FactorEngine, make_factor
    from djinn.factor.analysis import analyze_factor, compute_forward_returns

    job = registry.get(job_id)
    meta = (job.result or {}).get("__meta__", {}) if job and job.result else {}
    preg = provider_registry or default_registry()
    try:
        registry.update(job_id, status="running", progress=0.05, stage="解析标的池")
        factor_name = str(meta["factor"])
        params = meta.get("params") or {}
        market = Market(meta["market"]) if meta.get("market") else None
        start = date.fromisoformat(str(meta["start"]))
        end = date.fromisoformat(str(meta["end"]))
        adjust = Adjust(str(meta.get("adjust", "backward")))
        symbols = _resolve_universe(meta, preg)
        if not symbols:
            raise ValueError("标的池为空(需提供 symbols 或可解析的 index)")
        factor = make_factor(factor_name, **params)

        if registry.is_cancel_requested(job_id):
            raise BacktestCancelled("因子分析已取消")
        registry.update(job_id, progress=0.2, stage=f"拉取 {len(symbols)} 只行情")
        eng = FactorEngine()
        prices, ohlcv = eng._ohlcv_panels(symbols, start, end, preg, market, adjust)
        if registry.is_cancel_requested(job_id):
            raise BacktestCancelled("因子分析已取消")
        registry.update(job_id, progress=0.45, stage="计算因子面板")
        fundamentals = _build_fundamental_panels(
            symbols, prices.index, start, end, preg, market, engine=eng
        )
        factor.validate_inputs(fundamentals, ohlcv)
        # C6:因子声明了 benchmark 时注入真实基准日收益(否则 BetaFactor 退化为等权代理)
        bench = getattr(factor, "benchmark", None)
        if bench:
            bench_rets = eng._benchmark_returns(
                str(bench), start, end, preg, market, adjust
            )
            if len(bench_rets):
                ohlcv_any: dict[str, Any] = ohlcv
                ohlcv_any["__benchmark__"] = bench_rets
        factor_panel = factor.compute(prices, ohlcv, fundamentals)

        if registry.is_cancel_requested(job_id):
            raise BacktestCancelled("因子分析已取消")
        registry.update(job_id, progress=0.7, stage="IC / 分层分析")
        periods = tuple(int(p) for p in (meta.get("periods") or [1, 5, 10]))
        fwd = compute_forward_returns(prices, periods)
        report = analyze_factor(
            factor_panel,
            fwd,
            name=factor.name,
            ic_method=str(meta.get("ic_method", "spearman")),
            n_quantiles=int(meta.get("n_quantiles", 5)),
            industry_map=_industry_map(preg, symbols),
            caveats=eng.caveats(),
        )
        registry.update(
            job_id,
            status="done",
            progress=1.0,
            stage="完成",
            result={"__meta__": meta, "report": report.to_dict(), "symbols": symbols},
        )
    except BacktestCancelled:
        # E4:协作式取消 → cancelled(非 error),__meta__ 保留可重新提交
        registry.update(job_id, status="cancelled", stage="已取消")
    except Exception as e:
        _log.exception("因子分析任务 %s 失败", job_id)
        registry.update(job_id, status="error", error=f"{type(e).__name__}: {e}")


def run_factor_matrix_job(
    registry: JobRegistry,
    job_id: str,
    provider_registry: ProviderRegistry | None = None,
) -> None:
    """后台执行多因子诊断:universe × 区间 → 因子相关矩阵 + 各因子 IC 汇总。"""
    from djinn.data import default_registry
    from djinn.data.schema import Adjust, Market
    from djinn.factor import FactorEngine, make_factor
    from djinn.factor.analysis import analyze_factor_matrix

    job = registry.get(job_id)
    meta = (job.result or {}).get("__meta__", {}) if job and job.result else {}
    preg = provider_registry or default_registry()
    try:
        registry.update(job_id, status="running", progress=0.05, stage="解析标的池")
        pts = meta.get("factors") or []
        market = Market(meta["market"]) if meta.get("market") else None
        start = date.fromisoformat(str(meta["start"]))
        end = date.fromisoformat(str(meta["end"]))
        adjust = Adjust(str(meta.get("adjust", "backward")))
        symbols = _resolve_universe(meta, preg)
        if not symbols:
            raise ValueError("标的池为空(需提供 symbols 或可解析的 index)")
        if not pts or len(pts) < 2:
            raise ValueError("多因子诊断需至少 2 个因子")

        if registry.is_cancel_requested(job_id):
            raise BacktestCancelled("多因子诊断已取消")
        registry.update(job_id, progress=0.15, stage=f"拉取 {len(symbols)} 只行情")
        eng = FactorEngine()
        prices, ohlcv = eng._ohlcv_panels(symbols, start, end, preg, market, adjust)
        if registry.is_cancel_requested(job_id):
            raise BacktestCancelled("多因子诊断已取消")
        registry.update(job_id, progress=0.45, stage=f"计算 {len(pts)} 个因子面板")
        fundamentals = _build_fundamental_panels(
            symbols, prices.index, start, end, preg, market
        )
        panels: dict[str, Any] = {}
        for pt in pts:
            if registry.is_cancel_requested(job_id):
                raise BacktestCancelled("多因子诊断已取消")
            name = str(pt["factor"])
            params = pt.get("params") or {}
            direction = int(pt.get("direction", 1))
            f = make_factor(name, **params)
            f.validate_inputs(fundamentals, ohlcv)
            panel = f.compute(prices, ohlcv, fundamentals)
            # direction=-1 → 翻符号(诊断相关用同一口径因子值)
            if direction < 0:
                panel = -panel
            # 同名因子重复入组合:加序号避免字典覆盖
            key = (
                name
                if name not in panels
                else f"{name}#{sum(1 for k in panels if k.startswith(name))}"
            )
            panels[key] = panel

        registry.update(job_id, progress=0.75, stage="相关 / IC 汇总")
        periods = tuple(int(p) for p in (meta.get("periods") or [1, 5, 10]))
        ic_method = str(meta.get("ic_method", "spearman"))
        report = analyze_factor_matrix(
            panels,
            prices,
            periods=periods,
            ic_method=ic_method,  # type: ignore[arg-type]
            orthogonalized=bool(meta.get("orthogonalized", False)),
        )
        registry.update(
            job_id,
            status="done",
            progress=1.0,
            stage="完成",
            result={"__meta__": meta, "report": report.to_dict(), "symbols": symbols},
        )
    except BacktestCancelled:
        # E4:协作式取消 → cancelled(非 error),__meta__ 保留可重新提交
        registry.update(job_id, status="cancelled", stage="已取消")
    except Exception as e:
        _log.exception("多因子诊断任务 %s 失败", job_id)
        registry.update(job_id, status="error", error=f"{type(e).__name__}: {e}")


def _score_symbols(
    registry: ProviderRegistry,
    symbols: list[str],
    scores_meta: list[dict[str, Any]],
    when: date,
    market: Any,
    lookback_days: int,
) -> dict[str, float]:
    """对候选池在 ``when``(或之前最近交易日)截面做多因子合成打分。"""
    import pandas as pd

    from djinn.data.schema import Adjust
    from djinn.factor import FactorEngine, make_factor
    from djinn.screen.scoring import FactorScore, score_cross_section

    scores = [FactorScore(**s) for s in scores_meta]
    factors = [make_factor(s.factor) for s in scores]
    # 预留足够自然日覆盖 lookback_days 个交易日
    start = when - timedelta(days=max(lookback_days * 2, 60))
    eng = FactorEngine()
    prices, ohlcv = eng._ohlcv_panels(
        symbols, start, when, registry, market, Adjust.BACKWARD
    )
    fundamentals = _build_fundamental_panels(
        symbols, prices.index, start, when, registry, market
    )
    data: dict[str, Any] = {}
    for f in factors:
        f.validate_inputs(fundamentals, ohlcv)
        data[f.name] = f.compute(prices, ohlcv, fundamentals)
    idx = prices.index[prices.index <= pd.Timestamp(when)]
    if len(idx) == 0:
        return {}
    ts = idx[-1]
    cross = pd.DataFrame(
        {name: df.loc[ts] for name, df in data.items() if ts in df.index}
    )
    scored = score_cross_section(cross, scores)
    return {str(k): float(v) for k, v in scored.items()}


def _screen_row(symbol: str, snap: Any, score_map: dict[str, float]) -> dict[str, Any]:
    """单标的选股结果行:symbol + 基本面字段 + 可选得分。"""
    row: dict[str, Any] = {"symbol": symbol}
    if snap is not None and symbol in snap.index:
        for col in snap.columns:
            row[str(col)] = _json_scalar(snap.loc[symbol, col])
    if symbol in score_map:
        row["score"] = score_map[symbol]
    return row


def run_screen_job(
    registry: JobRegistry,
    job_id: str,
    provider_registry: ProviderRegistry | None = None,
) -> None:
    """后台执行截面选股:条件过滤 + 可选多因子打分排序,产出股票列表 + 得分。"""
    from djinn.data import default_registry
    from djinn.data.providers.fundamentals_router import FundamentalsRouter
    from djinn.data.schema import Market
    from djinn.screen.screener import ScreenCondition, Screener

    job = registry.get(job_id)
    meta = (job.result or {}).get("__meta__", {}) if job and job.result else {}
    preg = provider_registry or default_registry()
    try:
        registry.update(job_id, status="running", progress=0.05, stage="解析候选池")
        market = Market(meta["market"]) if meta.get("market") else None
        symbols = _resolve_universe(meta, preg)
        if not symbols:
            raise ValueError("候选池为空(需提供 symbols 或可解析的 index)")
        when = (
            date.fromisoformat(str(meta["when"])) if meta.get("when") else date.today()
        )

        if registry.is_cancel_requested(job_id):
            raise BacktestCancelled("选股已取消")
        registry.update(job_id, progress=0.2, stage=f"拉取 {len(symbols)} 只基本面快照")
        router = FundamentalsRouter(preg.providers)
        snap = router.get_snapshot(symbols, when, market)
        conditions = [ScreenCondition(**c) for c in (meta.get("conditions") or [])]
        passed = Screener.apply(conditions, snap)

        score_map: dict[str, float] = {}
        scores_meta = meta.get("scores") or []
        if scores_meta:
            if registry.is_cancel_requested(job_id):
                raise BacktestCancelled("选股已取消")
            registry.update(job_id, progress=0.5, stage="多因子打分排序")
            score_map = _score_symbols(
                preg,
                symbols,
                scores_meta,
                when,
                market,
                int(meta.get("lookback_days", 120)),
            )
            passed = [s for s in passed if s in score_map]
            passed.sort(key=lambda s: score_map[s], reverse=True)
            top_n = meta.get("top_n")
            if top_n:
                passed = passed[: int(top_n)]
        else:
            passed = sorted(passed)

        registry.update(job_id, progress=0.85, stage="汇总结果")
        rows = [_screen_row(s, snap, score_map) for s in passed]
        registry.update(
            job_id,
            status="done",
            progress=1.0,
            stage="完成",
            result={"__meta__": meta, "count": len(rows), "results": rows},
        )
    except BacktestCancelled:
        # E4:协作式取消 → cancelled(非 error),__meta__ 保留可重新提交
        registry.update(job_id, status="cancelled", stage="已取消")
    except Exception as e:
        _log.exception("选股任务 %s 失败", job_id)
        registry.update(job_id, status="error", error=f"{type(e).__name__}: {e}")


# ── 孤儿任务恢复(进程重启)──────────────────────────────
# 长任务经 BackgroundTasks 在进程内后台线程执行,进程重启即线程被杀,
# 只留下持久化的 running/pending 快照。启动时扫描并重新提交续跑。
# 每个 runner 首行从 job 行的 __meta__ 重建输入(config/grid/factor/index 等),
# 故只需 (registry, job_id) 即可复现原任务。
_RUNNERS: dict[str, Callable[..., None]] = {
    "backtest": run_backtest_job,
    "sweep": run_sweep_job,
    "walk-forward": run_walk_forward_job,
    "factor-analysis": run_factor_analysis_job,
    "factor-matrix": run_factor_matrix_job,
    "screen": run_screen_job,
}

# E7:已提交恢复的任务 id(防 lifespan 重入重复提交)。
_recovered_jobs: set[str] = set()


def recover_orphaned_jobs(
    registry: JobRegistry,
    provider_registry: ProviderRegistry | None = None,
) -> int:
    """启动时重新提交 running / pending 孤儿任务,返回恢复数。

    用后台线程(而非 ``BackgroundTasks``——启动阶段无 HTTP 请求上下文)。
    传入 ``provider_registry`` 复用共享缓存,避免恢复任务另建默认 registry
    造成缓存不一致。

    测试环境(``DJINN_TEST=1``)下不执行:测试注入 stub registry,不应恢复真实任务。
    """
    if os.environ.get("DJINN_TEST") == "1":
        return 0
    # E7:按状态集查询(无 limit,避免更老孤儿被漏掉);幂等防重复提交。
    orphaned = [
        j
        for j in registry.list_by_status(["running", "pending"])
        if j.kind in _RUNNERS and j.job_id not in _recovered_jobs
    ]
    for job in orphaned:
        _recovered_jobs.add(job.job_id)
        try:
            thread = threading.Thread(
                target=_RUNNERS[job.kind],
                args=(registry, job.job_id),
                kwargs={"provider_registry": provider_registry},
                daemon=True,
                name=f"recover-{job.kind}-{job.job_id}",
            )
            thread.start()
            _log.info("恢复孤儿任务 %s (%s)", job.job_id, job.kind)
        except Exception as e:
            _recovered_jobs.discard(job.job_id)  # 失败可重试
            _log.error("恢复任务 %s 失败: %s", job.job_id, e)
    return len(orphaned)


class JobScheduler:
    """进程内 FIFO 任务调度:最多 ``max_concurrent`` 个任务并发,其余排队(E5)。

    防止多个大回测并发导致内存爆炸。``max_concurrent`` 默认从 env
    ``DJINN_MAX_JOBS`` 读(默认 2),可在构造时覆盖。
    """

    def __init__(
        self, registry: JobRegistry, max_concurrent: int | None = None
    ) -> None:
        self.registry = registry
        self.max_concurrent = max_concurrent or int(
            os.environ.get("DJINN_MAX_JOBS", "2")
        )
        self._sem = threading.Semaphore(self.max_concurrent)
        self._queue: queue.Queue[
            tuple[Callable[..., None], tuple[Any, ...], dict[str, Any]]
        ] = queue.Queue()
        self._dispatcher_thread: threading.Thread | None = None
        self._dispatch_lock = threading.Lock()

    def submit(self, runner: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """入队一个任务;dispatcher 线程按 FIFO 取件,受信号量限流。"""
        self._queue.put((runner, args, kwargs))
        self._ensure_dispatcher()

    def _ensure_dispatcher(self) -> None:
        with self._dispatch_lock:
            if (
                self._dispatcher_thread is None
                or not self._dispatcher_thread.is_alive()
            ):
                self._dispatcher_thread = threading.Thread(
                    target=self._dispatch_loop, daemon=True, name="job-scheduler"
                )
                self._dispatcher_thread.start()

    def _dispatch_loop(self) -> None:
        while True:
            runner, args, kwargs = self._queue.get()
            self._sem.acquire()

            def _run(
                runner: Callable[..., None] = runner,
                args: tuple[Any, ...] = args,
                kwargs: dict[str, Any] = kwargs,
            ) -> None:
                try:
                    runner(*args, **kwargs)
                finally:
                    self._sem.release()

            threading.Thread(
                target=_run,
                daemon=True,
                name=f"job-{args[1] if len(args) > 1 else '?'}",
            ).start()

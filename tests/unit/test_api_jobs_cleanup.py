"""E6(结果过期清理)/ E7(孤儿恢复修复)单测。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from djinn.api.jobs import JobRegistry
from djinn.api.report_store import exists as report_exists
from djinn.api.report_store import save as save_report


def _backdate(registry: JobRegistry, job_id: str, days: int) -> None:
    old = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with registry._lock, registry._conn() as c:
        c.execute("UPDATE jobs SET updated_at=? WHERE job_id=?", (old, job_id))
        c.commit()


def test_list_by_status_no_limit(tmp_path) -> None:
    registry = JobRegistry(db_path=tmp_path / "jobs.db")
    for _ in range(5):
        registry.create("backtest", meta={})
    # 全部置 running
    for j in registry.list(limit=100):
        registry.update(j.job_id, status="running")
    running = registry.list_by_status(["running", "pending"])
    assert len(running) == 5  # 不截断(原 list(limit=1000) 会漏更老的)


def test_purge_older_than_deletes_only_old_terminal(tmp_path) -> None:
    registry = JobRegistry(db_path=tmp_path / "jobs.db")
    old_done = registry.create("backtest", meta={})
    old_running = registry.create("backtest", meta={})
    recent_done = registry.create("backtest", meta={})

    registry.update(old_done.job_id, status="done")
    _backdate(registry, old_done.job_id, 40)  # 40 天前 done → 应删
    save_report(old_done.job_id, {"v": 2})

    registry.update(old_running.job_id, status="running")
    _backdate(registry, old_running.job_id, 40)  # 40 天前 running → 不删

    registry.update(recent_done.job_id, status="done")  # 刚刚 done → 不删

    removed = registry.purge_older_than(days=30)
    assert removed == 1
    assert registry.get(old_done.job_id) is None
    assert not report_exists(old_done.job_id)  # 报告缓存同步删除
    assert registry.get(old_running.job_id) is not None
    assert registry.get(recent_done.job_id) is not None


def test_purge_keep_kinds(tmp_path) -> None:
    registry = JobRegistry(db_path=tmp_path / "jobs.db")
    old = registry.create("sweep", meta={})
    registry.update(old.job_id, status="done")
    _backdate(registry, old.job_id, 40)
    assert registry.purge_older_than(days=30, keep_kinds=("sweep",)) == 0
    assert registry.get(old.job_id) is not None


def test_recover_orphaned_idempotent(tmp_path, monkeypatch) -> None:
    import djinn.api.jobs as jobs_mod

    monkeypatch.delenv("DJINN_TEST", raising=False)
    monkeypatch.setattr(
        jobs_mod,
        "_RUNNERS",
        {"backtest": lambda *a, **k: None, "sweep": lambda *a, **k: None},
    )
    jobs_mod._recovered_jobs.clear()
    registry = JobRegistry(db_path=tmp_path / "jobs.db")
    j1 = registry.create("backtest", meta={})
    j2 = registry.create("sweep", meta={})
    registry.update(j1.job_id, status="running")
    registry.update(j2.job_id, status="pending")

    assert jobs_mod.recover_orphaned_jobs(registry) == 2
    # 幂等:重复调用不重复提交(任务仍在 _recovered_jobs)
    assert jobs_mod.recover_orphaned_jobs(registry) == 0

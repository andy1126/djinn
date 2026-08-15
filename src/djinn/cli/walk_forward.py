"""Walk-Forward 分析:滚动样本外验证(H 计划)。

在 ``period`` 全区间上滚动窗口。每个窗口:
1. **样本内(IS)** 用参数网格独立选参(复用 ``sweep`` 的 ``_run_one``,带暖机),
   按 ``target`` 取最优组合;
2. **样本外(OOS)** 用该窗口 IS 最优参数评估(``run_backtest``,带暖机),账本从
   OOS 起点开、净值天然就是 OOS 段;
3. 全部 OOS 段按段首净值归一化后**拼接**成无前视的 walk-forward 样本外净值。

防未来函数:IS 用 ``[warmup, is_end]``、OOS 用 ``[warmup, oos_end]``,暖机只是更早的
历史;``min_is_sharpe`` 门槛不达标时该窗口不部署(OOS 空仓),避免把 IS 噪声里的
最优参数硬塞进样本外。
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, cast

import pandas as pd

from djinn.analytics.metrics import Metrics, compute_metrics
from djinn.cli.runner import _resolve_universe_symbols, run_backtest
from djinn.cli.sweep import REVERSE_MIN_TARGETS, _apply_param, _expand_grid, _run_one
from djinn.config.models import BacktestConfig, WalkForwardConfig
from djinn.data import ProviderRegistry, default_registry
from djinn.utils.exceptions import BacktestCancelled
from djinn.utils.logging import get_logger

_log = get_logger(__name__)


def _safe_f(v: object) -> float | None:
    """NaN/Inf → None(JSON 不接受);否则转 float。"""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clean_metrics(d: dict[str, Any] | None) -> dict[str, Any] | None:
    """指标 dict 的 NaN/Inf → None(供 to_dict 序列化)。"""
    if d is None:
        return None
    return {
        k: (
            _safe_f(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        )
        for k, v in d.items()
    }


# sweep._run_one 返回结果里的可展示指标键(IS 最优组合 / OOS 段共用)。
_METRIC_KEYS: tuple[str, ...] = (
    "sharpe",
    "sortino",
    "calmar",
    "total_return",
    "annual_return",
    "max_drawdown",
    "volatility",
    "n_trades",
)


@dataclass
class WFWindow:
    """单个 WFO 窗口:IS 最优参数 + OOS 评估结果。``deployed=False`` 表示 IS 未达标未部署。"""

    no: int
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date
    best_params: dict[str, Any] | None = None
    is_metrics: dict[str, Any] | None = None  # IS 最优组合的目标 + 指标
    oos_metrics: dict[str, Any] | None = None
    oos_equity: pd.Series | None = None
    deployed: bool = False

    def to_dict(self) -> dict[str, Any]:
        eq = self.oos_equity
        return {
            "no": self.no,
            "is_start": str(self.is_start),
            "is_end": str(self.is_end),
            "oos_start": str(self.oos_start),
            "oos_end": str(self.oos_end),
            "deployed": self.deployed,
            "best_params": self.best_params,
            "is_metrics": _clean_metrics(self.is_metrics),
            "oos_metrics": _clean_metrics(self.oos_metrics),
            "oos_equity": (
                {
                    "index": [str(d) for d in eq.index],
                    "values": [_safe_f(x) for x in eq],
                }
                if eq is not None and len(eq)
                else None
            ),
        }


@dataclass
class WalkForwardReport:
    """Walk-Forward 完整结果:逐窗口 + 拼接样本外净值 + 整体指标。"""

    windows: list[WFWindow] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: Metrics | None = None
    target: str = "sharpe"
    full_start: date | None = None
    full_end: date | None = None

    def to_dict(self) -> dict[str, Any]:
        eq = self.equity_curve
        md = self.metrics.to_dict() if self.metrics is not None else None
        return {
            "target": self.target,
            "full_start": str(self.full_start) if self.full_start else None,
            "full_end": str(self.full_end) if self.full_end else None,
            "windows": [w.to_dict() for w in self.windows],
            "equity_curve": (
                {
                    "index": [str(d) for d in eq.index],
                    "values": [_safe_f(x) for x in eq],
                }
                if len(eq)
                else None
            ),
            "metrics": _clean_metrics(md),
        }


def _sort_key(target: str) -> Callable[[dict[str, Any]], float]:
    """按 sweep 语义的目标排序键(REVERSE_MIN_TARGETS 升序,其余降序;NaN 兜底)。"""
    reverse = target not in REVERSE_MIN_TARGETS
    nan_val = float("-inf") if reverse else float("inf")

    def key(r: dict[str, Any]) -> float:
        v = r.get(target)
        if v is None:
            return nan_val
        try:
            f = float(v)
        except (TypeError, ValueError):
            return nan_val
        return f if math.isfinite(f) else nan_val

    return key


def _metrics_slice(best: dict[str, Any], target: str) -> dict[str, Any]:
    """IS 最优组合的可展示指标(含 target)。"""
    out: dict[str, Any] = {k: best.get(k) for k in _METRIC_KEYS}
    out["target"] = best.get(target)
    return out


def _metrics_from_report(report: Any) -> dict[str, Any]:
    """OOS 段 Report → 可展示指标 dict。"""
    m = report.metrics
    return {
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "calmar": m.calmar,
        "total_return": m.total_return,
        "annual_return": m.annual_return,
        "max_drawdown": m.max_drawdown,
        "volatility": m.annual_volatility,
        "n_trades": m.n_trades,
    }


def _build_windows(
    trading_days: pd.DatetimeIndex, wf: WalkForwardConfig
) -> list[tuple[date, date, date, date]]:
    """在交易日索引上滚动:IS=[i:i+is],OOS=[i+is:i+is+oos];步长 step(默认 oos)。

    v1 仅支持非重叠窗口(``step == oos_days``):重叠窗口的段拼接与统计独立性
    更复杂,显式拒绝。
    """
    step = wf.step or wf.oos_days
    if step != wf.oos_days:
        raise ValueError("v1 仅支持非重叠窗口(step == oos_days)")
    n = (len(trading_days) - wf.is_days - wf.oos_days) // step + 1
    if n <= 0:
        raise ValueError(
            f"全区间({len(trading_days)} 个交易日)不足以容纳 "
            f"is_days={wf.is_days} + oos_days={wf.oos_days} 的窗口"
        )
    if wf.n_windows is not None:
        n = min(n, wf.n_windows)
    out: list[tuple[date, date, date, date]] = []
    for i in range(n):
        a = i * step
        is_ = trading_days[a : a + wf.is_days]
        oos = trading_days[a + wf.is_days : a + wf.is_days + wf.oos_days]
        out.append((is_[0].date(), is_[-1].date(), oos[0].date(), oos[-1].date()))
    return out


def _warmup_start(cfg: BacktestConfig, window_start: date, warmup_days: int) -> date:
    """把「窗口起点 - 暖机交易日」换算成取数起点(自然日),不早于全区间起点。"""
    if warmup_days <= 0:
        return window_start
    cal = window_start - timedelta(days=int(warmup_days * 1.6) + 30)  # 交易日→自然日
    return max(cfg.period.start, cal)


def _full_trading_days(
    cfg: BacktestConfig, registry: ProviderRegistry, market: Any
) -> pd.DatetimeIndex:
    """全区间交易日:基准日历优先,否则标的并集日历。"""
    if cfg.universe.benchmark:
        from djinn.data import load_benchmark

        try:
            bm = load_benchmark(
                registry,
                cfg.universe.benchmark,
                cfg.period.start,
                cfg.period.end,
                market=market,
                adjust=cfg.adjust,
            )
            idx = pd.DatetimeIndex(bm.df.index)
            if len(idx):
                return idx.sort_values()
        except Exception as e:
            _log.warning("基准日历不可用,退化为标的并集日历: %s", e)
    syms = _resolve_universe_symbols(cfg, registry, market)
    idx = pd.DatetimeIndex([])
    for s in syms:
        md = registry.get_ohlcv(
            s, cfg.period.start, cfg.period.end, cfg.adjust, market=market
        )
        idx = idx.union(pd.DatetimeIndex(md.df.index))
    idx = idx.sort_values()
    if len(idx) == 0:
        raise ValueError("无法解析全区间交易日历")
    return idx


def _prefetch(cfg: BacktestConfig, registry: ProviderRegistry, market: Any) -> None:
    """预拉全区间行情入缓存(之后所有窗口的暖机 / IS / OOS 取数均为缓存切片)。"""
    syms = _resolve_universe_symbols(cfg, registry, market)
    workers = int(os.environ.get("DJINN_FETCH_WORKERS", "8"))

    def _fetch(sym: str) -> None:
        registry.get_ohlcv(
            sym, cfg.period.start, cfg.period.end, cfg.adjust, market=market
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_fetch, syms))


def _window_config(cfg: BacktestConfig, start: date, end: date) -> BacktestConfig:
    """cfg 深拷贝,period 改为窗口区间(暖机起点只用于取数,不进 period)。"""
    c = cfg.model_copy(deep=True)
    c.period.start = start
    c.period.end = end
    return c


def _stitch(segments: list[pd.Series]) -> pd.Series:
    """按段首净值归一化后顺序拼接(复利延续),得到整条样本外净值。

    非重叠窗口下各段在时间轴上不相交、连续,拼接天然连续。
    """
    if not segments:
        return pd.Series(dtype=float)
    parts: list[pd.Series] = []
    running = 1.0
    for seg in segments:
        s = seg.dropna()
        if len(s) == 0:
            continue
        v = (s / float(s.iloc[0])) * running
        parts.append(v)
        running = float(v.iloc[-1])
    if not parts:
        return pd.Series(dtype=float)
    return cast(pd.Series, pd.concat(parts))


def walk_forward(
    cfg: BacktestConfig,
    *,
    registry: ProviderRegistry | None = None,
    grid: dict[str, list[Any]] | None = None,
    target: str | None = None,
    warmup_days: int | None = None,
    parallel: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> WalkForwardReport:
    """执行 walk-forward 分析:逐窗口 IS 独立选参 + OOS 评估,拼接样本外净值。

    ``grid`` / ``target`` 覆盖 ``cfg.walk_forward.grid`` / ``.target``(CLI/API 优先)。
    ``warmup_days`` 覆盖配置值。``parallel`` 并行化每窗口的 IS 组合(共享缓存线程安全,
    ``_run_one`` 已深拷贝 cfg)。``on_progress`` 每完成一窗口回调 ``(done, total)``。
    ``should_stop`` 为协作式取消回调:返回 True 抛 :class:`BacktestCancelled`(E4)。
    """
    wf = cfg.walk_forward
    if wf is None:
        raise ValueError("配置缺少 walk_forward 段")
    grid = grid or wf.grid
    if not grid:
        raise ValueError("walk_forward.grid 为空,无法做 IS 优化")
    if registry is None:
        registry = default_registry()
    market = cfg.resolved_market()
    warmup_days = wf.warmup_days if warmup_days is None else warmup_days
    target = wf.target if target is None else target

    trading_days = _full_trading_days(cfg, registry, market)
    windows = _build_windows(trading_days, wf)
    combos = _expand_grid(grid)
    _prefetch(cfg, registry, market)
    _log.info(
        "walk-forward:%d 个窗口、%d 种组合,目标=%s,暖机=%d 交易日",
        len(windows),
        len(combos),
        target,
        warmup_days,
    )

    segments: list[pd.Series] = []
    out: list[WFWindow] = []
    for wno, (is_s, is_e, oos_s, oos_e) in enumerate(windows, 1):
        if should_stop is not None and should_stop():
            raise BacktestCancelled(f"walk-forward 已取消 @窗口 {wno}")
        w = WFWindow(
            no=wno,
            is_start=is_s,
            is_end=is_e,
            oos_start=oos_s,
            oos_end=oos_e,
        )
        # IS:按窗口独立选参(带暖机)
        is_cfg = _window_config(cfg, is_s, is_e)
        is_warmup = _warmup_start(cfg, is_s, warmup_days)

        def _is_one(
            c: dict[str, Any], _cfg: BacktestConfig = is_cfg, _warm: date = is_warmup
        ) -> dict[str, Any]:
            return _run_one(_cfg, registry, dict(c), target, warmup_start=_warm)

        if parallel and len(combos) > 1:
            workers = int(os.environ.get("DJINN_FETCH_WORKERS", "8"))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_is_one, combos))
        else:
            results = [_is_one(c) for c in combos]
        results.sort(key=_sort_key(target), reverse=target not in REVERSE_MIN_TARGETS)
        best = results[0]
        w.is_metrics = _metrics_slice(best, target)

        # IS 达标才部署:目标缺失 / 非有限 / 低于门槛 → 该窗口不部署(OOS 空仓)
        target_val = best.get(target)
        deployed = True
        if wf.min_is_sharpe is not None:
            try:
                tv = float(target_val) if target_val is not None else None
            except (TypeError, ValueError):
                tv = None
            if tv is None or not math.isfinite(tv) or tv < wf.min_is_sharpe:
                deployed = False
        if not deployed:
            _log.info(
                "窗口 %d IS %s=%s 未达标(门槛 %s),不部署",
                wno,
                target,
                target_val,
                wf.min_is_sharpe,
            )
            out.append(w)
            if on_progress is not None:
                on_progress(wno, len(windows))
            continue

        w.best_params = dict(best["params"])
        w.deployed = True
        # OOS:用 IS 最优参数评估(带暖机)
        oos_cfg = _window_config(cfg, oos_s, oos_e)
        for k, v in w.best_params.items():
            _apply_param(oos_cfg, k, v)
        oos_warmup = _warmup_start(cfg, oos_s, warmup_days)
        run = run_backtest(oos_cfg, registry=registry, warmup_start=oos_warmup)
        eq = run.report.equity_curve
        w.oos_equity = eq
        w.oos_metrics = _metrics_from_report(run.report)
        out.append(w)
        if len(eq):
            segments.append(eq)
        _log.info(
            "窗口 %d 完成:OOS %s~%s,params=%s,oos_sharpe=%s",
            wno,
            oos_s,
            oos_e,
            w.best_params,
            w.oos_metrics.get("sharpe") if w.oos_metrics else None,
        )
        if on_progress is not None:
            on_progress(wno, len(windows))

    equity = _stitch(segments)
    metrics = (
        compute_metrics(equity, [], rf=cfg.risk_free_rate, market=market.value)
        if len(equity)
        else None
    )
    return WalkForwardReport(
        windows=out,
        equity_curve=equity,
        metrics=metrics,
        target=target,
        full_start=cfg.period.start,
        full_end=cfg.period.end,
    )

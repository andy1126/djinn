"""回测报告磁盘缓存:把完整序列化报告落到 ``.cache/djinn_results/{job_id}.json``。

目的:``/backtests/{id}/report`` 与 ``/backtests/{id}/export`` 不再每次重跑回测
(原简化实现里这两个端点都调 ``run_backtest(with_attribution=True)``,大宇宙回测下
极慢且归因网络开销翻倍)。后台任务完成后调 :func:`save` 落盘;两个读端点直接
:func:`load`。

序列化口径与原 ``routers/backtests.py`` 内联实现一致:
- Series  → ``{"index":[str], "values":[float|null]}``
- DataFrame → ``{"index":[str], "columns":[str], "data":[[...]]}``
- NaN/Inf → None(JSON 不接受);date/DateTime → isoformat
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any

import pandas as pd

from djinn.analytics.report import Report

CACHE_DIR = Path(".cache/djinn_results")


def _jsonable(v: object) -> bool:
    """判断值是否 JSON 可序列化(用于过滤)。"""
    if v is None or isinstance(v, (str, int, bool)):
        return True
    if isinstance(v, float):
        return math.isfinite(v)
    import json

    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _sanitize(v: object) -> Any:
    """把 NaN/Inf 转 None(float),其余递归清洗;JSON 默认编码不接受 NaN/Inf。"""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, list):
        return [_sanitize(x) for x in v]
    if isinstance(v, tuple):
        return [_sanitize(x) for x in v]
    if isinstance(v, dict):
        return {k: _sanitize(val) for k, val in v.items()}
    return v


def _safe_float(v: object) -> float | None:
    """Series value → finite float,否则 None。"""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _trade_val(v: object) -> Any:
    """单笔交易字段值的 JSON 友好化:date/DateTime → isoformat, NaN/Inf → None。"""
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, list):
        return [_trade_val(x) for x in v]
    return v


def _dictify(obj: object) -> dict[str, Any]:
    """把 dataclass/对象转为 dict,过滤非 JSON 可序列化字段,NaN/Inf → None。"""
    if obj is None:
        return {}
    raw: dict[str, Any]
    if isinstance(obj, dict):
        raw = obj
    elif hasattr(obj, "__dict__"):
        raw = dict(vars(obj))
    else:
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if not _jsonable(v):
            continue
        if isinstance(v, float):
            out[k] = v if math.isfinite(v) else None
        else:
            out[k] = v
    return out


def _series_to_list(s: pd.Series | None) -> dict[str, Any]:
    if s is None or len(s) == 0:
        return {"index": [], "values": []}
    return {
        "index": [str(x) for x in s.index],
        "values": [_safe_float(x) for x in s.values],
    }


def _df_to_dict(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "index": [],
            "columns": list(df.columns) if df is not None else [],
            "data": [],
        }
    return {
        "index": [str(x) for x in df.index],
        "columns": list(df.columns),
        "data": [[_sanitize(v) for v in row] for row in df.values.tolist()],
    }


def serialize_report(report: Report) -> dict[str, Any]:
    """把 :class:`Report` 序列化为 JSON 友好的报告 payload(供落盘 + 直传前端)。"""
    import dataclasses

    trades_out: list[dict[str, Any]] = []
    for t in report.trades:
        if dataclasses.is_dataclass(t) and not isinstance(t, type):
            inst = dataclasses.asdict(t)
            trades_out.append({k: _trade_val(v) for k, v in inst.items()})
        elif isinstance(t, dict):
            trades_out.append(_sanitize(t))
    return {
        "symbols": report.symbols,
        "metrics": _dictify(report.metrics),
        "trade_stats": _dictify(report.trade_stats),
        "benchmark_stats": (
            _dictify(report.benchmark_stats)
            if report.benchmark_stats is not None
            else None
        ),
        "equity_curve": _series_to_list(report.equity_curve),
        "benchmark_curve": _series_to_list(report.benchmark_curve),
        "drawdown_curve": _series_to_list(report.drawdown_curve),
        "monthly_returns": _df_to_dict(report.monthly_returns),
        "yearly_returns": _series_to_list(report.yearly_returns),
        "rolling_sharpe": _series_to_list(report.rolling_sharpe),
        "rolling_volatility": _series_to_list(report.rolling_volatility),
        "trades": trades_out,
        "rejections": [_dictify(r) for r in report.rejections],
        # D5:positions/weights 稀疏化(变动行),降磁盘 JSON 体积
        "positions": _df_to_sparse(report.positions),
        "weights": _df_to_sparse(report.weights),
        # 价格序列(收盘价,每日变化,用稠密格式)
        "prices": _df_to_dict(report.prices),
        "attribution": report.attribution,
        "factor_exposure": report.factor_exposure,
        "v": 2,
    }


def _path(job_id: str) -> Path:
    return CACHE_DIR / f"{job_id}.json"


def save(job_id: str, payload: dict[str, Any]) -> Path:
    """落盘报告 payload。"""
    import json

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(job_id)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def load(job_id: str) -> dict[str, Any] | None:
    """读取报告 payload;不存在返回 None。"""
    import json

    p = _path(job_id)
    if not p.exists():
        return None
    try:
        loaded: Any = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def exists(job_id: str) -> bool:
    return _path(job_id).exists()


def _series_from_list(d: dict[str, Any]) -> pd.Series:
    """``{"index":[str], "values":[float|null]}`` → pandas Series(index=Date)。"""
    idx = d.get("index", [])
    vals = d.get("values", [])
    dt_idx = pd.to_datetime(idx, errors="coerce") if idx else pd.DatetimeIndex([])
    return pd.Series(vals, index=dt_idx, dtype="float64")


def _df_from_dict(d: dict[str, Any]) -> pd.DataFrame:
    """``{"index":[str], "columns":[str], "data":[[...]]}`` → DataFrame。"""
    idx = d.get("index", [])
    cols = d.get("columns", [])
    data = d.get("data", [])
    dt_idx = pd.to_datetime(idx, errors="coerce") if idx else pd.DatetimeIndex([])
    return pd.DataFrame(data, index=dt_idx, columns=cols)


def _df_to_sparse(df: pd.DataFrame | None) -> dict[str, Any]:
    """稠密 DataFrame → 稀疏变动行(D5):``{"dates":[str], "rows":[{"date","values"}]}``。

    仅输出非零值发生变化的行 + 首行;持仓/权重面板大量 0,稀疏后 JSON 体积大幅下降。
    """
    if df is None or df.empty:
        return {"dates": [], "rows": []}
    rows_out: list[dict[str, Any]] = []
    prev: dict[str, float] | None = None
    for ts, row in df.iterrows():
        cur = {str(c): float(row[c]) for c in df.columns if float(row[c]) != 0.0}
        if prev is None or cur != prev:
            rows_out.append({"date": str(ts), "values": cur})
            prev = cur
    return {"dates": [r["date"] for r in rows_out], "rows": rows_out}


def _df_from_sparse(
    d: dict[str, Any], full_index: pd.DatetimeIndex, columns: list[str]
) -> pd.DataFrame:
    """稀疏 → 稠密(D5):reindex 到全量日历 + ffill + fillna(0)。"""
    rows = d.get("rows", [])
    if not rows:
        return pd.DataFrame(0.0, index=full_index, columns=columns)
    idx = pd.to_datetime([r["date"] for r in rows])
    col_pos = {c: i for i, c in enumerate(columns)}
    n = len(idx)
    data_arr = [[0.0] * n for _ in columns]
    for i, r in enumerate(rows):
        for sym, val in r["values"].items():
            p = col_pos.get(sym)
            if p is not None:
                data_arr[p][i] = float(val)
    df = pd.DataFrame({c: data_arr[col_pos[c]] for c in columns}, index=idx)
    return df.reindex(full_index).ffill().fillna(0.0)


def rebuild_report(payload: dict[str, Any]) -> Report:
    """把 :func:`serialize_report` 的 payload 还原为 :class:`Report`(供导出复用,
    避免在 ``/export`` 端点重跑回测)。``metrics`` / ``trade_stats`` /
    ``benchmark_stats`` 用 dict 占位——:func:`djinn.io.export.export_csv` /
    :func:`djinn.io.export.export_excel` 仅调它们的 ``to_dict()``,故包一层实现该
    方法即可。"""
    from djinn.analytics.report import Report

    class _DictLike:
        """以 dict 为底、暴露 ``to_dict()`` 的最小适配器。"""

        def __init__(self, d: dict[str, Any]) -> None:
            self._d = d

        def to_dict(self) -> dict[str, Any]:
            return self._d

    reject_raw = payload.get("rejections") or []
    rejections = (
        [_Plain(**r) for r in reject_raw]
        if reject_raw and isinstance(reject_raw[0], dict)
        else reject_raw
    )
    # D5:positions/weights 稀疏(v2)时用 equity 全量日历 + symbols 重建稠密
    equity_series = _series_from_list(payload.get("equity_curve") or {})
    full_index = pd.DatetimeIndex(equity_series.index)
    symbols = [str(s) for s in (payload.get("symbols") or [])]
    if payload.get("v") == 2:
        positions_df = _df_from_sparse(
            payload.get("positions") or {}, full_index, symbols
        )
        weights_df = _df_from_sparse(payload.get("weights") or {}, full_index, symbols)
    else:
        positions_df = _df_from_dict(payload.get("positions") or {})
        weights_df = _df_from_dict(payload.get("weights") or {})
    return Report(
        metrics=_DictLike(payload.get("metrics") or {}),  # type: ignore[arg-type]
        trade_stats=_DictLike(payload.get("trade_stats") or {}),  # type: ignore[arg-type]
        benchmark_stats=(
            _DictLike(payload.get("benchmark_stats") or {})  # type: ignore[arg-type]
            if payload.get("benchmark_stats") is not None
            else None
        ),
        equity_curve=_series_from_list(payload.get("equity_curve") or {}),
        benchmark_curve=(
            _series_from_list(payload.get("benchmark_curve") or {})
            if payload.get("benchmark_curve")
            else None
        ),
        drawdown_curve=_series_from_list(payload.get("drawdown_curve") or {}),
        monthly_returns=_df_from_dict(payload.get("monthly_returns") or {}),
        yearly_returns=_series_from_list(payload.get("yearly_returns") or {}),
        rolling_sharpe=_series_from_list(payload.get("rolling_sharpe") or {}),
        rolling_volatility=_series_from_list(payload.get("rolling_volatility") or {}),
        trades=_trades_from_list(payload.get("trades") or []),
        rejections=rejections,
        positions=positions_df,
        weights=weights_df,
        prices=_df_from_dict(payload.get("prices") or {}),
        symbols=payload.get("symbols") or [],
        attribution=payload.get("attribution"),
        factor_exposure=payload.get("factor_exposure"),
    )


def densify_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """把 v2(稀疏)positions/weights 还原为稠密 DataFrameData,供前端 /report 直读。

    磁盘缓存存稀疏(降体积);前端 PositionAreaChart/WeightHeatmap 消费稠密
    ``{index, columns, data}`` 格式,故 /report 返回前先 densify。
    """
    if payload.get("v") != 2:
        return payload
    equity = _series_from_list(payload.get("equity_curve") or {})
    full_index = pd.DatetimeIndex(equity.index)
    symbols = [str(s) for s in (payload.get("symbols") or [])]
    out = dict(payload)
    out["positions"] = _df_to_dict(
        _df_from_sparse(payload.get("positions") or {}, full_index, symbols)
    )
    out["weights"] = _df_to_dict(
        _df_from_sparse(payload.get("weights") or {}, full_index, symbols)
    )
    return out


def _trades_from_list(trades: list[dict[str, Any]]) -> list[Any]:
    """导出端点用 ``trades_to_df`` 取 ``getattr(t, field)``;这里还原成简单对象。"""
    return [_Plain(**t) for t in trades]


class _Plain:
    """属性容器(供 :func:`djinn.io.export.trades_to_df` 用 ``getattr`` 取值)。"""

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def delete(job_id: str) -> None:
    """删除报告缓存(忽略不存在)。"""
    from contextlib import suppress

    p = _path(job_id)
    with suppress(OSError):
        if p.exists():
            p.unlink()

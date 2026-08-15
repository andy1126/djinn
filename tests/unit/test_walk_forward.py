"""H 计划 Walk-Forward 分析测试:窗口几何 / IS 独立选参 / 门槛 / 拼接 / 序列化。

用确定性 stub provider(线性 + 正弦行情)合成数据,不触网。
"""

from __future__ import annotations

import itertools
import json
import math
from datetime import date

import pandas as pd
import pytest

from djinn.cli.walk_forward import _build_windows, walk_forward
from djinn.config import load_config
from djinn.config.models import BacktestConfig, WalkForwardConfig
from djinn.data.market_data import MarketData
from djinn.data.provider import DataProvider, ProviderRegistry
from djinn.data.schema import Adjust, Market
from djinn.utils.exceptions import ConfigError

_SYMBOL = "AAPL"


def _synth_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    """线性趋势 + 正弦波动:不同均线窗口在不同相位会有不同表现,网格可区分。"""
    idx = pd.bdate_range(start, end)
    n = len(idx)
    closes = [100.0 + 0.08 * i + 6.0 * math.sin(i / 12.0) for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1.0e6] * n,
        },
        index=idx,
    )


class _StubProvider(DataProvider):
    name = "stub"
    market = Market.US

    def supports(self, symbol: str, market: Market | None = None) -> bool:
        return True

    def get_ohlcv(
        self, symbol: str, start: date, end: date, adjust: Adjust = Adjust.BACKWARD
    ) -> MarketData:
        return MarketData(
            symbol=symbol,
            market=Market.US,
            df=_synth_ohlcv(symbol, start, end),
            adjust=adjust,
        )


_stub_registry = ProviderRegistry([_StubProvider()])

_GRID = {"fast": [5, 10], "slow": [20, 30]}
_START = "2020-01-01"
_END = "2021-12-31"


def _cfg(
    *,
    is_days: int = 126,
    oos_days: int = 63,
    min_is_sharpe: float | None = None,
    grid: dict | None = None,
) -> BacktestConfig:
    return BacktestConfig(
        universe={"symbols": [_SYMBOL]},
        period={"start": _START, "end": _END},
        account={"initial_cash": 100000, "currency": "USD"},
        strategy={"name": "MACrossover", "params": {}},
        walk_forward={
            "is_days": is_days,
            "oos_days": oos_days,
            "min_is_sharpe": min_is_sharpe,
            "grid": grid or _GRID,
        },
    )


# ── 窗口几何 ───────────────────────────────────────────
def test_build_windows_geometry() -> None:
    """窗口不重叠、相邻首尾相接、个数按区间推导。"""
    idx = pd.bdate_range("2020-01-01", periods=300)
    wf = WalkForwardConfig(is_days=100, oos_days=60, grid=_GRID)
    windows = _build_windows(idx, wf)  # n=(300-160)//60+1=3
    assert len(windows) == 3
    for is_s, is_e, oos_s, oos_e in windows:
        assert is_s < is_e < oos_s < oos_e
    # OOS 段连续:上一段 oos_end 的下一个业务日 == 下一段 oos_start(非重叠、无缝)
    for a, b in itertools.pairwise(windows):
        nxt = idx[idx > pd.Timestamp(a[3])][0].date()
        assert nxt == b[2]


def test_build_windows_step_must_equal_oos() -> None:
    """v1 仅支持非重叠(step == oos_days),重叠显式拒绝。"""
    wf = WalkForwardConfig(is_days=100, oos_days=60, step=30, grid=_GRID)
    with pytest.raises(ValueError, match="非重叠"):
        _build_windows(pd.bdate_range("2020-01-01", periods=300), wf)


def test_build_windows_insufficient_range() -> None:
    """区间不足以容纳一个 IS+OOS 窗口 → 报错。"""
    wf = WalkForwardConfig(is_days=200, oos_days=100, grid=_GRID)
    with pytest.raises(ValueError, match="不足以容纳"):
        _build_windows(pd.bdate_range("2020-01-01", periods=200), wf)


# ── 端到端 walk_forward ────────────────────────────────
def test_walk_forward_end_to_end() -> None:
    """全流程:每窗口 IS 独立选参 + OOS 评估,拼接样本外净值。"""
    cfg = _cfg()
    report = walk_forward(cfg, registry=_stub_registry)

    assert len(report.windows) >= 2
    deployed = [w for w in report.windows if w.deployed]
    assert len(deployed) >= 1
    for w in deployed:
        # IS 最优参数 ∈ 网格组合(按窗口独立选参)
        assert w.best_params in [
            {"fast": f, "slow": s} for f in _GRID["fast"] for s in _GRID["slow"]
        ]
        assert w.is_metrics is not None and w.is_metrics["target"] is not None
        assert w.oos_metrics is not None
        assert w.oos_equity is not None and len(w.oos_equity) > 0
        # OOS 净值段落在窗口内
        assert w.oos_equity.index.min() >= pd.Timestamp(w.oos_start)
        assert w.oos_equity.index.max() <= pd.Timestamp(w.oos_end)

    # 拼接曲线长度 = 各部署段长度之和,且无前视(整体连续)
    total = sum(len(w.oos_equity) for w in deployed if w.oos_equity is not None)
    assert len(report.equity_curve) == total
    assert report.metrics is not None
    assert math.isfinite(report.metrics.sharpe)


def test_is_best_params_used_in_oos(monkeypatch: pytest.MonkeyPatch) -> None:
    """核心:每个窗口 OOS 用的是**该窗口** IS 最优参数(按窗口独立选参)。"""
    import djinn.cli.walk_forward as wfmod

    recorded: list[dict] = []
    real = wfmod.run_backtest

    def spy(oos_cfg: BacktestConfig, **kw: object) -> object:
        recorded.append(dict(oos_cfg.strategy.params))
        return real(oos_cfg, **kw)

    monkeypatch.setattr(wfmod, "run_backtest", spy)
    report = wfmod.walk_forward(_cfg(), registry=_stub_registry)

    oos_used = [w for w in report.windows if w.deployed]
    assert len(oos_used) == len(recorded)
    for w, params in zip(oos_used, recorded, strict=False):
        assert params == w.best_params


def test_min_is_sharpe_gate_skips_deployment() -> None:
    """IS 目标低于门槛 → 该窗口不部署(OOS 空仓),拼接曲线为空。"""
    cfg = _cfg(min_is_sharpe=1.0e9)  # 不可能达标
    report = walk_forward(cfg, registry=_stub_registry)
    assert len(report.windows) >= 1
    assert all(not w.deployed for w in report.windows)
    assert all(w.best_params is None and w.oos_equity is None for w in report.windows)
    assert len(report.equity_curve) == 0
    assert report.metrics is None


def test_report_serialization() -> None:
    """WalkForwardReport.to_dict() 可 JSON 序列化(每窗口指标均为标量)。"""
    report = walk_forward(_cfg(), registry=_stub_registry)
    d = report.to_dict()
    assert d["target"] == "sharpe"
    assert isinstance(d["windows"], list) and len(d["windows"]) >= 1
    assert d["metrics"] is not None
    text = json.dumps(d, ensure_ascii=False, default=str)
    assert "windows" in json.loads(text)


# ── 配置解析 ───────────────────────────────────────────
def test_walk_forward_config_parses() -> None:
    """walk_forward 段 YAML 解析 + 校验;env 覆盖自动生效。"""
    data = {
        "universe": {"symbols": [_SYMBOL]},
        "period": {"start": _START, "end": _END},
        "strategy": {"name": "MACrossover", "params": {}},
        "walk_forward": {"is_days": 126, "oos_days": 63, "grid": _GRID},
    }
    cfg = load_config(data=data)
    assert cfg.walk_forward is not None
    assert cfg.walk_forward.is_days == 126
    assert cfg.walk_forward.grid == _GRID

    bad = dict(data)
    bad["walk_forward"] = {"is_days": 0, "oos_days": 63, "grid": _GRID}
    with pytest.raises(ConfigError):
        load_config(data=bad)

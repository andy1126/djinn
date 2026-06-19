"""引擎端到端集成测试(配置 → 引擎)。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from djinn.cli.runner import run_backtest
from djinn.config import load_config
from djinn.data import CSVProvider, DataCache, default_registry
from djinn.data.schema import Market
from djinn.engine import EngineConfig, EventDrivenEngine
from djinn.engine.commission import USCommissionModel
from djinn.engine.slippage import ZeroSlippage
from djinn.strategy import MACrossover


def test_engine_single_symbol_money_conservation(make_csv, tmp_csv_dir: Path):
    """单标的:回测全程资金守恒。"""
    make_csv("AAPL", periods=120, drift=0.001, vol=0.015, seed=0)
    prov = CSVProvider(tmp_csv_dir, default_market=Market.US)
    md = prov.get_ohlcv("AAPL", date(2024, 1, 2), date(2024, 6, 30))
    strat = MACrossover(fast=10, slow=30)
    cfg = EngineConfig(
        initial_cash=100000, commission=USCommissionModel(), slippage=ZeroSlippage()
    )
    result = EventDrivenEngine(cfg).run(strat, {"AAPL": md})
    # 末态资金守恒
    last_price = float(md.df["close"].iloc[-1])
    result.account.check_invariant({"AAPL": last_price})
    assert len(result.equity_curve) == len(md)


def test_run_backtest_full_pipeline(make_csv, tmp_csv_dir: Path, tmp_path: Path):
    """run_backtest 端到端:配置 → 数据 → 引擎 → 报告 → 导出。"""
    make_csv("AAPL", periods=120, drift=0.001, vol=0.015, seed=0)
    data = {
        "universe": {"symbols": ["AAPL"], "market": "US"},
        "period": {"start": "2024-01-02", "end": "2024-06-30"},
        "account": {"initial_cash": 50000},
        "strategy": {"name": "MACrossover", "params": {"fast": 10, "slow": 30}},
        "output": {"dir": str(tmp_path / "out"), "export": ["csv", "excel"]},
    }
    cfg = load_config(data=data)
    cache = DataCache()
    registry = default_registry(csv_dir=str(tmp_csv_dir), cache=cache)
    result = run_backtest(cfg, registry=registry, cache=cache)
    assert result.report.metrics.n_days > 0
    assert len(result.exported_files) > 0
    # 指标已计算
    m = result.report.metrics
    assert m.n_trades >= 0
    # 导出文件存在
    for f in result.exported_files:
        assert f.exists()


def test_engine_benchmark_curve_loaded(make_csv, tmp_csv_dir: Path):
    """基准曲线被加载并归一化到策略起点。"""
    make_csv("AAPL", periods=60, seed=1)
    make_csv("BENCH", periods=60, seed=2)
    prov = CSVProvider(tmp_csv_dir, default_market=Market.US)
    md = prov.get_ohlcv("AAPL", date(2024, 1, 2), date(2024, 3, 27))
    bench = prov.get_ohlcv("BENCH", date(2024, 1, 2), date(2024, 3, 27))
    strat = MACrossover(fast=5, slow=15)
    cfg = EngineConfig(
        initial_cash=100000, commission=USCommissionModel(), slippage=ZeroSlippage()
    )
    result = EventDrivenEngine(cfg).run(strat, {"AAPL": md}, benchmark=bench)
    assert result.benchmark_curve is not None
    # 基准起点 = 策略起点
    assert result.benchmark_curve.iloc[0] == pytest.approx(
        result.equity_curve.iloc[0], rel=1e-6
    )

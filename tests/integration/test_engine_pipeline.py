"""引擎端到端集成测试(配置 → 引擎)。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from djinn.analytics.trades import compare_benchmark
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


def test_presignal_equivalence(make_csv, tmp_csv_dir: Path):
    """D1:signals 预计算路径与慢路径的 fills/equity 逐值一致。"""
    make_csv("AAPL", periods=120, seed=0)
    prov = CSVProvider(tmp_csv_dir, default_market=Market.US)
    md = prov.get_ohlcv("AAPL", date(2024, 1, 2), date(2024, 6, 30))
    cfg = EngineConfig(
        initial_cash=100000, commission=USCommissionModel(), slippage=ZeroSlippage()
    )

    fast = MACrossover(fast=10, slow=30)
    r_fast = EventDrivenEngine(cfg).run(fast, {"AAPL": md})

    slow = MACrossover(fast=10, slow=30)
    slow.precompute_signals = False  # 强制慢路径
    r_slow = EventDrivenEngine(cfg).run(slow, {"AAPL": md})

    assert len(r_fast.trades) == len(r_slow.trades)
    for f1, f2 in zip(r_fast.trades, r_slow.trades, strict=True):
        assert f1.symbol == f2.symbol
        assert f1.qty == f2.qty
        assert f1.price == f2.price
    pd.testing.assert_series_equal(
        r_fast.equity_curve, r_slow.equity_curve, check_exact=True
    )


def test_benchmark_starts_late(make_csv, tmp_csv_dir: Path):
    """基准数据起点晚于策略首日时,基准曲线无前导 NaN(bfill 回填)。

    修复前:bm.iloc[0] 为 NaN → 整条基准曲线 NaN → compare_benchmark 静默全 0。
    修复后:基准曲线无 NaN,基准收益非 0。
    """
    make_csv("AAPL", start="2024-01-02", periods=60, seed=1)
    make_csv("BENCH", start="2024-03-01", periods=30, seed=2)
    prov = CSVProvider(tmp_csv_dir, default_market=Market.US)
    md = prov.get_ohlcv("AAPL", date(2024, 1, 2), date(2024, 3, 29))
    bench = prov.get_ohlcv("BENCH", date(2024, 3, 1), date(2024, 3, 29))
    strat = MACrossover(fast=5, slow=15)
    cfg = EngineConfig(
        initial_cash=100000, commission=USCommissionModel(), slippage=ZeroSlippage()
    )
    result = EventDrivenEngine(cfg).run(strat, {"AAPL": md}, benchmark=bench)
    assert result.benchmark_curve is not None
    # 关键断言:基准曲线无前导 NaN(修复前整条为 NaN)
    assert result.benchmark_curve.notna().all()
    # 基准起点归一化到策略起点(前导段视为持平)
    assert result.benchmark_curve.iloc[0] == pytest.approx(
        result.equity_curve.iloc[0], rel=1e-6
    )
    # compare_benchmark 不再静默全 0:基准收益非 0
    bs = compare_benchmark(result.equity_curve, result.benchmark_curve, market="US")
    assert bs.benchmark_return != 0.0

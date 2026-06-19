"""示例:单标的 MACrossover 回测(美股)。

用法:
    python examples/single_symbol_backtest.py
    # 或用 CLI:
    djinn run -c configs/backtest.example.yaml --csv-dir tests/fixtures/csv
"""

from __future__ import annotations

from datetime import date

from djinn import (
    BacktestConfig,
    EngineConfig,
    EventDrivenEngine,
    MACrossover,
    build_report,
    default_registry,
    load_config,
)
from djinn.engine.commission import USCommissionModel
from djinn.engine.slippage import FixedBpsSlippage


def main() -> None:
    # 方式一:从 YAML 配置加载(推荐,可复现)
    cfg = load_config("configs/backtest.example.yaml")
    registry = default_registry(csv_dir="tests/fixtures/csv")
    market = cfg.resolved_market()
    data = {
        sym: registry.get_ohlcv(sym, cfg.period.start, cfg.period.end, cfg.adjust, market=market)
        for sym in cfg.universe.symbols
    }

    # 方式二:直接构造引擎配置
    engine_cfg = EngineConfig(
        initial_cash=cfg.account.initial_cash,
        commission=USCommissionModel(),
        slippage=FixedBpsSlippage(5),
    )
    strategy = MACrossover(fast=10, slow=30)
    result = EventDrivenEngine(engine_cfg).run(strategy, data)

    report = build_report(result, market="US")
    m = report.metrics
    print(f"=== {', '.join(report.symbols)} MACrossover(10,30) ===")
    print(f"区间: {result.equity_curve.index[0].date()} ~ {result.equity_curve.index[-1].date()}")
    print(f"累计收益: {m.total_return:.2%}  年化: {m.annual_return:.2%}")
    print(f"夏普: {m.sharpe:.3f}  最大回撤: {m.max_drawdown:.2%}")
    print(f"成交: {m.n_trades} 笔")


if __name__ == "__main__":
    main()

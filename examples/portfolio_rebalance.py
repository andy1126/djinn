"""示例:等权组合 + 季度再平衡。

用法:
    python examples/portfolio_rebalance.py
"""

from __future__ import annotations

from datetime import date

from djinn import (
    EngineConfig,
    EventDrivenEngine,
    EqualWeight,
    Rebalancer,
)
from djinn.portfolio import RebalanceConfig
from djinn.data import default_registry
from djinn.engine.commission import USCommissionModel
from djinn.engine.slippage import FixedBpsSlippage
from djinn.strategy import MACrossover


def main() -> None:
    symbols = ["AAPL", "MSFT", "GOOGL"]
    registry = default_registry(csv_dir="tests/fixtures/csv")
    data = {
        s: registry.get_ohlcv(s, date(2020, 1, 1), date(2024, 12, 31))
        for s in symbols
    }

    cfg = EngineConfig(
        initial_cash=100000,
        commission=USCommissionModel(),
        slippage=FixedBpsSlippage(5),
        allocation=EqualWeight(),
        rebalance=Rebalancer(RebalanceConfig(period="quarterly", threshold=0.05)),
    )
    # MACrossover 在组合模式下对每个成分独立跑信号
    strategy = MACrossover(fast=10, slow=30)
    result = EventDrivenEngine(cfg).run(strategy, data)

    print(f"=== 等权组合 {symbols} + 季度再平衡 ===")
    print(f"成交: {len(result.trades)} 笔  拒单: {len(result.rejections)}")
    print(f"末态净值: {result.equity_curve.iloc[-1]:.2f}")
    print("末态权重:")
    for s in symbols:
        w = result.weights_curve.iloc[-1][s]
        print(f"  {s}: {w:.2%}")


if __name__ == "__main__":
    main()

"""示例:参数扫描求最优均线参数。

用法:
    python examples/parameter_sweep.py
    # 或 CLI:
    djinn sweep -c configs/sweep.example.yaml --grid '{"fast":[5,10,20],"slow":[20,30,60]}' --csv-dir tests/fixtures/csv
"""

from __future__ import annotations

import itertools
import json
from datetime import date

from djinn import EventDrivenEngine, EngineConfig, MACrossover
from djinn.analytics import build_report
from djinn.data import default_registry
from djinn.engine.commission import USCommissionModel
from djinn.engine.slippage import FixedBpsSlippage


def main() -> None:
    registry = default_registry(csv_dir="tests/fixtures/csv")
    md = registry.get_ohlcv("AAPL", date(2020, 1, 1), date(2024, 12, 31))
    data = {"AAPL": md}

    grid = {"fast": [5, 10, 20], "slow": [20, 30, 60]}
    keys = list(grid)
    results = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        strat = MACrossover(**params)
        cfg = EngineConfig(
            initial_cash=100000,
            commission=USCommissionModel(),
            slippage=FixedBpsSlippage(5),
        )
        result = EventDrivenEngine(cfg).run(strat, data)
        m = build_report(result, market="US").metrics
        results.append({"params": params, "sharpe": m.sharpe, "return": m.total_return, "mdd": m.max_drawdown})

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    print("=== 最优均线参数(按夏普)===")
    for r in results[:5]:
        print(f"  {r['params']}  夏普={r['sharpe']:.3f}  收益={r['return']:.2%}  回撤={r['mdd']:.2%}")
    print("\n完整结果:", json.dumps(results, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

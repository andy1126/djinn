"""前向收益:因子分析的"标签"。

``compute_forward_returns`` 给出 t 收盘 → t+N 收盘的持有收益(date × symbol),
用于 IC 与分层回测。注意:第 t 行的前向收益用到 t+N 的价格,**属于标签而非特征**,
绝不参与当日因子计算(因子侧已保证 point-in-time)。
"""

from __future__ import annotations

import pandas as pd


def compute_forward_returns(
    prices: pd.DataFrame, periods: list[int] | tuple[int, ...] = (1, 5, 10)
) -> dict[int, pd.DataFrame]:
    """多周期前向收益面板。

    Args:
        prices: 收盘价宽表(date × symbol)。
        periods: 持有期(交易日)列表。

    Returns:
        ``{period: DataFrame(date × symbol)}``,``fwd[t] = close[t+period]/close[t] - 1``;
        末尾 ``period`` 行无未来数据 → NaN。
    """
    out: dict[int, pd.DataFrame] = {}
    for p in periods:
        p = int(p)
        out[p] = prices.shift(-p) / prices - 1.0
    return out

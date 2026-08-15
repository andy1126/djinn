"""signals 因果性自检(D1)。

``Strategy.signals(data)`` 是**无状态纯函数**,引擎可预计算一次全量信号、每日
O(1) 查表(见 ``event_engine`` 的 ``_presignals``)。这要求 signals 是**因果运算**:
``t`` 日输出只依赖 ``≤ t`` 输入(rolling / ewm / shift(+1) 天然满足)。若用户
写了 ``shift(-1)`` / 全样本统计量(``df.mean()``)等依赖未来的运算,预计算与逐日
切片不等价,结果被未来函数污染。

:func:`check_causal` 随机取若干截断点,对比「截断序列末值」与「全量序列对应值」,
不等即报出非因果。引擎在 ``DJINN_DEBUG=1`` 下启动时对每个标的跑一次。
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, cast

import pandas as pd

SignalsFn = Callable[[pd.DataFrame], pd.Series]


def check_causal(
    signals_fn: SignalsFn, df: pd.DataFrame, n_probe: int = 5
) -> list[str]:
    """校验 ``signals_fn(df)`` 是因果运算。

    Args:
        signals_fn: 输入完整历史、输出信号 Series 的函数(即 ``strategy.signals``)。
        df: 单标的完整历史(``index=DatetimeIndex``)。
        n_probe: 随机截断点数(默认 5)。

    Returns:
        非因果的日期字符串列表(空 = 通过)。计算失败时不误报(返回空)。
    """
    if df.empty or len(df) < 2:
        return []
    try:
        full = signals_fn(df)
        if len(full) == 0:  # 含 signals 返回 None 的情形(None 无 len → 抛异常)
            return []
    except Exception:
        return []
    idx = list(df.index)
    rng = random.Random(0)  # 固定种子,结果可复现
    probes = sorted(rng.sample(idx[1:], min(n_probe, len(idx) - 1)))
    problems: list[str] = []
    for ts in probes:
        pos = cast(int, df.index.get_loc(ts))  # DatetimeIndex 唯一,get_loc 返回 int
        try:
            truncated = signals_fn(df.iloc[: pos + 1])
            if len(truncated) == 0:
                continue
        except Exception:
            continue
        v_trunc: Any = truncated.iloc[-1]
        v_full: Any = full.loc[ts]
        # 两值同为 NaN 视为一致;否则不相等即非因果
        both_nan = bool(pd.isna(v_trunc) and pd.isna(v_full))
        if not both_nan and v_trunc != v_full:
            problems.append(f"{ts}: 截断末值 {v_trunc!r} != 全量 {v_full!r}")
    return problems

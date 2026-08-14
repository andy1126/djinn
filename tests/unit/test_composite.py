"""C9:滚动 ICIR 加权合成器测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from djinn.factor.composite import composite_score, rolling_ic_weights


def _panels(seed: int = 0, n_days: int = 80, n_syms: int = 30):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    syms = [f"S{i}" for i in range(n_syms)]
    f1 = pd.DataFrame(rng.normal(0, 1, (n_days, n_syms)), index=idx, columns=syms)
    f2 = pd.DataFrame(rng.normal(0, 1, (n_days, n_syms)), index=idx, columns=syms)
    fwd = pd.DataFrame(rng.normal(0, 1, (n_days, n_syms)), index=idx, columns=syms)
    return f1, f2, fwd


def test_rolling_ic_weights_normalized() -> None:
    """权重面板每行 |w| 之和 ≈ 1(符号保留)。"""
    f1, f2, fwd = _panels()
    w = rolling_ic_weights(
        {"f1": f1, "f2": f2}, fwd, window=30, min_periods=10, shift_periods=1
    )
    assert list(w.columns) == ["f1", "f2"]
    for t in w.index[-10:]:
        total = float(w.loc[t].abs().sum())
        if total > 0:
            assert total == pytest.approx(1.0, abs=1e-9)


def test_composite_score_shape() -> None:
    """合成得分面板形状 = 权重日期 × 标的。"""
    f1, f2, fwd = _panels()
    w = rolling_ic_weights(
        {"f1": f1, "f2": f2}, fwd, window=30, min_periods=10, shift_periods=1
    )
    score = composite_score({"f1": f1, "f2": f2}, w)
    assert score.shape == (len(w.index), f1.shape[1])


def test_rolling_ic_weights_shift_moves_panel() -> None:
    """shift_periods 会右移 IC 序列(首日权重为 0,因 IC(-p) 不存在)。"""
    f1, f2, fwd = _panels()
    w = rolling_ic_weights(
        {"f1": f1, "f2": f2}, fwd, window=20, min_periods=5, shift_periods=2
    )
    # 前 2 行(shift 后 IC 为 NaN → fillna 0)权重应全 0
    assert w.iloc[0].abs().sum() == 0.0
    assert w.iloc[1].abs().sum() == 0.0

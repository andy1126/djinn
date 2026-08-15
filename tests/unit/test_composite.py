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


def test_icir_weights_pit() -> None:
    """关键 PIT(C9):因子在 t0 后才获得预测力,权重在 t0+p 前不反映(看不到未来)。

    因子 A 与 fwd 在 t>=t0 后同序(IC_A≈+1),t<t0 为噪声(IC_A≈0);因子 B 恒为
    噪声。``shift_periods=p`` 使 ``ic_effective(t)=ic(t−p)``:在 t<t0+p 时权重仍
    由 t0 之前的噪声 IC 决定(A/B 各半、无信息),直到 t>=t0+p+window 后 A 的
    权重才 →1。
    """
    n_days, n_syms, p, t0, window = 150, 30, 5, 60, 30
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    syms = [f"S{i}" for i in range(n_syms)]
    rng = np.random.default_rng(0)
    base = np.arange(n_syms, dtype=float)
    fa = pd.DataFrame(np.tile(base, (n_days, 1)), index=idx, columns=syms)
    fb = pd.DataFrame(rng.normal(0, 1, (n_days, n_syms)), index=idx, columns=syms)
    fwd = pd.DataFrame(rng.normal(0, 1, (n_days, n_syms)), index=idx, columns=syms)
    for t in idx[t0:]:
        fwd.loc[t] = base + rng.normal(0, 1.0, n_syms)  # 与 A 同序 → IC_A≈+1
    w = rolling_ic_weights(
        {"A": fa, "B": fb}, fwd, window=window, min_periods=10, shift_periods=p
    )
    before = float(w.loc[idx[: t0 + p], "A"].mean())
    after = float(w.loc[idx[t0 + p + window :], "A"].mean())
    # 充分晚后 A 权重 →1(滚动 ICIR 占优)
    assert after > 0.9
    # PIT:预测力在 t0 才出现,故 t0+p 之前的权重仍由噪声 IC 决定,远低于 after
    assert after - before > 0.4

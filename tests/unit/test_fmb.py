"""C8:Newey-West t 值 + Fama-MacBeth 回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from djinn.factor.analysis.fmb import fama_macbeth
from djinn.factor.analysis.ic import _newey_west_t, ic_summary


def test_newey_west_white_noise() -> None:
    """白噪声 IC 序列(零均值)的 NW t 值应不显著(|t| 小)。"""
    rng = np.random.default_rng(0)
    ic = pd.Series(rng.normal(0.0, 1.0, 500))
    assert abs(_newey_west_t(ic)) < 3.0


def test_newey_west_lags0_matches_classic() -> None:
    """lags=0 时 NW t 退化为经典 t 值(ddof=0,精确相等)。"""
    rng = np.random.default_rng(1)
    ic = pd.Series(rng.normal(0.05, 0.1, 200))
    t = _newey_west_t(ic, lags=0)
    classic = float(ic.mean() / (ic.std(ddof=0) / np.sqrt(len(ic))))
    assert t == pytest.approx(classic, rel=1e-9)


def test_newey_west_penalizes_autocorrelation() -> None:
    """正自相关 IC 序列:HAC t 值应小于经典 t 值(se 更宽,更保守)。"""
    rng = np.random.default_rng(4)
    n = 500
    eps = rng.normal(0, 1, n)
    x = np.zeros(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + eps[i]  # AR(1) 正自相关
    ic = pd.Series(0.05 + x)
    t_hac = _newey_west_t(ic)
    t_classic = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(n)))
    assert abs(t_hac) < abs(t_classic)


def test_ic_summary_includes_nw() -> None:
    """ic_summary 输出含 ic_t / ic_pvalue。"""
    rng = np.random.default_rng(3)
    ic = pd.Series(rng.normal(0.05, 0.1, 300))
    s = ic_summary(ic)
    assert "ic_t" in s and "ic_pvalue" in s
    assert s["ic_pvalue"] >= 0.0 and s["ic_pvalue"] <= 1.0


def test_fmb_recovers_lambda() -> None:
    """合成 r = 2*f1 + 0*f2 + 噪声 → λ̂₁≈2、λ̂₂≈0。"""
    rng = np.random.default_rng(2)
    n_days, n_syms = 100, 50
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    syms = [f"S{i}" for i in range(n_syms)]
    f1 = pd.DataFrame(rng.normal(0, 1, (n_days, n_syms)), index=idx, columns=syms)
    f2 = pd.DataFrame(rng.normal(0, 1, (n_days, n_syms)), index=idx, columns=syms)
    noise = pd.DataFrame(rng.normal(0, 0.5, (n_days, n_syms)), index=idx, columns=syms)
    fwd = 2.0 * f1 + 0.0 * f2 + noise

    res = fama_macbeth({"f1": f1, "f2": f2}, fwd, standardize=False)
    assert res.lambdas["f1"]["lambda_mean"] == pytest.approx(2.0, abs=0.5)
    assert res.lambdas["f2"]["lambda_mean"] == pytest.approx(0.0, abs=0.5)
    assert res.n_days >= 2

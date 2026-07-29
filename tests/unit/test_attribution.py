"""绩效归因(Phase 5)测试:Brinson 三效应恒等式 / 因子归因 / 暴露与行业分布。

不依赖网络:全部为人工构造的小样本面板。核心不变量:
- Brinson:配置 + 选股 + 交互 = 超额收益(满仓 / 含现金均成立);
- 因子归因:Σ 因子贡献 + α = 组合总收益。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from djinn.analytics import (
    brinson_attribution,
    build_exposure_report,
    factor_attribution,
)

# ── 共享小样本 ────────────────────────────────────────────
_DAYS = pd.date_range("2024-01-01", periods=4)
_RETURNS = pd.DataFrame(
    {
        "A": [0.01, 0.02, -0.01, 0.00],
        "B": [0.00, 0.01, 0.01, -0.02],
        "C": [0.02, -0.01, 0.00, 0.01],
    },
    index=_DAYS,
)
# 满仓组合权重(每行和 = 1)
_WEIGHTS = pd.DataFrame(
    {
        "A": [0.5, 0.4, 0.3, 0.3],
        "B": [0.3, 0.3, 0.4, 0.4],
        "C": [0.2, 0.3, 0.3, 0.3],
    },
    index=_DAYS,
)
_BENCH = {"A": 0.25, "B": 0.25, "C": 0.5}
_IND = {"A": "tech", "B": "tech", "C": "fin"}


# ── Brinson ──────────────────────────────────────────────
def test_brinson_effects_sum_to_excess() -> None:
    res = brinson_attribution(_WEIGHTS, _BENCH, _RETURNS, _IND)
    assert res.total_effect == pytest.approx(res.excess_return, abs=1e-9)
    assert res.excess_return != 0.0  # 非平凡(超额收益非零)
    # 按行业汇总三效应之和也等于超额收益
    assert res.by_industry().to_numpy().sum() == pytest.approx(res.excess_return)


def test_brinson_with_cash_keeps_identity() -> None:
    weights = _WEIGHTS * 0.6  # 每行和 0.6,40% 现金
    res = brinson_attribution(weights, _BENCH, _RETURNS, _IND)
    assert res.total_effect == pytest.approx(res.excess_return, abs=1e-9)
    assert "现金" in res.allocation.index  # 现金段作为一个配置行业出现


def test_brinson_missing_industry_falls_to_other() -> None:
    ind = {"A": "tech", "B": "tech"}  # C 缺行业 → "其他"
    res = brinson_attribution(_WEIGHTS, _BENCH, _RETURNS, ind)
    assert "其他" in res.allocation.index
    assert res.total_effect == pytest.approx(res.excess_return, abs=1e-9)


def test_brinson_to_dict_serializable() -> None:
    res = brinson_attribution(_WEIGHTS, _BENCH, _RETURNS, _IND)
    payload = res.to_dict()
    json.dumps(payload)
    assert set(payload["allocation"]["index"]) == {"tech", "fin"}


# ── 因子归因 ─────────────────────────────────────────────
def test_factor_attribution_identity() -> None:
    days = pd.date_range("2024-01-01", periods=6)
    expo = pd.DataFrame(
        {"value": [1, 1, 0.5, 0.5, -1, -1], "mom": [0.5] * 6}, index=days, dtype=float
    )
    fret = pd.DataFrame(
        {"value": [0.01, -0.01, 0.02, 0.0, 0.01, -0.02], "mom": [0.005] * 6},
        index=days,
        dtype=float,
    )
    alpha = pd.Series([0.001] * 6, index=days)
    port = (expo * fret).sum(axis=1) + alpha
    res = factor_attribution(port, expo, fret)
    assert res.total_return == pytest.approx(float(port.sum()))
    assert res.contributions.sum() + res.alpha == pytest.approx(res.total_return)
    # 已知 α = 6 × 0.001;因子贡献 = 暴露×因子收益 之和
    assert res.alpha == pytest.approx(0.006, abs=1e-9)
    assert res.contributions["value"] == pytest.approx(0.02)
    json.dumps(res.to_dict())


def test_factor_attribution_zero_alpha_when_exact() -> None:
    days = pd.date_range("2024-01-01", periods=3)
    expo = pd.DataFrame({"f": [1.0, 2.0, 3.0]}, index=days)
    fret = pd.DataFrame({"f": [0.01, 0.02, -0.01]}, index=days)
    port = (expo * fret).sum(axis=1)  # 无噪声 → α≈0
    res = factor_attribution(port, expo, fret)
    assert res.alpha == pytest.approx(0.0, abs=1e-12)
    assert res.contributions["f"] == pytest.approx(res.total_return)


# ── 因子暴露 / 行业分布 ───────────────────────────────────
def test_build_exposure_report() -> None:
    days = pd.date_range("2024-01-01", periods=3)
    w = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5], "B": [0.3, 0.3, 0.3], "C": [0.2, 0.2, 0.2]}, index=days
    )
    panel = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [0.0, 0.0, 0.0], "C": [-1.0, -1.0, -1.0]},
        index=days,
    )
    rep = build_exposure_report(w, {"value": panel}, _IND)
    # 因子暴露 = 权重 × 因子值:day0 = 0.5*1 + 0.3*0 + 0.2*(-1) = 0.3
    assert rep.exposures["value"].iloc[0] == pytest.approx(0.3)
    assert rep.exposures["value"].iloc[1] == pytest.approx(0.8)
    # 行业分布:tech = A+B = 0.8,fin = C = 0.2
    assert rep.industry_distribution["tech"].iloc[0] == pytest.approx(0.8)
    assert rep.industry_distribution["fin"].iloc[0] == pytest.approx(0.2)
    json.dumps(rep.to_dict())

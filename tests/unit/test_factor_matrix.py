"""多因子诊断矩阵测试:相关矩阵 / IC 汇总 / 序列化(不依赖网络)。"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from djinn.factor.analysis import analyze_factor_matrix


def _prices(rows: int = 80, cols: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([date(2024, 1, 1) + timedelta(days=i) for i in range(rows)])
    rets = rng.normal(0.0005, 0.02, (rows, cols))
    close = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(close, index=idx, columns=[f"S{j}" for j in range(cols)])


def test_correlated_vs_independent() -> None:
    """两个高相关因子相关 > 0.8,独立因子对 ~0,对角线恒为 1。"""
    prices = _prices()
    rng = np.random.default_rng(7)
    base = pd.DataFrame(
        rng.normal(size=prices.shape), index=prices.index, columns=prices.columns
    )
    # fa / fb 同源(共享 base + 独立噪声)→ 截面高度相关
    fa = base + pd.DataFrame(
        rng.normal(0, 0.05, prices.shape), index=prices.index, columns=prices.columns
    )
    fb = base + pd.DataFrame(
        rng.normal(0, 0.05, prices.shape), index=prices.index, columns=prices.columns
    )
    # fc 完全独立
    fc = pd.DataFrame(
        rng.normal(size=prices.shape), index=prices.index, columns=prices.columns
    )
    report = analyze_factor_matrix({"fa": fa, "fb": fb, "fc": fc}, prices)

    corr = report.correlation
    assert list(corr.index) == ["fa", "fb", "fc"]
    # 对角线 = 1
    for name in ("fa", "fb", "fc"):
        assert corr.loc[name, name] == pytest.approx(1.0)
    # 矩阵对称
    pd.testing.assert_frame_equal(corr, corr.T, check_names=False)
    # fa-fb 高相关
    assert corr.loc["fa", "fb"] > 0.8
    # 独立因子对相关接近 0(固定种子,容忍小噪声)
    assert abs(corr.loc["fa", "fc"]) < 0.2
    assert abs(corr.loc["fb", "fc"]) < 0.2


def test_ic_summary_and_turnover() -> None:
    """每因子每期 IC 汇总齐全;完美因子 IC 高,随机因子 IC 接近 0;换手在 [0,1]。"""
    prices = _prices()
    from djinn.factor.analysis import compute_forward_returns

    fwd = compute_forward_returns(prices, [1, 5])[1]
    # 完美因子(因子值≈下期收益)
    perfect = fwd + pd.DataFrame(
        np.random.default_rng(11).normal(0, 1e-4, fwd.shape),
        index=fwd.index,
        columns=fwd.columns,
    )
    random = pd.DataFrame(
        np.random.default_rng(12).normal(size=prices.shape),
        index=prices.index,
        columns=prices.columns,
    )
    report = analyze_factor_matrix(
        {"perfect": perfect, "random": random}, prices, periods=[1, 5]
    )
    # ic_summary 键是 period → factor
    assert set(report.ic_summary.keys()) == {1, 5}
    for p in (1, 5):
        assert set(report.ic_summary[p].keys()) == {"perfect", "random"}
        summ = report.ic_summary[p]["perfect"]
        assert {"ic_mean", "ic_std", "icir", "ic_pos_ratio"} <= set(summ.keys())
    # 完美因子 period=1 的 IC 高
    assert report.ic_summary[1]["perfect"]["ic_mean"] > 0.9
    # 随机因子 IC 接近 0
    assert abs(report.ic_summary[1]["random"]["ic_mean"]) < 0.1
    # 换手在合理区间
    for v in report.turnover.values():
        assert 0.0 <= v <= 1.0


def test_to_dict_json_serializable() -> None:
    """to_dict() 可 JSON 序列化,无 NaN/Inf,结构符合 {index,columns,data} 约定。"""
    prices = _prices()
    rng = np.random.default_rng(3)
    fa = pd.DataFrame(
        rng.normal(size=prices.shape), index=prices.index, columns=prices.columns
    )
    fb = pd.DataFrame(
        rng.normal(size=prices.shape), index=prices.index, columns=prices.columns
    )
    report = analyze_factor_matrix({"fa": fa, "fb": fb}, prices)
    d = report.to_dict()
    # json.dumps 不抛错(内部 _finite 已把 NaN/Inf → None)
    json.dumps(d)
    assert d["factors"] == ["fa", "fb"]
    # correlation 是 2×2 方阵
    assert d["correlation"]["index"] == ["fa", "fb"]
    assert d["correlation"]["columns"] == ["fa", "fb"]
    assert len(d["correlation"]["data"]) == 2
    assert all(len(row) == 2 for row in d["correlation"]["data"])
    # 对角线 = 1
    assert d["correlation"]["data"][0][0] == pytest.approx(1.0)
    assert d["correlation"]["data"][1][1] == pytest.approx(1.0)
    # ic_summary 字符串期键
    assert set(d["ic_summary"].keys()) == {"1", "5", "10"}
    assert set(d["turnover"].keys()) == {"fa", "fb"}

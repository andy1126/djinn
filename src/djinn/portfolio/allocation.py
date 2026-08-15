"""权重分配:等权 / 市值 / 自定义 / 打分 / 风险平价 / 最小方差 / 均值-方差。

基础三种(``equal`` / ``market_cap`` / ``custom``)只依赖标的列表与价格;进阶四种
(``score`` / ``risk_parity`` / ``min_variance`` / ``mean_variance``)额外依赖:

- ``scores``:综合得分(选股 alpha 信号),驱动 :class:`ScoreWeight` 与
  :class:`MeanVarianceWeight`(作预期收益代理);
- ``cov``:标的 × 标的 日收益协方差宽表(可由 :func:`estimate_covariance` 估计),
  驱动 :class:`RiskParityWeight` / :class:`MinVarianceWeight` / :class:`MeanVarianceWeight`。

缺少所需输入时,进阶分配器一律退化为等权,保证任何情形下都能产出合法权重。
组合优化(最小方差 / 均值-方差)用 ``scipy.optimize.minimize``(SLSQP,约束 Σw=1、
w≥0,多头),为延迟加载——仅在真正用到优化分配器时才 import scipy。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from djinn.utils.exceptions import StrategyError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

AllocationType = Literal[
    "equal",
    "market_cap",
    "custom",
    "score",
    "risk_parity",
    "min_variance",
    "mean_variance",
]


def _equal_weights(symbols: list[str]) -> dict[str, float]:
    """等权 {symbol: 1/N}(空列表返回 {})。"""
    n = len(symbols)
    if n == 0:
        return {}
    return dict.fromkeys(symbols, 1.0 / n)


class Allocation(ABC):
    """分配策略基类:给定标的列表与上下文,产出目标权重 {symbol: weight}。"""

    # A8:分配器所需的输入依赖(引擎再平衡路径无法提供时启动校验拒绝)
    requires: frozenset[str] = frozenset()
    _warned: bool = False  # 退化告警只警一次(A8)

    def _warn_degrade(self) -> None:
        """缺参退化为等权时显式告警一次(替代静默,便于排查配置错误)。"""
        if not self._warned:
            self._warned = True
            _log.warning(
                "%s 需要 %s 但调用方未提供,退化为等权",
                type(self).__name__,
                ", ".join(sorted(self.requires)),
            )

    @abstractmethod
    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
        cov: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """返回归一化目标权重(和为 1,空列表返回 {})。"""


class EqualWeight(Allocation):
    """等权分配:每个成分 1/N。"""

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
        cov: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        return _equal_weights(symbols)


class MarketCapWeight(Allocation):
    """市值加权:按最新价 * 流通股(此处用最新价代理,缺流通股数据时退化为等权)。"""

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
        cov: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        if not symbols or not prices:
            return _equal_weights(symbols)
        caps = {s: max(prices.get(s, 0.0), 0.0) for s in symbols}
        total = sum(caps.values())
        if total <= 0:
            return _equal_weights(symbols)
        return {s: caps[s] / total for s in symbols}


class CustomWeight(Allocation):
    """自定义权重:按用户给定 dict 归一化。"""

    def __init__(self, weights: dict[str, float]) -> None:
        # 校验非负
        for s, w in weights.items():
            if w < 0:
                raise StrategyError(f"自定义权重不能为负:{s}={w}")
        self._raw = dict(weights)

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
        cov: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        sub = {s: self._raw.get(s, 0.0) for s in symbols}
        total = sum(sub.values())
        if total <= 0:
            return dict.fromkeys(symbols, 0.0)
        return {s: sub[s] / total for s in symbols}


class ScoreWeight(Allocation):
    """打分加权:权重与综合得分单调一致(得分越高权重越大)。

    得分可为负(z-score),先减去截面最小得分平移到非负,再加 ``0.1 × 得分跨度``
    的基底,保证严格单调且最低分标的仍分到非平凡权重(不会近乎清零)。
    无打分时退化为等权。
    """

    requires = frozenset({"scores"})

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
        cov: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        if not symbols or not scores:
            self._warn_degrade()
            return _equal_weights(symbols)
        present = [scores[s] for s in symbols if s in scores]
        if not present:
            self._warn_degrade()
            return _equal_weights(symbols)
        floor = min(present)
        span = max(present) - floor
        base = 0.1 * span if span > 0 else 1.0
        raw = {s: (scores.get(s, floor) - floor) + base for s in symbols}
        return normalize_weights(raw)


class RiskParityWeight(Allocation):
    """风险平价:迭代使各成分对组合波动的风险贡献近似相等。

    需要协方差 ``cov``;缺失或不可用时退化为等权。
    """

    requires = frozenset({"cov"})

    def __init__(self, max_iter: int = 1000, tol: float = 1e-10) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
        cov: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        m = _cov_matrix(cov, symbols)
        if m is None:
            self._warn_degrade()
            return _equal_weights(symbols)
        w = _risk_parity(m, self.max_iter, self.tol)
        return {s: float(w[i]) for i, s in enumerate(symbols)}


class MinVarianceWeight(Allocation):
    """最小方差:min w'Σw,s.t. Σw=1、w≥0(SLSQP)。无 cov 退化等权。"""

    requires = frozenset({"cov"})

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
        cov: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        m = _cov_matrix(cov, symbols)
        if m is None:
            self._warn_degrade()
            return _equal_weights(symbols)
        w = _solve_markowitz(m, mu=None, risk_aversion=1.0)
        return {s: float(w[i]) for i, s in enumerate(symbols)}


class MeanVarianceWeight(Allocation):
    """均值-方差:max μ'w − (γ/2) w'Σw,s.t. Σw=1、w≥0(SLSQP,多头)。

    以综合得分 ``scores`` 为预期收益代理(α 信号),``cov`` 为风险。
    ``risk_aversion``(γ)越大越保守、越贴近最小方差;得分量纲与协方差不同,
    实务上 γ 常取较大值(默认 10)。缺 ``cov`` 退化等权,缺 ``scores`` 退化最小方差。
    """

    requires = frozenset({"cov"})

    def __init__(self, risk_aversion: float = 10.0) -> None:
        self.risk_aversion = float(risk_aversion)

    def target_weights(
        self,
        symbols: list[str],
        ctx: object | None = None,
        prices: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
        cov: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        m = _cov_matrix(cov, symbols)
        if m is None:
            self._warn_degrade()
            return _equal_weights(symbols)
        mu: NDArray[np.float64] | None = None
        if scores:
            mu = np.array([scores.get(s, 0.0) for s in symbols], dtype=np.float64)
        w = _solve_markowitz(m, mu=mu, risk_aversion=self.risk_aversion)
        return {s: float(w[i]) for i, s in enumerate(symbols)}


def make_allocation(
    kind: AllocationType,
    weights: dict[str, float] | None = None,
    *,
    risk_aversion: float = 10.0,
) -> Allocation:
    """工厂:按类型字符串构造分配器。"""
    if kind == "equal":
        return EqualWeight()
    if kind == "market_cap":
        return MarketCapWeight()
    if kind == "custom":
        if not weights:
            raise StrategyError("custom 分配需要 weights 字典")
        return CustomWeight(weights)
    if kind == "score":
        return ScoreWeight()
    if kind == "risk_parity":
        return RiskParityWeight()
    if kind == "min_variance":
        return MinVarianceWeight()
    if kind == "mean_variance":
        return MeanVarianceWeight(risk_aversion)
    raise StrategyError(f"未知分配类型: {kind}")


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """归一化权重(和为 1)。"""
    total = sum(max(w, 0.0) for w in weights.values())
    if total <= 0:
        return dict.fromkeys(weights, 0.0)
    return {s: max(w, 0.0) / total for s, w in weights.items()}


def estimate_covariance(returns: pd.DataFrame, shrink: float = 0.0) -> pd.DataFrame:
    """由日收益宽表估计协方差;``shrink`` ∈ [0,1] 向对角阵收缩(类 Ledoit-Wolf)。

    ``shrink=0`` 为样本协方差;``shrink=1`` 只保留各方差(相关性归零)。
    小样本下适度收缩可提高协方差矩阵的数值稳定性,避免优化病态。
    """
    sample = returns.cov()
    s = min(max(float(shrink), 0.0), 1.0)
    if s == 0.0:
        return sample
    diag = pd.DataFrame(
        np.diag(np.diag(sample.to_numpy(dtype=np.float64))),
        index=sample.index,
        columns=sample.columns,
    )
    return (1.0 - s) * sample + s * diag


# ── 数值求解(私有)────────────────────────────────────────


def _cov_matrix(
    cov: pd.DataFrame | None, symbols: list[str]
) -> NDArray[np.float64] | None:
    """把协方差宽表重排到 ``symbols`` 顺序并校验可用性;不可用返回 None(退化等权)。"""
    if not symbols or cov is None or cov.empty:
        return None
    sub = cov.reindex(index=symbols, columns=symbols)
    arr: NDArray[np.float64] = np.asarray(
        sub.to_numpy(dtype=np.float64), dtype=np.float64
    )
    n = len(symbols)
    if arr.shape != (n, n) or not np.all(np.isfinite(arr)):
        return None
    arr = 0.5 * (arr + arr.T)  # 数值对称化
    if bool(np.any(np.diag(arr) <= 0)):  # 方差必须为正
        return None
    return arr


def _normalize_nonneg(w: NDArray[np.float64]) -> NDArray[np.float64]:
    """裁剪微小负值并归一化(和为 1);全零回退等权。"""
    arr = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return np.full(arr.shape[0], 1.0 / arr.shape[0])
    return arr / total


def _risk_parity(
    cov: NDArray[np.float64], max_iter: int, tol: float
) -> NDArray[np.float64]:
    """阻尼不动点迭代求风险平价权重(各成分风险贡献 w_i·(Σw)_i 相等)。

    不动点 w ∝ 1/(Σw) ⇒ w_i·(Σw)_i 为常数 ⇒ 各成分对组合波动的风险贡献相等
    (风险贡献 RC_i = w_i·(Σw)_i / σ_p,σ_p 为组合波动,对全体成分相同)。
    """
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        m = cov @ w
        m = np.where(m <= 0, 1e-12, m)  # 防御:非正边际风险贡献
        inv = 1.0 / m
        inv /= float(inv.sum())
        w_new = 0.5 * (w + inv)  # 阻尼更新防振荡
        w_new /= float(w_new.sum())
        if float(np.max(np.abs(w_new - w))) < tol:
            w = w_new
            break
        w = w_new
    return _normalize_nonneg(w)


def _solve_markowitz(
    cov: NDArray[np.float64],
    mu: NDArray[np.float64] | None,
    risk_aversion: float,
) -> NDArray[np.float64]:
    """SLSQP 求解 min (γ/2)·w'Σw − μ'w,s.t. Σw=1、w≥0(多头)。

    ``mu=None`` 时退化为最小方差(γ 不影响无 μ 的最优解)。求解失败回退等权,
    保证任何情形下都产出合法权重。
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, minimize
    except ImportError as e:  # pragma: no cover
        raise StrategyError("scipy 未安装(pip install scipy),无法做组合优化") from e
    n = cov.shape[0]
    w0 = np.full(n, 1.0 / n)

    def objective(w: NDArray[np.float64]) -> float:
        val = 0.5 * risk_aversion * float(w @ cov @ w)
        if mu is not None:
            val -= float(mu @ w)
        return val

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=[LinearConstraint(np.ones(n), lb=1.0, ub=1.0)],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    w = res.x if res.success and np.all(np.isfinite(res.x)) else w0
    return _normalize_nonneg(w)

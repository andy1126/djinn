"""时序指标库:纯 pandas/numpy,签名统一 ``(Series[, ...]) -> Series/DataFrame``。

既是用户自定义策略(Python 编辑器)的安全词汇表,也是 Pine 转译(``ta.*``)的
映射目标,命名尽量贴近 TradingView Pine 的 ``ta.*`` 语义。
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "wma",
    "rma",
    "vwma",
    "hma",
    "rsi",
    "macd",
    "stoch",
    "cci",
    "obv",
    "mfi",
    "wpr",
    "dmi",
    "aroon",
    "supertrend",
    "psar",
    "atr",
    "tr",
    "bb",
    "stdev",
    "variance",
    "highest",
    "lowest",
    "donchian",
    "kc",
    "change",
    "roc",
    "momentum",
    "cross_over",
    "cross_under",
    "valuewhen",
    "barssince",
    "rising",
    "falling",
]


def _n(v: int | float) -> int:
    return int(v)


# ── 均线 / 趋势 ─────────────────────────────────────────
def sma(s: pd.Series, n: int | float) -> pd.Series:
    """简单移动平均(不足 n 根为 NaN)。"""
    n = _n(n)
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int | float) -> pd.Series:
    """指数移动平均(alpha = 2/(n+1),与 Pine ta.ema 一致)。"""
    n = _n(n)
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int | float) -> pd.Series:
    """Wilder 平滑(alpha = 1/n,Pine ta.rma)。"""
    n = _n(n)
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int | float) -> pd.Series:
    """加权移动平均(线性权重 1..n)。"""
    n = _n(n)
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(
        lambda x: float(np.dot(x, weights[-len(x) :]) / weights[-len(x) :].sum()),
        raw=True,
    )


def vwma(close: pd.Series, volume: pd.Series, n: int | float) -> pd.Series:
    """成交量加权移动平均 VWMA。"""
    n = _n(n)
    pv = (close * volume).rolling(n, min_periods=n).sum()
    v = volume.rolling(n, min_periods=n).sum()
    return pv / v.replace(0, np.nan)


def hma(close: pd.Series, n: int | float) -> pd.Series:
    """Hull 移动平均(WMA 组合,更平滑低滞后)。"""
    n = _n(n)
    half = max(1, n // 2)
    sqrt_n = max(1, round(n**0.5))
    return wma(2 * wma(close, half) - wma(close, n), sqrt_n)


# ── 振荡器 ──────────────────────────────────────────────
def rsi(close: pd.Series, period: int | float = 14) -> pd.Series:
    """RSI(Wilder 平滑,与内置 RSIReversal 一致)。"""
    period = _n(period)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def macd(
    close: pd.Series,
    fast: int | float = 12,
    slow: int | float = 26,
    signal: int | float = 9,
) -> pd.DataFrame:
    """MACD,返回 ``{macd, signal, hist}``。"""
    fast, slow, signal = _n(fast), _n(slow), _n(signal)
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "hist": macd_line - signal_line,
        }
    )


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int | float = 14,
    d: int | float = 3,
    smooth: int | float = 3,
) -> pd.DataFrame:
    """随机指标,返回 ``{k, d}``(Pine ta.stoch 的平滑方式)。"""
    k, d, smooth = _n(k), _n(d), _n(smooth)
    hh = high.rolling(k, min_periods=k).max()
    ll = low.rolling(k, min_periods=k).min()
    raw_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k_line = raw_k.rolling(smooth, min_periods=smooth).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"k": k_line, "d": d_line})


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int | float = 20
) -> pd.Series:
    """顺势指标 CCI。"""
    n = _n(n)
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(n, min_periods=n).mean()
    mad = tp.rolling(n, min_periods=n).apply(
        lambda x: float(np.abs(x - x.mean()).mean()), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """能量潮 OBV。"""
    diff = close.diff()
    direction = (diff > 0).astype(float) - (diff < 0).astype(float)
    direction = direction.fillna(0.0)
    return (direction * volume).cumsum()


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    n: int | float = 14,
) -> pd.Series:
    """资金流量指标 MFI(典型价 × 成交量,0~100)。"""
    n = _n(n)
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(n, min_periods=n).sum()
    neg_sum = neg.rolling(n, min_periods=n).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


def wpr(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int | float = 14
) -> pd.Series:
    """威廉指标 Williams %R(-100~0,-80 以下超卖,-20 以上超买)。"""
    n = _n(n)
    hh = highest(high, n)
    ll = lowest(low, n)
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


# ── 趋势强度 ────────────────────────────────────────────
def dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    di_length: int | float = 14,
    adx_smoothing: int | float = 14,
) -> pd.DataFrame:
    """方向指标 DMI,返回 ``{plus_di, minus_di, adx}``(Wilder 平滑)。"""
    di_length = _n(di_length)
    adx_smoothing = _n(adx_smoothing)
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=high.index, dtype="float64"
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=high.index, dtype="float64"
    )
    atr_ = rma(tr(high, low, close), di_length)
    plus_di = 100 * rma(plus_dm, di_length) / atr_.replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, di_length) / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = rma(dx, adx_smoothing)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def aroon(high: pd.Series, low: pd.Series, n: int | float = 14) -> pd.DataFrame:
    """阿隆指标,返回 ``{aroon_up, aroon_down}``(0~100,100=近期新高/新低)。"""
    n = _n(n)
    up = high.rolling(n, min_periods=n).apply(
        lambda x: float((n - 1) - int(np.argmax(x))), raw=True
    )
    down = low.rolling(n, min_periods=n).apply(
        lambda x: float((n - 1) - int(np.argmin(x))), raw=True
    )
    return pd.DataFrame(
        {"aroon_up": 100 * (n - up) / n, "aroon_down": 100 * (n - down) / n}
    )


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    factor: float = 3.0,
    atr_period: int | float = 10,
) -> pd.DataFrame:
    """超级趋势,返回 ``{supertrend, direction}``(direction=+1 多头 / -1 空头)。

    下轨做多止损、上轨做空止损,逐根递推(无闭式,首 ``atr_period`` 根为 NaN)。
    """
    atr_period = _n(atr_period)
    hl2 = (high + low) / 2
    atr_ = atr(high, low, close, atr_period)
    basic_ub = hl2 + factor * atr_
    basic_lb = hl2 - factor * atr_

    n = len(close)
    final_ub = pd.Series(np.nan, index=close.index, dtype="float64")
    final_lb = pd.Series(np.nan, index=close.index, dtype="float64")
    st = pd.Series(np.nan, index=close.index, dtype="float64")
    direction = pd.Series(0, index=close.index, dtype="int64")

    start = atr_period - 1  # 第一个有效 ATR 的位置(0-indexed)
    if start >= n:
        return pd.DataFrame({"supertrend": st, "direction": direction})
    final_ub.iloc[start] = basic_ub.iloc[start]
    final_lb.iloc[start] = basic_lb.iloc[start]
    direction.iloc[start] = 1
    st.iloc[start] = final_lb.iloc[start]

    for i in range(start + 1, n):
        final_ub.iloc[i] = (
            basic_ub.iloc[i]
            if (basic_ub.iloc[i] < final_ub.iloc[i - 1])
            or (close.iloc[i - 1] > final_ub.iloc[i - 1])
            else final_ub.iloc[i - 1]
        )
        final_lb.iloc[i] = (
            basic_lb.iloc[i]
            if (basic_lb.iloc[i] > final_lb.iloc[i - 1])
            or (close.iloc[i - 1] < final_lb.iloc[i - 1])
            else final_lb.iloc[i - 1]
        )
        if direction.iloc[i - 1] == 1:  # 前一根多头
            if close.iloc[i] < final_lb.iloc[i]:
                direction.iloc[i] = -1
                st.iloc[i] = final_ub.iloc[i]
            else:
                direction.iloc[i] = 1
                st.iloc[i] = final_lb.iloc[i]
        else:  # 前一根空头
            if close.iloc[i] > final_ub.iloc[i]:
                direction.iloc[i] = 1
                st.iloc[i] = final_lb.iloc[i]
            else:
                direction.iloc[i] = -1
                st.iloc[i] = final_ub.iloc[i]

    return pd.DataFrame({"supertrend": st, "direction": direction})


def psar(
    high: pd.Series,
    low: pd.Series,
    start: float = 0.02,
    inc: float = 0.02,
    max_: float = 0.2,
) -> pd.Series:
    """抛物线 SAR(加速因子 ``start``,步进 ``inc``,上限 ``max_``)。

    逐根递推;多头起点 SAR 从首根最低价开始,SAR 不超过前两根低点(空头反之)。
    """
    n = len(high)
    sar = pd.Series(np.nan, index=high.index, dtype="float64")
    trend_up = True
    ep = float(high.iloc[0])
    af = start
    sar.iloc[0] = float(low.iloc[0])
    for i in range(1, n):
        prev_sar = float(sar.iloc[i - 1])
        if trend_up:
            sar_i = prev_sar + af * (ep - prev_sar)
            if i >= 2:
                sar_i = min(sar_i, float(low.iloc[i - 1]), float(low.iloc[i - 2]))
            if float(low.iloc[i]) < sar_i:
                sar.iloc[i] = ep
                trend_up = False
                ep = float(low.iloc[i])
                af = start
            else:
                sar.iloc[i] = sar_i
                if float(high.iloc[i]) > ep:
                    ep = float(high.iloc[i])
                    af = min(af + inc, max_)
        else:
            sar_i = prev_sar - af * (prev_sar - ep)
            if i >= 2:
                sar_i = max(sar_i, float(high.iloc[i - 1]), float(high.iloc[i - 2]))
            if float(high.iloc[i]) > sar_i:
                sar.iloc[i] = ep
                trend_up = True
                ep = float(high.iloc[i])
                af = start
            else:
                sar.iloc[i] = sar_i
                if float(low.iloc[i]) < ep:
                    ep = float(low.iloc[i])
                    af = min(af + inc, max_)
    return sar


# ── 波动 ────────────────────────────────────────────────
def tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """真实波幅 TR = max(high-low, |high-prev_close|, |low-prev_close|)。"""
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int | float = 14
) -> pd.Series:
    """平均真实波幅 ATR(用 RMA 平滑,同 Pine)。"""
    n = _n(n)
    return rma(tr(high, low, close), n)


def bb(close: pd.Series, n: int | float = 20, mult: float = 2.0) -> pd.DataFrame:
    """布林带,返回 ``{upper, mid, lower}``。"""
    n = _n(n)
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame(
        {"upper": mid + mult * sd, "mid": mid, "lower": mid - mult * sd}
    )


def stdev(s: pd.Series, n: int | float) -> pd.Series:
    """滚动标准差(总体 ddof=0,同 Pine)。"""
    n = _n(n)
    return s.rolling(n, min_periods=n).std(ddof=0)


def variance(s: pd.Series, n: int | float) -> pd.Series:
    """滚动方差(总体 ddof=0)。"""
    n = _n(n)
    return s.rolling(n, min_periods=n).var(ddof=0)


# ── 通道 / 极值 ─────────────────────────────────────────
def highest(s: pd.Series, n: int | float) -> pd.Series:
    """N 根内最高值(Pine ta.highest)。"""
    n = _n(n)
    return s.rolling(n, min_periods=n).max()


def lowest(s: pd.Series, n: int | float) -> pd.Series:
    """N 根内最低值(Pine ta.lowest)。"""
    n = _n(n)
    return s.rolling(n, min_periods=n).min()


def donchian(high: pd.Series, low: pd.Series, n: int | float) -> pd.DataFrame:
    """唐奇安通道,返回 ``{upper, lower}``。"""
    n = _n(n)
    return pd.DataFrame(
        {
            "upper": high.rolling(n, min_periods=n).max(),
            "lower": low.rolling(n, min_periods=n).min(),
        }
    )


def kc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int | float = 20,
    mult: float = 1.5,
) -> pd.DataFrame:
    """肯特纳通道,返回 ``{mid, upper, lower}``(SMA 中轨 + ATR 带宽)。"""
    n = _n(n)
    mid = sma(close, n)
    rng = sma(tr(high, low, close), n)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + mult * rng, "lower": mid - mult * rng}
    )


# ── 变化率 ──────────────────────────────────────────────
def change(s: pd.Series, n: int | float = 1) -> pd.Series:
    """N 根前的变化量(source - source[n])。"""
    n = _n(n)
    return s - s.shift(n)


def roc(s: pd.Series, n: int | float = 1) -> pd.Series:
    """变动率 ROC(百分比)。"""
    n = _n(n)
    return (s / s.shift(n) - 1) * 100


def momentum(s: pd.Series, n: int | float = 1) -> pd.Series:
    """动量(Pine ta.mom:source - source[n])。"""
    return change(s, n)


# ── 信号辅助 ────────────────────────────────────────────
def cross_over(a: pd.Series, b: pd.Series | float) -> pd.Series:
    """a 上穿 b(前一 bar a<=b 且当前 a>b);b 可为标量(如阈值)。"""
    a = a if isinstance(a, pd.Series) else pd.Series(a)
    prev_b = b.shift(1) if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift(1) <= prev_b)


def cross_under(a: pd.Series, b: pd.Series | float) -> pd.Series:
    """a 下穿 b;b 可为标量。"""
    a = a if isinstance(a, pd.Series) else pd.Series(a)
    prev_b = b.shift(1) if isinstance(b, pd.Series) else b
    return (a < b) & (a.shift(1) >= prev_b)


def valuewhen(cond: pd.Series, source: pd.Series, occurrence: int = 0) -> pd.Series:
    """取第 occurrence 次(0=最近)满足 cond 时的 source 值,前向填充。"""
    cond = pd.Series(cond).fillna(False).astype(bool)
    src = pd.Series(source).astype(float)
    out = pd.Series(np.nan, index=src.index, dtype="float64")
    true_pos = np.flatnonzero(cond.to_numpy())
    if len(true_pos) == 0:
        return out
    for i in range(len(src)):
        pos = int(np.searchsorted(true_pos, i, side="right")) - 1
        if pos - occurrence >= 0:
            out.iloc[i] = src.iloc[true_pos[pos - occurrence]]
    return out


def barssince(cond: pd.Series) -> pd.Series:
    """距最近一次 cond 为真的 bar 数(cond 当根为 0,从未满足为 NaN)。"""
    cond = pd.Series(cond).fillna(False).astype(bool)
    out = pd.Series(np.nan, index=cond.index, dtype="float64")
    last = -1
    for i in range(len(cond)):
        if bool(cond.iloc[i]):
            out.iloc[i] = 0
            last = i
        elif last >= 0:
            out.iloc[i] = float(i - last)
    return out


def rising(s: pd.Series, n: int | float = 1) -> pd.Series:
    """最近 n 根内严格上升(布尔,同 Pine ta.rising)。"""
    n = _n(n)
    out = s > s.shift(1)
    for i in range(1, n):
        out &= s.shift(i) > s.shift(i + 1)
    return out


def falling(s: pd.Series, n: int | float = 1) -> pd.Series:
    """最近 n 根内严格下降(布尔,同 Pine ta.falling)。"""
    n = _n(n)
    out = s < s.shift(1)
    for i in range(1, n):
        out &= s.shift(i) < s.shift(i + 1)
    return out


# ── 元数据(供「指标库」页展示)────────────────────────────
INDICATOR_CATEGORIES: dict[str, str] = {
    "sma": "趋势",
    "ema": "趋势",
    "wma": "趋势",
    "rma": "趋势",
    "vwma": "趋势",
    "hma": "趋势",
    "rsi": "振荡",
    "macd": "振荡",
    "stoch": "振荡",
    "cci": "振荡",
    "obv": "振荡",
    "mfi": "振荡",
    "wpr": "振荡",
    "dmi": "趋势",
    "aroon": "趋势",
    "supertrend": "趋势",
    "psar": "趋势",
    "atr": "波动",
    "tr": "波动",
    "bb": "波动",
    "stdev": "波动",
    "variance": "波动",
    "highest": "通道",
    "lowest": "通道",
    "donchian": "通道",
    "kc": "通道",
    "change": "变化",
    "roc": "变化",
    "momentum": "变化",
    "cross_over": "信号",
    "cross_under": "信号",
    "valuewhen": "信号",
    "barssince": "信号",
    "rising": "信号",
    "falling": "信号",
}


def _fmt_signature(name: str, func: Any) -> str:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return name
    parts: list[str] = []
    for p in sig.parameters.values():
        s = p.name
        if p.default is not inspect.Parameter.empty:
            s += f"={p.default!r}"
        parts.append(s)
    return f"{name}({', '.join(parts)})"


def indicator_specs() -> list[dict[str, Any]]:
    """列出内置指标元数据:名称 / 分类 / 说明 / 签名 / 参数 / 源码。"""
    out: list[dict[str, Any]] = []
    for name in __all__:
        func = globals().get(name)
        if not callable(func):
            continue
        doc = (func.__doc__ or "").strip()
        desc = doc.splitlines()[0].strip() if doc else ""
        try:
            params = [
                {
                    "name": p.name,
                    "default": (
                        None if p.default is inspect.Parameter.empty else p.default
                    ),
                }
                for p in inspect.signature(func).parameters.values()
            ]
        except (TypeError, ValueError):
            params = []
        try:
            source = inspect.getsource(func)
        except OSError:
            source = ""
        out.append(
            {
                "name": name,
                "category": INDICATOR_CATEGORIES.get(name, "其他"),
                "description": desc,
                "doc": doc,
                "signature": _fmt_signature(name, func),
                "params": params,
                "source": source,
            }
        )
    return out

"""Adaptive Trend Trail 指标(Pine → Python 移植)。

原:Uptrick "Adaptive Trend Trail"(@version=6,CC BY-SA 4.0,https://creativecommons.org/licenses/by-sa/4.0/)。
仅移植指标**信号核心**(趋势状态机 / 翻转信号 / 自适应趋势带 / 估值),不含 Pine
专属的 ``plotcandle`` 染色与 ``table`` 估值表渲染;回测请用 djinn 引擎 + 本信号。

实现要点:
- 指标部分(EMA/ATR/Supertrend/RSI/highest/lowest/滚动和)向量化;
- 状态机(趋势 + 确认计数 + 冷却)逐根递推,与 Pine ``var`` 语义一致;
- ``barstate.isconfirmed`` 在历史回测中视为恒真(所有 bar 已收盘);
- 自适应 Supertrend 的 ``factor`` 为逐根 Series(随 chop/波动率动态调整)。

返回 DataFrame 列:
- ``trend``: 1 多头 / -1 空头 / 0 中性
- ``up_signal`` / ``down_signal``: 趋势翻转信号(t 日生成,t+1 开盘执行)
- ``bullish`` / ``bearish``: 布尔
- ``inner_trail`` / ``outer_trail`` / ``depth_trail``: 自适应趋势带(``outer_trail`` 即移动止损线)
- ``valuation``: 估值表(EMA(RSI,3),clip 0~100)
- ``regime`` / ``chop`` / ``efficiency`` / ``dynamic_gate``: 中间量(调试)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from djinn.indicators import atr, ema, highest, lowest, rsi, supertrend


def adaptive_trend_trail(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    *,
    trend_length: int = 34,
    momentum_length: int = 12,
    sensitivity: float = 0.35,
    st_fast_length: int = 9,
    st_fast_factor: float = 1.45,
    st_mid_length: int = 14,
    st_mid_factor: float = 1.95,
    st_slow_length: int = 21,
    st_slow_factor: float = 2.55,
    trail_size: float = 1.0,
    smoothness: int = 5,
    mintick: float = 1e-8,
) -> pd.DataFrame:
    """计算 Adaptive Trend Trail 趋势信号(输入为对齐的 OHLC Series)。"""
    idx = close.index

    # ── 01. 基础市场状态 ────────────────────────────────
    basis = ema(close, trend_length)
    safe_atr = atr(high, low, close, 14).clip(lower=mintick)
    momentum = ema(close - close.shift(momentum_length), 5) / safe_atr
    distance = (close - basis) / safe_atr

    # ── 02. 方向效率 ────────────────────────────────────
    eff_length = 10
    net_movement = (close - close.shift(eff_length)).abs()
    travel_path = (
        (close - close.shift(1)).abs().rolling(eff_length, min_periods=eff_length).sum()
    )
    efficiency = (net_movement / travel_path).clip(0.0, 1.0)
    efficiency = efficiency.where(travel_path > 0.0, 0.0)
    chop = 1.0 - efficiency
    eff_direction = pd.Series(0.0, index=idx)
    eff_direction = eff_direction.mask(close > close.shift(eff_length), 1.0)
    eff_direction = eff_direction.mask(close < close.shift(eff_length), -1.0)
    efficiency_field = efficiency * eff_direction

    # ── 03. 波动率 regime ───────────────────────────────
    normal_atr = ema(safe_atr, 50)
    vol_ratio = (safe_atr / normal_atr).where(normal_atr > 0.0, 1.0)
    vol_expansion = (vol_ratio - 1.0).clip(0.0, 1.25)
    vol_deviation = (vol_ratio - 1.0).abs().clip(0.0, 1.50)

    # ── 04. 自适应 Supertrend 矩阵 ──────────────────────
    fast_factor = st_fast_factor + chop * 0.25 + vol_expansion * 0.10
    mid_factor = st_mid_factor + chop * 0.35 + vol_expansion * 0.15
    slow_factor = st_slow_factor + chop * 0.45 + vol_expansion * 0.20

    st_fast = supertrend(high, low, close, fast_factor, st_fast_length)["direction"]
    st_mid = supertrend(high, low, close, mid_factor, st_mid_length)["direction"]
    st_slow = supertrend(high, low, close, slow_factor, st_slow_length)["direction"]

    # Pine direction<0 = 多头;djinn direction=+1 多头 / -1 空头
    st_fast_bull = st_fast == 1
    st_fast_bear = st_fast == -1
    st_mid_bull = st_mid == 1
    st_mid_bear = st_mid == -1
    st_slow_bull = st_slow == 1
    st_slow_bear = st_slow == -1

    bull_st_votes = (
        st_fast_bull.astype(int) + st_mid_bull.astype(int) + st_slow_bull.astype(int)
    )
    bear_st_votes = (
        st_fast_bear.astype(int) + st_mid_bear.astype(int) + st_slow_bear.astype(int)
    )

    required_st_votes = pd.Series(3, index=idx, dtype=int).where(chop > 0.70, 2)
    bull_st_confirmed = bull_st_votes >= required_st_votes
    bear_st_confirmed = bear_st_votes >= required_st_votes
    full_bull_st = bull_st_votes == 3
    full_bear_st = bear_st_votes == 3

    # ── 05. 核心 regime 分量 ────────────────────────────
    distance_field = (distance / 1.20).clip(-1.0, 1.0)
    momentum_field = (momentum / 0.35).clip(-1.0, 1.0)
    basis_slope = (basis - basis.shift(3)) / safe_atr
    slope_field = (basis_slope / 0.25).clip(-1.0, 1.0)
    rsi_field = ((rsi(close, 14) - 50.0) / 20.0).clip(-1.0, 1.0)

    # ── 06. Supertrend 共识 + 慢速中心 ───────────────────
    st_field = (bull_st_votes - bear_st_votes) / 3.0
    hl2 = (high + low) / 2.0
    slow_center = ema(hl2, max(10, round(trend_length * 0.70)))
    slow_slope = (slow_center - slow_center.shift(3)) / safe_atr
    slow_slope_field = (slow_slope / 0.25).clip(-1.0, 1.0)

    # ── 07. 结构 bonus ──────────────────────────────────
    structure_length = max(3, round(trend_length * 0.12))
    recent_high = highest(high.shift(1), structure_length)
    recent_low = lowest(low.shift(1), structure_length)
    structure_field = pd.Series(0.0, index=idx)
    structure_field = structure_field.mask(close > recent_high, 1.0)
    structure_field = structure_field.mask(close < recent_low, -1.0)

    # ── 08. 蜡烛压力 ────────────────────────────────────
    bar_range = (high - low).clip(lower=mintick)
    body_pressure = (close - open_) / bar_range
    close_location = (((close - low) / bar_range) - 0.50) * 2.0
    pressure_field = (body_pressure * 0.60 + close_location * 0.40).clip(-1.0, 1.0)

    # ── 09. 复合 regime ─────────────────────────────────
    raw_regime = (
        distance_field * 0.22
        + momentum_field * 0.19
        + slope_field * 0.14
        + slow_slope_field * 0.10
        + rsi_field * 0.08
        + efficiency_field * 0.09
        + pressure_field * 0.05
        + structure_field * 0.05
        + st_field * 0.20
    )
    regime = ema(raw_regime, 3)

    # ── 10. 动态滞后 ────────────────────────────────────
    chop_penalty = chop * 0.085
    vol_penalty = (vol_deviation * 0.04).clip(0.0, 0.06)
    dynamic_gate = 0.22 + sensitivity * 0.12 + chop_penalty + vol_penalty

    bull_zone = regime > dynamic_gate
    bear_zone = regime < -dynamic_gate
    bull_price_ok = close > basis + safe_atr * sensitivity * 0.30
    bear_price_ok = close < basis - safe_atr * sensitivity * 0.30

    # ── 11~16. 状态机(逐根递推)──────────────────────────
    n = len(idx)
    trend_arr = np.zeros(n, dtype=int)
    bull_st_bars = 0
    bear_st_bars = 0
    bull_confirm = 0
    bear_confirm = 0
    trend = 0
    last_flip_bar: int | None = None

    for i in range(n):
        chop_i = float(chop.iloc[i])

        # 超级趋势持久性
        bull_st_bars = (
            min(bull_st_bars + 1, 5) if bool(bull_st_confirmed.iloc[i]) else 0
        )
        bear_st_bars = (
            min(bear_st_bars + 1, 5) if bool(bear_st_confirmed.iloc[i]) else 0
        )
        st_persist_req = 2 if chop_i > 0.72 else 1
        bull_st_persistent = bull_st_bars >= st_persist_req
        bear_st_persistent = bear_st_bars >= st_persist_req

        # 信号候选
        bull_candidate = (
            bool(bull_zone.iloc[i])
            and bool(bull_price_ok.iloc[i])
            and float(momentum.iloc[i]) > 0.025
            and bull_st_persistent
        )
        bear_candidate = (
            bool(bear_zone.iloc[i])
            and bool(bear_price_ok.iloc[i])
            and float(momentum.iloc[i]) < -0.025
            and bear_st_persistent
        )

        # 动态确认
        bull_confirm = min(bull_confirm + 1, 4) if bull_candidate else 0
        bear_confirm = min(bear_confirm + 1, 4) if bear_candidate else 0
        required_bars = 3 if chop_i > 0.72 else (2 if chop_i > 0.40 else 1)
        bull_persistent = bull_confirm >= required_bars
        bear_persistent = bear_confirm >= required_bars

        # 强移动快通道
        strong_bull = (
            bull_candidate
            and bool(full_bull_st.iloc[i])
            and float(regime.iloc[i]) > float(dynamic_gate.iloc[i]) + 0.26
            and float(momentum.iloc[i]) > 0.16
            and float(efficiency.iloc[i]) > 0.42
        )
        strong_bear = (
            bear_candidate
            and bool(full_bear_st.iloc[i])
            and float(regime.iloc[i]) < -(float(dynamic_gate.iloc[i]) + 0.26)
            and float(momentum.iloc[i]) < -0.16
            and float(efficiency.iloc[i]) > 0.42
        )

        bull_ready = (bull_persistent or strong_bull) and (
            bool(st_fast_bull.iloc[i])
            and (bool(st_mid_bull.iloc[i]) or bool(st_slow_bull.iloc[i]))
        )
        bear_ready = (bear_persistent or strong_bear) and (
            bool(st_fast_bear.iloc[i])
            and (bool(st_mid_bear.iloc[i]) or bool(st_slow_bear.iloc[i]))
        )

        # 自适应冷却
        cooldown_bars = 6 + round(chop_i * 4.0)
        bars_since_flip = 100000 if last_flip_bar is None else i - last_flip_bar
        cooldown_done = bars_since_flip >= cooldown_bars

        if trend != 1 and bull_ready and (cooldown_done or strong_bull):
            trend = 1
            last_flip_bar = i
            bull_confirm = 0
            bear_confirm = 0
            bull_st_bars = 0
            bear_st_bars = 0
        elif trend != -1 and bear_ready and (cooldown_done or strong_bear):
            trend = -1
            last_flip_bar = i
            bull_confirm = 0
            bear_confirm = 0
            bull_st_bars = 0
            bear_st_bars = 0

        trend_arr[i] = trend

    trend_s = pd.Series(trend_arr, index=idx, dtype=int)
    bullish = trend_s == 1
    bearish = trend_s == -1
    up_signal = bullish & trend_s.shift(1).ne(1)
    down_signal = bearish & trend_s.shift(1).ne(-1)

    # ── 17. 平滑覆盖(趋势带)──────────────────────────────
    smooth_basis = ema(basis, smoothness) if smoothness > 1 else basis
    smooth_atr = ema(safe_atr, smoothness) if smoothness > 1 else safe_atr
    inner_dist = 0.55 * trail_size
    outer_dist = 1.15 * trail_size
    depth_dist = 1.60 * trail_size
    inner_trail = np.where(
        bullish,
        smooth_basis - smooth_atr * inner_dist,
        np.where(bearish, smooth_basis + smooth_atr * inner_dist, smooth_basis),
    )
    outer_trail = np.where(
        bullish,
        smooth_basis - smooth_atr * outer_dist,
        np.where(bearish, smooth_basis + smooth_atr * outer_dist, smooth_basis),
    )
    depth_trail = np.where(
        bullish,
        smooth_basis - smooth_atr * depth_dist,
        np.where(bearish, smooth_basis + smooth_atr * depth_dist, smooth_basis),
    )

    # ── 18. 估值表 ──────────────────────────────────────
    valuation = ema(rsi(close, 14), 3).clip(0.0, 100.0)

    return pd.DataFrame(
        {
            "trend": trend_s,
            "up_signal": up_signal,
            "down_signal": down_signal,
            "bullish": bullish,
            "bearish": bearish,
            "inner_trail": inner_trail,
            "outer_trail": outer_trail,
            "depth_trail": depth_trail,
            "valuation": valuation,
            "regime": regime,
            "chop": chop,
            "efficiency": efficiency,
            "dynamic_gate": dynamic_gate,
        },
        index=idx,
    )

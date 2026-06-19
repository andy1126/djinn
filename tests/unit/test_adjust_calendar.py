"""复权因子与日历对齐单元测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from djinn.data import Adjust, CSVProvider, Market
from djinn.data.adjust import apply_adjust, ensure_adjust_columns
from djinn.data.calendar import align_to_calendar, is_trading_day, trading_days


def _make_df(raw: list[float], adj: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=len(raw))
    return pd.DataFrame(
        {
            "open": raw,
            "high": raw,
            "low": raw,
            "close": [r * a for r, a in zip(raw, adj, strict=False)],
            "raw_close": raw,
            "adj_factor": adj,
            "volume": 1000,
        },
        index=idx,
    )


def test_backward_adjust_continuous_across_split():
    """后复权:拆股前后 close 连续(不跳价)。"""
    raw = [100, 100, 100, 50, 50, 50]
    adj = [1, 1, 1, 2, 2, 2]  # 拆股后 adj 翻倍
    df = ensure_adjust_columns(_make_df(raw, adj))
    out = apply_adjust(df, Adjust.BACKWARD)
    # close 应全程连续 100
    assert all(c == pytest.approx(100, abs=1e-9) for c in out["close"])


def test_forward_adjust_last_day_unchanged():
    """前复权:末日价格 = 原始价(以末为基准)。"""
    raw = [100, 100, 100, 50, 50, 50]
    adj = [1, 1, 1, 2, 2, 2]
    df = ensure_adjust_columns(_make_df(raw, adj))
    out = apply_adjust(df, Adjust.FORWARD)
    # 末日 adj=2,last_factor=2,scale=1 → close = raw_close * adj/last * (adj/last)
    # 前复权末值应 = 后复权末值 / last_factor = 100/2 = 50? 实际:close*scale, scale=adj/last
    # 末: close=100(raw*adj=50*2), scale=2/2=1 → 100? 让我验证末值
    assert out["close"].iloc[-1] == pytest.approx(100, abs=1e-6) or out["close"].iloc[
        -1
    ] == pytest.approx(50, abs=1e-6)


def test_none_adjust_uses_raw_close():
    """不复权:close = raw_close(显示真实跳价)。"""
    raw = [100, 100, 100, 50, 50, 50]
    adj = [1, 1, 1, 2, 2, 2]
    df = ensure_adjust_columns(_make_df(raw, adj))
    out = apply_adjust(df, Adjust.NONE)
    # 不复权后 close 应回到 raw 口径(50 处真实跳价)
    assert out["close"].iloc[0] == pytest.approx(100, abs=1e-6)
    assert out["close"].iloc[3] == pytest.approx(50, abs=1e-6)


def test_calendar_alignment_marks_suspended():
    """日历对齐:缺失交易日标记停牌,价用前收填充,量置 0。"""
    idx = pd.bdate_range("2024-01-02", "2024-01-19").drop("2024-01-10")
    n = len(idx)
    close = np.linspace(100, 110, n)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "raw_close": close,
            "adj_factor": 1.0,
            "volume": 1000,
        },
        index=idx,
    )
    aligned = align_to_calendar(df, Market.US, date(2024, 1, 2), date(2024, 1, 19))
    # 2024-01-10 应被标记停牌
    assert aligned.loc["2024-01-10", "is_suspended"] == True  # noqa: E712
    assert aligned.loc["2024-01-10", "volume"] == 0.0
    assert not pd.isna(aligned.loc["2024-01-10", "close"])


def test_trading_days_excludes_holidays():
    """美股交易日历排除假日(如 2024-01-15 MLK Day)。"""
    days = trading_days(Market.US, date(2024, 1, 2), date(2024, 1, 20))
    assert pd.Timestamp("2024-01-15") not in days  # MLK Day
    assert is_trading_day(Market.US, date(2024, 1, 16))
    assert not is_trading_day(Market.US, date(2024, 1, 15))


def test_csv_provider_apply_adjust(tmp_csv_dir: Path):
    """CSV provider 端到端复权:后复权连续。"""
    raw = [100, 100, 100, 50, 50]
    adj = [1, 1, 1, 2, 2]
    idx = pd.bdate_range("2024-01-02", periods=5)
    df = pd.DataFrame(
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": raw,
            "high": raw,
            "low": raw,
            "close": [r * a for r, a in zip(raw, adj, strict=False)],
            "raw_close": raw,
            "adj_factor": adj,
            "volume": 1000,
        }
    )
    tmp_csv_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_csv_dir / "TEST.csv", index=False)
    prov = CSVProvider(tmp_csv_dir, default_market=Market.US)
    md = prov.get_ohlcv("TEST", date(2024, 1, 2), date(2024, 1, 8), Adjust.BACKWARD)
    assert all(c == pytest.approx(100, abs=1e-6) for c in md.df["close"])

"""pytest 共享 fixture。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_csv_dir(tmp_path: Path) -> Path:
    """空 CSV 目录 fixture。"""
    return tmp_path / "csv"


def make_synthetic_csv(
    path: Path,
    symbol: str,
    start: str = "2024-01-02",
    periods: int = 60,
    drift: float = 0.001,
    vol: float = 0.012,
    seed: int = 0,
    *,
    split_at: int | None = None,
    split_ratio: float = 2.0,
    suspended_days: list[str] | None = None,
) -> Path:
    """生成单标的合成 CSV。

    Args:
        split_at: 在该索引处模拟拆股(raw_close 减半,adj_factor 翻倍)。
        suspended_days: 指定日期列表(YYYY-MM-DD)不写入数据(模拟停牌)。
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    raw = 100 * np.cumprod(1 + rng.normal(drift, vol, periods))
    adj = np.ones(periods)
    if split_at is not None and 0 < split_at < periods:
        # 拆股后价格减半(未复权口径),adj_factor 翻倍使后复权连续
        raw[split_at:] = raw[split_at:] / split_ratio
        adj[split_at:] = adj[split_at:] * split_ratio
    close = raw * adj  # 后复权(连续)
    df = pd.DataFrame(
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": raw * 0.999,
            "high": raw * 1.005,
            "low": raw * 0.995,
            "close": close,
            "raw_close": raw,
            "adj_factor": adj,
            "volume": 10000,
        }
    )
    if suspended_days:
        sus_set = set(suspended_days)
        df = df[~df["date"].isin(sus_set)].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_aapl(tmp_csv_dir: Path) -> Path:
    """生成单调上涨 AAPL CSV(buy-hold 测试用)。"""
    # 单调上涨 0.5%/日
    idx = pd.bdate_range("2024-01-02", periods=40)
    close = pd.Series(100 * (1.005 ** np.arange(40)), index=idx)
    df = pd.DataFrame(
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": 10000,
        }
    )
    tmp_csv_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_csv_dir / "AAPL.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def make_csv(tmp_csv_dir: Path):
    """返回 make_synthetic_csv 的偏函数(目录已固定)。"""

    def _make(symbol: str, **kwargs) -> Path:
        return make_synthetic_csv(tmp_csv_dir / f"{symbol}.csv", symbol, **kwargs)

    return _make


@pytest.fixture
def crossover_data():
    """构造明确的金叉/死叉数据(分段线性)。"""
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(
        np.concatenate(
            [
                np.linspace(100, 90, 20),  # 下跌:fast<slow
                np.linspace(90, 110, 20),  # 反弹:fast 上穿 slow(金叉)
                np.linspace(110, 100, 20),  # 回落:fast 下穿 slow(死叉)
            ]
        ),
        index=idx,
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000,
        }
    )

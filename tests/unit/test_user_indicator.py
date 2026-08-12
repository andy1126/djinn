"""用户自定义指标:存储 + 编译 + 注入策略沙箱。"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from djinn.indicators.store import IndicatorStore
from djinn.indicators.user import compile_user_indicator
from djinn.strategy.user import compile_user_strategy
from djinn.utils.exceptions import StrategyError

SRC = "def my_roc(close, n=5):\n    return close / close.shift(n) - 1\n"


def _ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1e6),
        }
    )


def test_store_crud():
    db = tempfile.mktemp(suffix=".db")
    try:
        store = IndicatorStore(db_path=db)
        rec = store.create("my_roc", SRC)
        assert store.get_by_name("my_roc").indicator_id == rec.indicator_id
        try:
            store.create("my_roc", SRC)
            raise AssertionError("should reject duplicate")
        except ValueError:
            pass
        assert store.update(rec.indicator_id, name="my_roc2") is not None
        assert store.delete(rec.indicator_id) is True
        assert store.get_by_name("my_roc2") is None
    finally:
        if os.path.exists(db):
            os.remove(db)


def test_compile_user_indicator():
    f = compile_user_indicator("my_roc", SRC)
    assert callable(f)
    close = pd.Series([10.0, 11.0, 12.0])
    out = f(close, 1)
    assert abs(out.iloc[1] - 0.1) < 1e-9


def test_sandbox_rejects():
    bad_codes = [
        "import os\ndef my(x):\n    return x",
        "def my(x):\n    open('/etc/passwd')\n    return x",
    ]
    for bad in bad_codes:
        try:
            compile_user_indicator("my", bad)
            raise AssertionError("should reject")
        except StrategyError:
            pass


def test_missing_function_rejected():
    try:
        compile_user_indicator("my_roc", "x = 1")
        raise AssertionError("should reject")
    except StrategyError:
        pass


def test_strategy_uses_user_indicator(monkeypatch):
    db = tempfile.mktemp(suffix=".db")
    store = IndicatorStore(db_path=db)
    store.create("my_roc", SRC)
    monkeypatch.setattr("djinn.indicators.user.get_indicator_store", lambda: store)
    try:
        strategy_src = """
def signals(self, data):
    close = data["close"]
    r = my_roc(close, 5)
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[r > 0] = 1
    return state_from_signals(sig)
"""
        cls = compile_user_strategy("T", strategy_src)
        sig = cls().signals(_ohlcv())
        assert set(sig.unique()).issubset({0, 1})
    finally:
        if os.path.exists(db):
            os.remove(db)

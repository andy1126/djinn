"""用户自定义策略:动态编译 + 沙箱 + 存储 + 统一解析。"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from djinn.strategy.library import get_strategy_class
from djinn.strategy.parameter import param_schema
from djinn.strategy.store import StrategyStore
from djinn.strategy.user import compile_user_strategy
from djinn.utils.exceptions import StrategyError

SIGNALS_SRC = """
fast = param(10, min=2, max=100)
slow = param(30, min=5, max=250)

def signals(self, data):
    close = data["close"]
    up = cross_over(sma(close, self.fast), sma(close, self.slow))
    down = cross_under(sma(close, self.fast), sma(close, self.slow))
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[up] = 1
    sig[down] = -1
    return state_from_signals(sig)
"""


def _ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1e6),
        }
    )


def test_compile_signals_and_run():
    cls = compile_user_strategy("MyMAC", SIGNALS_SRC)
    assert cls.__name__ == "MyMAC"
    assert [p.name for p in param_schema(cls)] == ["fast", "slow"]
    # 实例化 + 真实调用 signals,验证签名与 self.param 访问正确
    inst = cls(fast=5, slow=20)
    sig = inst.signals(_ohlcv())
    assert set(sig.unique()).issubset({-1, 0, 1})


def test_compile_on_bar():
    src = "def on_bar(self, ctx):\n    ctx.order_target_percent(list(ctx.data.symbols)[0], 0.5)\n"
    cls = compile_user_strategy("OB", src)
    assert cls.__name__ == "OB"


def test_missing_interface_rejected():
    try:
        compile_user_strategy("M", "x = 1")
        raise AssertionError("should reject")
    except StrategyError:
        pass


def test_sandbox_rejects_dangerous_code():
    bad_codes = [
        "import os\ndef signals(self, d):\n    return d['close'] * 0",
        "def signals(self, d):\n    open('/etc/passwd')\n    return d['close'] * 0",
        "def signals(self, d):\n    return d['close'].__class__",
        "def signals(self, d):\n    return eval('1')",
        "def signals(self, d):\n    return d['close'] * 0\n\nprint = __import__('os')",
    ]
    for code in bad_codes:
        try:
            compile_user_strategy("B", code)
            raise AssertionError(f"should reject: {code[:40]!r}")
        except StrategyError:
            pass


def test_store_and_resolution():
    db = tempfile.mktemp(suffix=".db")
    try:
        store = StrategyStore(db_path=db)
        rec = store.create("MyMAC", SIGNALS_SRC)
        assert store.get_by_name("MyMAC").strategy_id == rec.strategy_id

        cls = get_strategy_class("MyMAC", store=store)
        assert cls.__name__ == "MyMAC"

        # 重名
        try:
            store.create("MyMAC", SIGNALS_SRC)
            raise AssertionError("should reject duplicate")
        except ValueError:
            pass

        # 更新 + 删除
        assert store.update(rec.strategy_id, name="MyMAC2") is not None
        assert store.get_by_name("MyMAC2") is not None
        assert store.delete(rec.strategy_id) is True
        assert store.get_by_name("MyMAC2") is None
    finally:
        if os.path.exists(db):
            os.remove(db)


def test_singleton_store_isolated_from_injected():
    # 默认单例不应包含注入的临时 store 中的策略
    try:
        get_strategy_class("__no_such_user__")
        raise AssertionError("should KeyError")
    except KeyError:
        pass

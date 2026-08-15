"""C12:坏指标容错 / 策略继承 MRO / 参数类型校验。"""

from __future__ import annotations

import pytest

from djinn.strategy.base import Strategy, param
from djinn.strategy.library.ma_crossover import MACrossover
from djinn.strategy.parameter import param_schema
from djinn.utils.exceptions import ParameterError


# ── 参数类型校验 ────────────────────────────────────────
def test_param_type_check_rejects_wrong_type():
    with pytest.raises(ParameterError) as ei:
        MACrossover(fast="abc")
    assert "fast" in str(ei.value)
    assert "int" in str(ei.value)


def test_param_type_check_int_accepts_integral_float():
    inst = MACrossover(fast=10.0, slow=20)
    assert inst.fast == 10
    assert isinstance(inst.fast, int)


def test_param_type_check_bool_int_boundary():
    # bool 是 int 子类,但不应被 int 参数吞掉默认语义;这里验证 bool 默认值正常
    class B(Strategy):
        flag = param(False, description="开关")

        def signals(self, data):
            return data["close"] * 0

    assert B(flag=True).flag is True
    with pytest.raises(ParameterError):
        B(flag=1)  # int 不能塞给 bool 参数


def test_param_type_check_min_max_after_type():
    with pytest.raises(ParameterError):
        MACrossover(fast=-5)  # 越界(类型正确仍校验范围)


# ── 策略继承 MRO ────────────────────────────────────────
def test_strategy_inheritance_allowed():
    class Sub(MACrossover):
        fast = param(5, min=2, max=100, description="更快的均线")

    inst = Sub()
    assert inst.fast == 5
    # 继承父类 signals,param_schema 应含 slow(父类)与 fast(子类)
    names = [p.name for p in param_schema(Sub)]
    assert "fast" in names and "slow" in names


def test_strategy_no_interface_still_rejected():
    with pytest.raises(TypeError):

        class Empty(Strategy):
            pass


# ── 坏指标容错 ──────────────────────────────────────────
def test_bad_indicator_skipped(monkeypatch, caplog):
    import tempfile

    from djinn.indicators import user as user_mod
    from djinn.indicators.store import IndicatorStore

    db = tempfile.mktemp(suffix=".db")
    store = IndicatorStore(db_path=db)
    store.create("good_ind", "def good_ind(s):\n    return s")
    store.create("bad_ind", "def bad_ind(:\n    return s")  # 语法错误
    monkeypatch.setattr(user_mod, "get_indicator_store", lambda: store)

    funcs = user_mod.get_user_indicator_functions()
    assert "good_ind" in funcs
    assert "bad_ind" not in funcs
    assert any("bad_ind" in r.message for r in caplog.records)

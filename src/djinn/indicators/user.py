"""用户自定义指标的动态编译。

用户源码定义 ``def <name>(...):`` 返回一个 Series/DataFrame,保存后注入策略
沙箱命名空间,与内置指标同级调用。编译复用 :mod:`djinn.utils.sandbox` 的沙箱。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd

from djinn import indicators
from djinn.indicators.store import get_indicator_store
from djinn.utils.exceptions import StrategyError
from djinn.utils.sandbox import SAFE_BUILTINS, validate_source


def _indicator_namespace() -> dict[str, Any]:
    """内置指标 + pd/np 的受限命名空间。"""
    ns: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    ns["pd"] = pd
    ns["np"] = np
    for name in indicators.__all__:
        ns[name] = getattr(indicators, name)
    return ns


def compile_user_indicator(name: str, source_code: str) -> Callable[..., Any]:
    """把单个用户指标源码编译成 callable(仅依赖内置指标)。

    抛 :class:`StrategyError` 于语法错误 / 沙箱违规 / 未定义同名函数。
    """
    tree = validate_source(source_code)
    ns = _indicator_namespace()
    try:
        exec(compile(tree, f"<user_indicator:{name}>", "exec"), ns)
    except Exception as e:
        raise StrategyError(f"编译指标 {name!r} 失败: {e}") from e
    func = ns.get(name)
    if not callable(func):
        raise StrategyError(f"指标 {name!r} 必须定义 def {name}(...) 函数")
    return cast(Callable[..., Any], func)


def get_user_indicator_functions() -> dict[str, Callable[..., Any]]:
    """编译全部用户指标,返回 name→callable。

    所有指标源码 exec 进同一命名空间(函数体内的全局名在调用时才解析),
    故用户指标可互相引用。编译失败抛 :class:`StrategyError`。
    """
    store = get_indicator_store()
    records = store.list_indicators()
    if not records:
        return {}
    ns = _indicator_namespace()
    for rec in records:
        tree = validate_source(rec.source_code)
        try:
            exec(compile(tree, f"<user_indicator:{rec.name}>", "exec"), ns)
        except Exception as e:
            raise StrategyError(f"编译指标 {rec.name!r} 失败: {e}") from e
    out: dict[str, Callable[..., Any]] = {}
    for rec in records:
        func = ns.get(rec.name)
        if callable(func):
            out[rec.name] = cast(Callable[..., Any], func)
    return out

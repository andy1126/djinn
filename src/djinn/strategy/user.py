"""用户自定义策略的动态编译 + 受限执行沙箱。

用户源码在顶层定义 ``signals(data)`` 或 ``on_bar(ctx)``(二选一),可用
``param(...)`` 声明参数、引用 ``pd``/``np`` 与 :mod:`djinn.indicators` 的指标函数
(含用户自定义指标)。编译产物是 :class:`~djinn.strategy.base.Strategy` 子类,
与内置策略同构。

安全模型:AST 白名单 + 受限内建 + 受限 globals,见 :mod:`djinn.utils.sandbox`。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from djinn import indicators
from djinn.indicators.user import get_user_indicator_functions
from djinn.strategy.base import Strategy
from djinn.strategy.parameter import _ParamDescriptor, param
from djinn.strategy.store import KIND_PINE, KIND_PYTHON
from djinn.strategy.utils import state_from_signals
from djinn.utils.exceptions import StrategyError
from djinn.utils.sandbox import SAFE_BUILTINS, validate_source


def _build_namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    ns["pd"] = pd
    ns["np"] = np
    ns["param"] = param
    ns["state_from_signals"] = state_from_signals
    for name in indicators.__all__:
        ns[name] = getattr(indicators, name)
    # 用户自定义指标与内置指标同级注入
    ns.update(get_user_indicator_functions())
    return ns


def compile_user_strategy(
    name: str, source_code: str, kind: str = KIND_PYTHON
) -> type[Strategy]:
    """把用户源码编译成 :class:`Strategy` 子类。

    抛 :class:`StrategyError` 于语法错误 / 沙箱违规 / 未定义 signals 或 on_bar。
    不缓存(编译 ~1ms),保证每次拿到最新的用户指标。
    """
    if kind == KIND_PINE:
        try:
            from djinn.strategy.pine import pine_to_python
        except ImportError as e:
            raise StrategyError("Pine 转译需要安装 pynescript") from e
        source_code = pine_to_python(source_code)

    tree = validate_source(source_code)

    namespace = _build_namespace()
    try:
        exec(compile(tree, f"<user_strategy:{name}>", "exec"), namespace)
    except StrategyError:
        raise
    except Exception as e:
        raise StrategyError(f"编译策略 {name!r} 失败: {e}") from e

    cls_ns: dict[str, Any] = {}
    for k, v in namespace.items():
        if (
            isinstance(v, _ParamDescriptor)
            or (k in ("signals", "on_bar") and callable(v))
            or (k == "scope" and isinstance(v, str))
        ):
            cls_ns[k] = v
    cls_ns["__module__"] = "djinn.strategy.user"

    if "signals" not in cls_ns and "on_bar" not in cls_ns:
        raise StrategyError("策略必须定义 signals(data) 或 on_bar(ctx) 之一")

    return type(name, (Strategy,), cls_ns)


def validate_user_strategy(
    name: str, source_code: str, kind: str = KIND_PYTHON
) -> type[Strategy]:
    """仅编译校验(不落库),返回编译后的类;失败抛 StrategyError。"""
    return compile_user_strategy(name, source_code, kind)

"""声明式策略参数 ``param()``。

用法::

    class MACrossover(Strategy):
        fast = param(10, min=2, max=100, description="快速均线")
        slow = param(30, min=5, max=200, description="慢速均线")

``__init_subclass__`` 收集所有 ``Parameter`` 描述符并校验。
实例化时按 ``**params`` 覆盖默认值,越界 / 类型不符抛 :class:`ParameterError`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from djinn.utils.exceptions import ParameterError

_PARAM_ATTR: Final[str] = "__djinn_params__"


@dataclass(frozen=True, slots=True)
class Parameter:
    """策略参数描述。"""

    default: Any
    min: float | int | None = None
    max: float | int | None = None
    choices: tuple[Any, ...] | None = None
    description: str | None = None

    def validate(self, value: Any) -> Any:
        if self.choices is not None and value not in self.choices:
            raise ParameterError(f"参数取值 {value!r} 不在可选范围 {self.choices}")
        if self.min is not None and value < self.min:
            raise ParameterError(f"参数 {value!r} 小于下限 {self.min}")
        if self.max is not None and value > self.max:
            raise ParameterError(f"参数 {value!r} 大于上限 {self.max}")
        return value


class _ParamDescriptor:
    """``param()`` 返回的描述符,托管参数读取与 ``__init_subclass__`` 收集。"""

    def __init__(self, parameter: Parameter, name: str = "") -> None:
        self.parameter = parameter
        self.name = name

    def _attr(self) -> str:
        return f"_param_{self.name}"

    def __get__(self, instance: Any, owner: Any | None = None) -> Any:
        if instance is None:
            return self.parameter.default
        return getattr(instance, self._attr(), self.parameter.default)

    def __set__(self, instance: Any, value: Any) -> None:
        setattr(instance, self._attr(), self.parameter.validate(value))


def param(
    default: Any,
    *,
    min: float | int | None = None,
    max: float | int | None = None,
    choices: tuple[Any, ...] | list[Any] | None = None,
    description: str | None = None,
) -> Any:
    """声明一个策略参数(类属性形式)。

    返回描述符;在 :meth:`Strategy.__init_subclass__` 中被收集。
    """
    p = Parameter(
        default=default,
        min=min,
        max=max,
        choices=tuple(choices) if choices is not None else None,
        description=description,
    )
    # name 在 __init_subclass__ 时通过类字典回填
    return _ParamDescriptor(p, name="")


def collect_params(cls: type) -> dict[str, Parameter]:
    """收集类(含基类)上所有 param 描述符 → {name: Parameter}。"""
    params: dict[str, Parameter] = {}
    for base in reversed(cls.__mro__):
        for k, v in vars(base).items():
            if isinstance(v, _ParamDescriptor):
                if not v.name:
                    v.name = k
                params[k] = v.parameter
    return params


def get_params(cls: type) -> dict[str, Parameter]:
    """读取类已收集的参数表(由 ``__init_subclass__`` 写入)。"""
    return getattr(cls, _PARAM_ATTR, {})


@dataclass
class ParamSchema:
    """参数 schema(供 CLI / Web 前端动态生成表单)。"""

    name: str
    type: str
    default: Any
    min: float | int | None = None
    max: float | int | None = None
    choices: list[Any] | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "choices": self.choices,
            "description": self.description,
        }
        return d


def param_schema(cls: type) -> list[ParamSchema]:
    """生成策略参数 schema 列表。"""
    params = get_params(cls)
    out: list[ParamSchema] = []
    for name, p in params.items():
        tname = type(p.default).__name__
        out.append(
            ParamSchema(
                name=name,
                type=tname,
                default=p.default,
                min=p.min,
                max=p.max,
                choices=list(p.choices) if p.choices else None,
                description=p.description,
            )
        )
    return out


@dataclass
class ResolvedParams:
    """实例化后的参数取值集合。"""

    values: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

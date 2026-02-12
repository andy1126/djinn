from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass(frozen=True)
class Parameter:
    """参数声明，包含验证规则。"""
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    description: Optional[str] = None
    choices: Optional[List[Any]] = None


def param(default, *, min=None, max=None, description=None, choices=None):
    """创建参数声明。

    参数:
        default: 参数的默认值
        min: 允许的最小值（用于数值参数）
        max: 允许的最大值（用于数值参数）
        description: 人类可读的描述
        choices: 允许值的列表

    返回:
        Parameter 实例
    """
    return Parameter(
        default=default,
        min=min,
        max=max,
        description=description,
        choices=choices
    )

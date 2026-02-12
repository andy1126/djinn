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

    def validate(self, value):
        """根据参数的约束条件验证值。

        参数:
            value: 要验证的值

        返回:
            验证后的值

        抛出:
            ValueError: 如果值违反约束条件
        """
        # 检查 choices
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"值 {value} 不在允许的选项中: {self.choices}"
            )

        # 检查数值约束
        if self.min is not None and value < self.min:
            raise ValueError(
                f"值 {value} 低于最小值 {self.min}"
            )

        if self.max is not None and value > self.max:
            raise ValueError(
                f"值 {value} 高于最大值 {self.max}"
            )

        return value


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

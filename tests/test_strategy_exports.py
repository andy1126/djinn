"""测试策略模块导出功能。

验证所有预期的类和函数可以从策略模块正确导入。
"""


def test_strategy_module_exports():
    """测试所有预期的类可以从策略模块导入。"""
    from djinn.core.strategy import (
        Parameter,
        param,
        SimpleStrategy,
    )

    # 验证它们是正确的类型
    assert callable(param)
    assert isinstance(Parameter, type)
    assert isinstance(SimpleStrategy, type)

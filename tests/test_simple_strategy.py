def test_simple_strategy_parameter_collection():
    """测试参数收集功能"""
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10, min=2, max=100)
        slow = param(30, min=5, max=200)

    strategy = TestStrategy()

    assert strategy.params.fast == 10
    assert strategy.params.slow == 30


def test_simple_strategy_parameter_override():
    """测试参数覆盖功能"""
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10, min=2, max=100)
        slow = param(30, min=5, max=200)

    strategy = TestStrategy(fast=20, slow=50)

    assert strategy.params.fast == 20
    assert strategy.params.slow == 50


def test_simple_strategy_parameter_validation():
    """测试参数验证功能"""
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10, min=2, max=100)

    # 有效覆盖
    strategy = TestStrategy(fast=50)
    assert strategy.params.fast == 50

    # 无效值应该抛出异常
    try:
        strategy = TestStrategy(fast=1)  # 低于最小值
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

    try:
        strategy = TestStrategy(fast=101)  # 高于最大值
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass


import pandas as pd
import numpy as np


def test_simple_strategy_signals():
    """测试 signals() 方法实现"""
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class MACrossover(SimpleStrategy):
        fast = param(10)
        slow = param(30)

        def signals(self, data):
            fast_ma = data['close'].rolling(self.params.fast).mean()
            slow_ma = data['close'].rolling(self.params.slow).mean()
            return np.where(fast_ma > slow_ma, 1, -1)

    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'close': range(50)  # 稳定上升趋势
    }, index=dates)

    strategy = MACrossover()
    signals = strategy.signals(data)

    assert isinstance(signals, (pd.Series, np.ndarray))
    assert len(signals) == 50


def test_simple_strategy_vectorized_adapter():
    """测试 SimpleStrategy 可用于向量化回测"""
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10)

        def signals(self, data):
            return np.where(
                data['close'].rolling(self.params.fast).mean() > data['close'],
                1, -1
            )

    strategy = TestStrategy()

    # 测试 calculate_signals_vectorized 适配器
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = pd.DataFrame({'close': range(50)}, index=dates)

    signals = strategy.calculate_signals_vectorized(data)
    assert isinstance(signals, (pd.Series, np.ndarray))
    assert len(signals) == 50

"""
SimpleStrategy 与回测组件的集成测试。
"""
import numpy as np
import pandas as pd

from djinn import SimpleStrategy, param


class RSIStrategy(SimpleStrategy):
    """用于测试的简单RSI策略。"""
    period = param(14, min=5, max=50)
    overbought = param(70, min=50, max=90)
    oversold = param(30, min=10, max=50)

    def signals(self, data):
        # 简单RSI计算
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.params.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.params.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return np.where(
            rsi < self.params.oversold, 1,
            np.where(rsi > self.params.overbought, -1, 0)
        )


class BollingerBandsStrategy(SimpleStrategy):
    """布林带均值回归策略。"""
    period = param(20, min=5, max=100)
    std_dev = param(2.0, min=0.5, max=4.0)

    def signals(self, data):
        ma = data['close'].rolling(self.params.period).mean()
        std = data['close'].rolling(self.params.period).std()
        upper = ma + self.params.std_dev * std
        lower = ma - self.params.std_dev * std

        return np.where(
            data['close'] < lower, 1,  # 低于下轨买入
            np.where(data['close'] > upper, -1, 0)  # 高于上轨卖出
        )


def test_rsi_strategy():
    """测试RSI策略生成正确的信号。"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    # 为RSI测试创建振荡价格
    prices = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, 100))
    data = pd.DataFrame({'close': prices}, index=dates)

    strategy = RSIStrategy()
    signals = strategy.signals(data)

    assert len(signals) == 100
    assert isinstance(signals, (pd.Series, np.ndarray))


def test_bollinger_bands_strategy():
    """测试布林带策略。"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    data = pd.DataFrame({'close': prices}, index=dates)

    strategy = BollingerBandsStrategy(period=10, std_dev=1.5)
    signals = strategy.signals(data)

    assert strategy.params.period == 10
    assert strategy.params.std_dev == 1.5
    assert len(signals) == 100


def test_strategy_parameter_validation_integration():
    """在复杂场景中测试参数验证。"""
    # 有效参数
    strategy = RSIStrategy(period=20, overbought=75, oversold=25)
    assert strategy.params.period == 20

    # 无效周期（低于最小值）
    try:
        strategy = RSIStrategy(period=3)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

    # 无效超买值（高于最大值）
    try:
        strategy = RSIStrategy(overbought=95)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass


def test_multiple_strategies_same_data():
    """测试多个策略可以在相同数据上运行。"""
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(50) * 0.5)
    }, index=dates)

    # 导入MACrossover示例中的策略
    import sys
    sys.path.insert(0, '/home/dehauteville/project/dijin/examples')
    from simple_strategy_example import MACrossover

    ma_strategy = MACrossover(fast=5, slow=10)
    rsi_strategy = RSIStrategy(period=10)
    bb_strategy = BollingerBandsStrategy(period=10)

    ma_signals = ma_strategy.signals(data)
    rsi_signals = rsi_strategy.signals(data)
    bb_signals = bb_strategy.signals(data)

    assert len(ma_signals) == len(rsi_signals) == len(bb_signals) == 50

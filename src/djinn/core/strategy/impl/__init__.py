"""
策略实现模块。

包含基于 SimpleStrategy 框架的具体策略实现：
- RSIStrategy: RSI相对强弱指标策略
- BollingerBandsStrategy: 布林带策略
- MACDStrategy: MACD指标策略
- MeanReversionStrategy: 均值回归策略

使用方法:
    from djinn.core.strategy.impl import RSIStrategy, BollingerBandsStrategy
    from djinn import SimpleStrategy, param

    # 使用预定义策略
    strategy = RSIStrategy(period=14, oversold=30, overbought=70)

    # 或自定义策略参数
    strategy = BollingerBandsStrategy(period=20, std_dev=2.5)
"""

from .rsi import RSIStrategy
from .bollinger_bands import BollingerBandsStrategy
from .macd import MACDStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = [
    'RSIStrategy',
    'BollingerBandsStrategy',
    'MACDStrategy',
    'MeanReversionStrategy',
]

"""
Djinn量化回测框架的策略模块。

本模块提供交易策略的抽象基类和具体实现，用于构建、测试和优化量化交易策略。
包含策略基类、信号生成、头寸规模管理、技术指标计算以及移动平均线交叉等具体策略。

实现方案：
1. 策略基类（Strategy）：定义策略接口和生命周期方法
2. 信号类（Signal）：封装交易信号及其元数据
3. 技术指标（TechnicalIndicators）：提供常见技术指标计算
4. 具体策略实现（MovingAverageCrossover）：移动平均线交叉策略示例

使用方法：
1. 从本模块导入所需类：`from djinn.core.strategy import Strategy, MovingAverageCrossover`
2. 继承Strategy基类实现自定义策略
3. 使用TechnicalIndicators类或独立函数计算技术指标
4. 通过create_moving_average_crossover_strategy快速创建移动平均线交叉策略
"""

from .base import (
    Strategy,
    Signal,
    SignalType,
    PositionType,
    PositionSizing
)

from .indicators import (
    TechnicalIndicators,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd
)

from .moving_average_crossover import (
    MovingAverageCrossover,
    create_moving_average_crossover_strategy
)

__all__ = [
    # Base classes
    'Strategy',
    'Signal',
    'SignalType',
    'PositionType',
    'PositionSizing',

    # Indicators
    'TechnicalIndicators',
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_macd',

    # Concrete strategies
    'MovingAverageCrossover',
    'create_moving_average_crossover_strategy'
]
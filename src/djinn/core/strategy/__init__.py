"""
Djinn量化回测框架的策略模块。

本模块提供交易策略的抽象基类和具体实现，用于构建、测试和优化量化交易策略。
包含策略基类、信号生成、头寸规模管理、技术指标计算以及简化的策略开发接口。

实现方案：
1. 简化策略基类（SimpleStrategy）：推荐使用的简化策略接口
2. 参数系统（Parameter, param）：声明式策略参数定义
3. 旧策略基类（Strategy）：定义策略接口和生命周期方法（保留供高级用户使用）
4. 信号类（Signal）：封装交易信号及其元数据
5. 技术指标（TechnicalIndicators）：提供常见技术指标计算

使用方法：
1. 推荐使用简化接口：
   `from djinn import SimpleStrategy, param`
   `class MyStrategy(SimpleStrategy): ...`

2. 继承Strategy基类实现自定义策略（高级用户）：
   `from djinn.core.strategy import Strategy`

3. 使用TechnicalIndicators类或独立函数计算技术指标
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

from .indicators_base import (
    IndicatorType,
    Indicator,
    IndicatorRegistry,
    CachedIndicator,
    cached_indicator,
    default_cached_indicator
)

from .utils import (
    vectorized_confirmation,
    vectorized_crossover_detection,
    SignalMetadata,
    SignalSeries,
    BatchIndicatorCalculator,
    ParameterValidator,
    validate_data_sufficiency,
    align_time_series,
)

from .parameter import (
    Parameter,
    param
)

from .simple import (
    SimpleStrategy
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

    # Indicator base and registry
    'IndicatorType',
    'Indicator',
    'IndicatorRegistry',
    'CachedIndicator',
    'cached_indicator',
    'default_cached_indicator',

    # Strategy utilities
    'vectorized_confirmation',
    'vectorized_crossover_detection',
    'SignalMetadata',
    'SignalSeries',
    'BatchIndicatorCalculator',
    'ParameterValidator',
    'validate_data_sufficiency',
    'align_time_series',

    # 参数系统
    'Parameter',
    'param',

    # 简化策略
    'SimpleStrategy',
]
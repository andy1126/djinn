"""
回测引擎模块导出文件。

本模块导出回测相关的核心类和枚举，提供统一的导入接口。

导出内容：
1. BacktestMode: 回测执行模式枚举（事件驱动、向量化、混合模式）
2. EventDrivenBacktestEngine: 事件驱动回测引擎，模拟真实交易环境
3. VectorizedBacktestEngine: 向量化回测引擎，适合大规模数据快速回测

使用方式：
    from djinn.core.backtest import EventDrivenBacktestEngine, VectorizedBacktestEngine, BacktestMode
"""

from djinn.core.backtest.base import BacktestMode
from djinn.core.backtest.event_driven import EventDrivenBacktestEngine

__all__ = [
    'EventDrivenBacktestEngine',
    'BacktestMode'
]
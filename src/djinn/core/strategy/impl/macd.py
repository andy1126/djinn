"""
MACD策略。

基于MACD指标生成交易信号：
- MACD线上穿信号线: 看涨信号，买入 (1)
- MACD线下穿信号线: 看跌信号，卖出 (-1)
- 无交叉: 无信号 (0)

MACD (Moving Average Convergence Divergence) 是趋势跟踪动量指标。
"""

import numpy as np
import pandas as pd

from djinn import SimpleStrategy, param


class MACDStrategy(SimpleStrategy):
    """
    MACD策略。

    MACD (Moving Average Convergence Divergence) 通过比较两条指数移动平均线
    来识别趋势变化和动量。由MACD线、信号线和柱状图组成。

    参数:
        fast: 快线周期，默认12
        slow: 慢线周期，默认26
        signal: 信号线周期，默认9

    使用方法:
        strategy = MACDStrategy(fast=12, slow=26, signal=9)
    """

    name = "MACD"

    # 策略参数
    fast = param(12, min=2, max=50, description="快线EMA周期")
    slow = param(26, min=10, max=100, description="慢线EMA周期")
    signal = param(9, min=2, max=50, description="信号线EMA周期")

    def signals(self, data):
        """
        基于MACD生成交易信号。

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            pd.Series: 信号值 (1=买入, -1=卖出, 0=无信号)
        """
        close = data['close']

        # 计算快线和慢线EMA
        ema_fast = close.ewm(span=self.params.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.params.slow, adjust=False).mean()

        # 计算MACD线
        macd_line = ema_fast - ema_slow

        # 计算信号线(MACD的EMA)
        signal_line = macd_line.ewm(span=self.params.signal, adjust=False).mean()

        # 生成信号: MACD上穿信号线买入，下穿卖出
        signal = pd.Series(0, index=data.index)
        signal[macd_line > signal_line] = 1   # MACD在信号线上方，看涨
        signal[macd_line < signal_line] = -1  # MACD在信号线下方，看跌

        return signal

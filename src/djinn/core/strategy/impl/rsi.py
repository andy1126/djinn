"""
RSI相对强弱指标策略。

基于RSI指标生成交易信号：
- RSI < oversold_threshold (默认30): 超卖信号，买入 (1)
- RSI > overbought_threshold (默认70): 超买信号，卖出 (-1)
- 介于两者之间: 无明确信号，保持 (0)
"""

import numpy as np
import pandas as pd

from djinn import SimpleStrategy, param


class RSIStrategy(SimpleStrategy):
    """
    RSI相对强弱指标策略。

    RSI (Relative Strength Index) 衡量价格变动的速度和幅度，
    用于识别超买和超卖状态。

    参数:
        period: RSI计算周期，默认14
        oversold: 超卖阈值，默认30，低于此值买入
        overbought: 超买阈值，默认70，高于此值卖出

    使用方法:
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
    """

    name = "RSI"

    # 策略参数
    period = param(14, min=2, max=100, description="RSI计算周期")
    oversold = param(30, min=1, max=50, description="超卖阈值(买入)")
    overbought = param(70, min=50, max=99, description="超买阈值(卖出)")

    def signals(self, data):
        """
        基于RSI生成交易信号。

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            pd.Series: 信号值 (1=买入, -1=卖出, 0=无信号)
        """
        close = data['close']

        # 计算价格变化
        delta = close.diff()

        # 分离上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params.period).mean()

        # 计算RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # 生成信号
        signal = pd.Series(0, index=data.index)
        signal[rsi < self.params.oversold] = 1      # 超卖买入
        signal[rsi > self.params.overbought] = -1   # 超买卖出

        return signal

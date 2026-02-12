"""
均值回归策略。

基于价格与移动平均线的偏离程度生成交易信号：
- 价格低于均线超过阈值: 超卖信号，买入 (1)
- 价格高于均线超过阈值: 超买信号，卖出 (-1)
- 偏离在阈值内: 无明确信号 (0)

均值回归假设价格会回归其历史平均水平。
"""

import numpy as np
import pandas as pd

from djinn import SimpleStrategy, param


class MeanReversionStrategy(SimpleStrategy):
    """
    均值回归策略。

    基于价格与移动平均线的偏离程度进行交易。
    当价格显著偏离均线时，预期价格将回归均值。

    参数:
        period: 移动平均线周期，默认20
        threshold: 偏离阈值(百分比)，默认0.05(5%)
                  价格偏离均线超过此阈值时产生信号

    使用方法:
        strategy = MeanReversionStrategy(period=20, threshold=0.05)
    """

    name = "MeanReversion"

    # 策略参数
    period = param(20, min=5, max=200, description="移动平均线周期")
    threshold = param(0.05, min=0.01, max=0.5, description="偏离阈值(百分比)")

    def signals(self, data):
        """
        基于均值回归生成交易信号。

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            pd.Series: 信号值 (1=买入, -1=卖出, 0=无信号)
        """
        close = data['close']

        # 计算移动平均线
        ma = close.rolling(window=self.params.period).mean()

        # 计算价格与均线的偏离程度(百分比)
        deviation = (close - ma) / ma

        # 生成信号
        signal = pd.Series(0, index=data.index)
        signal[deviation < -self.params.threshold] = 1   # 价格低于均线超过阈值，买入
        signal[deviation > self.params.threshold] = -1   # 价格高于均线超过阈值，卖出

        return signal

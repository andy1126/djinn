"""
布林带策略。

基于布林带指标生成交易信号：
- 价格触及下轨: 超卖信号，买入 (1)
- 价格触及上轨: 超买信号，卖出 (-1)
- 介于上下轨之间: 无明确信号 (0)

布林带由中轨(移动平均线)和上下轨(标准差)组成。
"""

import numpy as np
import pandas as pd

from djinn import SimpleStrategy, param


class BollingerBandsStrategy(SimpleStrategy):
    """
    布林带策略。

    布林带(Bollinger Bands)由中轨(移动平均线)和上下轨(标准差倍数)组成，
    用于识别价格波动范围和潜在的超买超卖状态。

    参数:
        period: 移动平均周期，默认20
        std_dev: 标准差倍数，默认2.0

    使用方法:
        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
    """

    name = "BollingerBands"

    # 策略参数
    period = param(20, min=5, max=100, description="移动平均周期")
    std_dev = param(2.0, min=0.5, max=5.0, description="标准差倍数")

    def signals(self, data):
        """
        基于布林带生成交易信号。

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            pd.Series: 信号值 (1=买入, -1=卖出, 0=无信号)
        """
        close = data['close']

        # 计算中轨(简单移动平均)
        middle_band = close.rolling(window=self.params.period).mean()

        # 计算标准差
        std = close.rolling(window=self.params.period).std()

        # 计算上下轨
        upper_band = middle_band + (std * self.params.std_dev)
        lower_band = middle_band - (std * self.params.std_dev)

        # 生成信号
        signal = pd.Series(0, index=data.index)
        signal[close < lower_band] = 1   # 价格低于下轨，买入
        signal[close > upper_band] = -1  # 价格高于上轨，卖出

        return signal

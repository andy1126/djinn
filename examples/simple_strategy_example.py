"""
使用 SimpleStrategy 实现移动平均线交叉策略的示例。

这演示了以前需要约500行代码的策略现在可以用约15行代码编写。
"""

import numpy as np
import pandas as pd

from djinn import SimpleStrategy, param


class MACrossover(SimpleStrategy):
    """移动平均线交叉策略。

    当快速MA上穿慢速MA时生成买入信号，
    当快速MA下穿慢速MA时生成卖出信号。
    """
    fast = param(10, min=2, max=100, description="快速MA周期")
    slow = param(30, min=5, max=200, description="慢速MA周期")

    def signals(self, data):
        """基于MA交叉生成交易信号。

        参数:
            data: 包含 'close' 列的 DataFrame

        返回:
            pd.Series: 1表示买入，-1表示卖出
        """
        fast_ma = data['close'].rolling(self.params.fast).mean()
        slow_ma = data['close'].rolling(self.params.slow).mean()

        return np.where(fast_ma > slow_ma, 1, -1)


def main():
    """运行示例回测。"""
    # 创建示例数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    data = pd.DataFrame({'close': prices}, index=dates)

    # 使用自定义参数创建策略
    strategy = MACrossover(fast=5, slow=20)

    # 生成信号
    signals = strategy.signals(data)

    print(f"策略: {strategy.__class__.__name__}")
    print(f"参数: fast={strategy.params.fast}, slow={strategy.params.slow}")
    print(f"生成信号数: {len(signals)}")
    print(f"买入信号: {(signals == 1).sum()}")
    print(f"卖出信号: {(signals == -1).sum()}")


if __name__ == '__main__':
    main()

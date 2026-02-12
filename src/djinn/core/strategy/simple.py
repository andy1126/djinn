from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd

from .parameter import Parameter


class SimpleStrategy:
    """简化的策略基类。

    子类使用 param() 将参数声明为类属性，
    并实现 signals() 方法。

    示例:
        class MACrossover(SimpleStrategy):
            fast = param(10, min=2, max=100)
            slow = param(30, min=5, max=200)

            def signals(self, data):
                fast_ma = data.close.rolling(self.params.fast).mean()
                slow_ma = data.close.rolling(self.params.slow).mean()
                return np.where(fast_ma > slow_ma, 1, -1)
    """

    # 默认策略名称，子类可以覆盖
    name = "SimpleStrategy"

    # 默认最小数据点要求
    min_data_points = 20

    def __init__(self, **kwargs):
        """使用参数初始化策略。

        参数:
            **kwargs: 用于覆盖默认值的参数值
        """
        # 收集声明的参数
        self._param_definitions = self._collect_param_definitions()

        # 使用默认值构建参数命名空间
        params_dict = {
            name: param.default
            for name, param in self._param_definitions.items()
        }

        # 使用提供的值覆盖
        for key, value in kwargs.items():
            if key in params_dict:
                params_dict[key] = value
            else:
                raise ValueError(f"未知参数: {key}")

        # 验证所有参数
        self.params = SimpleNamespace(**self._validate_params(params_dict))

    def _collect_param_definitions(self):
        """从类属性收集 Parameter 声明。"""
        definitions = {}
        for name in dir(self.__class__):
            if name.startswith('_'):
                continue
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, Parameter):
                definitions[name] = attr
        return definitions

    def _validate_params(self, params_dict):
        """根据定义验证所有参数。"""
        validated = {}
        for name, value in params_dict.items():
            param_def = self._param_definitions[name]
            validated[name] = param_def.validate(value)
        return validated

    def initialize(self, market_data):
        """初始化策略。

        此方法由回测引擎在回测开始前调用。
        子类可选择性覆盖此方法以执行自定义初始化。

        参数:
            market_data: MarketData 对象，包含初始市场数据
        """
        pass

    def signals(self, data):
        """生成交易信号。

        子类必须实现此方法。

        参数:
            data: 包含 OHLCV 数据的 DataFrame

        返回:
            pd.Series，信号值:
                1 = 买入/持有多头
                -1 = 卖出/持有空头（或退出）
                0 = 无信号
        """
        raise NotImplementedError(
            "子类必须实现 signals() 方法"
        )

    # 回测引擎兼容的适配器方法

    def calculate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """向量化回测引擎的适配器。

        委托给 signals() 方法。

        参数:
            data: 包含 OHLCV 数据的 DataFrame

        返回:
            包含信号值的 pd.Series
        """
        result = self.signals(data)
        if isinstance(result, pd.Series):
            return result
        return pd.Series(result, index=data.index)

    def calculate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_date: datetime
    ) -> float:
        """事件驱动回测引擎的适配器。

        参数:
            symbol: 计算信号的品种代码
            data: 到 current_date 为止的历史数据
            current_date: 信号计算的当前日期

        返回:
            float: 当前日期的信号值
        """
        signals = self.signals(data)

        if isinstance(signals, pd.Series):
            if current_date in signals.index:
                return float(signals.loc[current_date])
            # 查找最后一个可用信号
            mask = signals.index <= current_date
            if mask.any():
                return float(signals[mask].iloc[-1])
        else:
            # numpy 数组 - 返回最后一个值
            if len(signals) > 0:
                return float(signals[-1])

        return 0.0

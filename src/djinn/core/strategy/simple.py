from types import SimpleNamespace
from typing import Any

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

"""通用信号策略:按 ``indicator`` 名调用注册的信号指标,免去每指标写一个策略类。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from djinn.strategy.base import Strategy, param
from djinn.strategy.signals import get_signal_indicator
from djinn.strategy.utils import state_from_signals


class SignalStrategy(Strategy):
    """通用信号策略(仅做多)。

    构造/配置:``strategy.name: SignalStrategy`` + ``strategy.params.indicator``
    指定信号指标名(见 :data:`djinn.strategy.signals.SIGNAL_INDICATORS`),其余
    参数透传给该信号函数。

    示例(YAML):

    .. code-block:: yaml

        strategy:
          name: SignalStrategy
          params:
            indicator: adaptive_trend_trail
            trend_length: 40
            sensitivity: 0.5
    """

    indicator = param(
        "",
        description="信号指标注册名(如 supertrend / adaptive_trend_trail / ma_cross)",
    )

    def __init__(self, indicator: str, **params: Any) -> None:
        super().__init__(indicator=indicator)
        self._signal_fn = get_signal_indicator(indicator)
        self._indicator_params = dict(params)

    @property
    def params(self) -> dict[str, Any]:
        return {"indicator": self.indicator, **self._indicator_params}

    def signals(self, data: pd.DataFrame) -> pd.Series:
        return state_from_signals(self._signal_fn(data, **self._indicator_params))

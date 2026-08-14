"""因子抽象基类。

因子是把"原始数据 → 截面可比较的数值"的向量化映射:给定宽表面板
(``date × symbol``),输出同形状的因子值面板。与策略不同,因子**纯向量化、
无状态、逐日截面**,天然满足防未来函数(``date t`` 仅用 ``≤ t`` 数据)。

复用 :mod:`djinn.strategy.parameter` 的 ``param()`` 声明式参数(与策略同一套
schema,前端可复用动态表单)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from djinn.strategy.parameter import (
    _PARAM_ATTR,
    collect_params,
    get_params,
    param,
)
from djinn.utils.exceptions import FactorError

# 因子输入面板的别名(均为 index=date、columns=symbol 的宽表)。
Panel = pd.DataFrame
PanelDict = dict[str, pd.DataFrame]


class Factor(ABC):
    """因子抽象基类。

    子类用 ``param()`` 声明参数,覆写 :meth:`compute`。

    Attributes:
        name: 因子名(默认类名转 snake)。
        category: 类别(momentum / value / quality / ...),供分组与归因。
    """

    name: str = ""
    category: str = "generic"

    # 声明式输入依赖:因子 compute() 所需的基本面 / 行情字段。默认空(纯价格因子)。
    # 声明后,FactorEngine / API 任务在计算前校验字段存在且非全 NaN,
    # 缺失即 fail-fast(取代静默产出 NaN)——见 :meth:`validate_inputs`。
    required_fundamentals: tuple[str, ...] = ()
    required_ohlcv: tuple[str, ...] = ()

    # D3:因子滚动计算所需的最大回看窗口(交易日)。FactorPortfolioStrategy 据此
    # 截断调仓日面板,避免对全历史重算。默认 252 覆盖所有内置价格类因子。
    max_lookback: int = 252

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        setattr(cls, _PARAM_ATTR, collect_params(cls))
        if not cls.name:
            cls.name = cls.__name__.lower()

    def __init__(self, **params: Any) -> None:
        if type(self) is Factor:
            raise TypeError("Factor 是抽象基类,不能直接实例化")
        declared = get_params(type(self))
        for k, p in declared.items():
            setattr(self, k, p.default)
        for k, v in params.items():
            if k not in declared:
                raise ValueError(f"因子 {type(self).__name__} 无参数 {k!r}")
            setattr(self, k, v)

    def validate_inputs(self, fundamentals: PanelDict, ohlcv: PanelDict) -> None:
        """校验 ``compute()`` 所需的输入字段存在且非全 NaN,缺失即抛 FactorError。

        由 :class:`~djinn.factor.engine.FactorEngine.compute` 与 API 后台任务在
        计算前调用;纯价格因子(默认空依赖)直接通过。
        """
        missing_f = [
            f
            for f in self.required_fundamentals
            if f not in fundamentals or fundamentals[f].isna().all().all()
        ]
        missing_o = [
            f
            for f in self.required_ohlcv
            if f not in ohlcv or ohlcv[f].isna().all().all()
        ]
        if missing_f or missing_o:
            parts: list[str] = []
            if missing_f:
                parts.append(f"基本面字段缺失或全空: {missing_f}")
            if missing_o:
                parts.append(f"行情字段缺失或全空: {missing_o}")
            raise FactorError(
                f"因子 {self.name} 所需输入不可用({'、'.join(parts)});"
                "请检查 DEFAULT_FUNDAMENTAL_FIELDS / 行情面板或 provider 支持"
            )

    @abstractmethod
    def compute(
        self,
        prices: Panel,
        ohlcv: PanelDict,
        fundamentals: PanelDict,
    ) -> Panel:
        """计算因子值面板(index=date、columns=symbol,与 ``prices`` 对齐)。

        Args:
            prices: 收盘价宽表(date × symbol)。
            ohlcv: 行情字段宽表字典(``open/high/low/volume/amount`` 等)。
            fundamentals: 基本面字段宽表字典(``pe/pb/roe/market_cap`` 等,
                已按 point-in-time 对齐到交易日)。

        Returns:
            因子值宽表;无足够数据的早期位置为 NaN。
        """

    @property
    def params(self) -> dict[str, Any]:
        declared = get_params(type(self))
        return {k: getattr(self, k) for k in declared}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"


__all__ = ["Factor", "Panel", "PanelDict", "param"]

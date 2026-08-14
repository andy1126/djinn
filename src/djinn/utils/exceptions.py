"""djinn 异常层级。

所有自定义异常继承自 :class:`DjinnError`,便于上层统一捕获与日志记录。
"""

from __future__ import annotations


class DjinnError(Exception):
    """所有 djinn 异常的基类。"""


class ConfigError(DjinnError):
    """配置加载 / 校验失败。"""


class DataError(DjinnError):
    """数据拉取 / 缓存 / 规范化失败。"""


class ProviderError(DataError):
    """数据提供器访问失败(网络 / 限流 / 鉴权)。"""


class SymbolNotFoundError(DataError):
    """标的代码无法被任何 provider 识别。"""


class FactorError(DjinnError):
    """因子计算失败(所需输入字段缺失或全空 / 参数非法)。"""


class StrategyError(DjinnError):
    """策略参数校验 / 执行失败。"""


class ParameterError(StrategyError):
    """声明式参数取值越界或类型不符。"""


class BrokerError(DjinnError):
    """撮合 / 成交过程中发生的逻辑错误(非可预期的拒单)。"""


class AccountError(DjinnError):
    """账户会计不一致(现金为负、股数为负、可用股数不足等)。"""


class OrderRejectedError(DjinnError):
    """订单被交易约束拒绝(停牌 / 涨跌停 / 资金不足 / 最小手 / T+1)。

    注意:这是一类**预期内**的回测事件,通常应被引擎捕获并记录为
    ``Rejection``,而非向上抛出。仅在调用方显式希望"失败即抛"时使用。
    """


class BacktestCancelled(DjinnError):
    """回测在协作式中断点检测到取消请求,提前终止。"""

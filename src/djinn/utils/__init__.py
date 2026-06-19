"""djinn.utils — 基础设施:异常、Decimal 辅助、日期、日志。"""

from __future__ import annotations

from djinn.utils.dates import (
    TRADING_DAYS_PER_YEAR,
    parse_date,
    to_timestamp,
    trading_days_per_year,
)
from djinn.utils.decimalmath import (
    D,
    floor_shares,
    is_zero,
    q_money,
    q_ratio,
    q_shares,
    to_float,
)
from djinn.utils.exceptions import (
    AccountError,
    BrokerError,
    ConfigError,
    DataError,
    DjinnError,
    OrderRejectedError,
    ParameterError,
    ProviderError,
    StrategyError,
    SymbolNotFoundError,
)
from djinn.utils.logging import get_logger, set_log_level

__all__ = [
    # 异常
    "DjinnError",
    "ConfigError",
    "DataError",
    "ProviderError",
    "SymbolNotFoundError",
    "StrategyError",
    "ParameterError",
    "BrokerError",
    "AccountError",
    "OrderRejectedError",
    # Decimal
    "D",
    "q_money",
    "q_shares",
    "q_ratio",
    "floor_shares",
    "to_float",
    "is_zero",
    # 日期
    "parse_date",
    "to_timestamp",
    "trading_days_per_year",
    "TRADING_DAYS_PER_YEAR",
    # 日志
    "get_logger",
    "set_log_level",
]

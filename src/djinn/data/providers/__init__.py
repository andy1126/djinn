"""数据提供器:Yahoo(美股)、AkShare(A/港股)、Tushare(A 股)、CSV(本地)。"""

from __future__ import annotations

from djinn.data.provider import DataProvider, ProviderRegistry
from djinn.data.providers.akshare import AkShareProvider
from djinn.data.providers.csv import CSVProvider
from djinn.data.providers.tushare import TushareProvider
from djinn.data.providers.yahoo import YahooProvider

__all__ = [
    "AkShareProvider",
    "CSVProvider",
    "DataProvider",
    "ProviderRegistry",
    "TushareProvider",
    "YahooProvider",
]

"""
Data providers for Djinn quantitative backtesting framework.

This module provides concrete implementations of DataProvider for
various market data sources.
"""

from ..base import DataProvider
from .yahoo_finance import YahooFinanceProvider, create_yahoo_finance_provider
from .akshare_provider import AKShareProvider, create_akshare_provider

__all__ = [
    # Base class
    "DataProvider",

    # Yahoo Finance provider
    "YahooFinanceProvider",
    "create_yahoo_finance_provider",

    # AKShare provider (Chinese/HK markets)
    "AKShareProvider",
    "create_akshare_provider",
]
"""股票池(universe)工具:全市场股票列表、指数成分、行业分类。

实现均委托给 provider(见 :mod:`djinn.data.provider` 的非抽象方法),
本模块负责:
- ``UNIVERSE_INDEX_MAP``:常用宽基指数 → 各市场 / 数据源代码的映射;
- 规范化(统一 symbol 代码、补 market 列)、缓存键约定。

行业 / 成分属低频数据,provider 内部按缓存键 ``universe`` 低频缓存(可按月)。
"""

from __future__ import annotations

from typing import Final

from djinn.data.schema import Market

# 指数成分缓存的失效时间(天)。
# 指数成分属低频数据:CSI 半年调整,SP500/NASDAQ100/HSI 季度调整,DOW 不定期。
# 统一 30 天(月级),保证在任一调整周期内自然刷新且远短于最短调整周期。
INDEX_COMPONENTS_TTL_DAYS: Final[int] = 30

# 宽基指数 → 数据源映射。
# - ``market``:指数所属市场;
# - ``akshare``:akshare ``index_stock_cons`` 用的纯代码(A 股指数);
# - ``yahoo``:yfinance 指数代码(港股 / 美股,仅用于行情,成分另见 provider)。
UNIVERSE_INDEX_MAP: Final[dict[str, dict[str, object]]] = {
    "CSI300": {"market": Market.CN, "akshare": "000300", "name": "沪深300"},
    "CSI500": {"market": Market.CN, "akshare": "000905", "name": "中证500"},
    "CSI800": {"market": Market.CN, "akshare": "000906", "name": "中证800"},
    "SSE50": {"market": Market.CN, "akshare": "000016", "name": "上证50"},
    "STAR50": {"market": Market.CN, "akshare": "000688", "name": "科创50"},
    "CHINEXT": {"market": Market.CN, "akshare": "399006", "name": "创业板指"},
    "CSI1000": {"market": Market.CN, "akshare": "000852", "name": "中证1000"},
    "HSI": {"market": Market.HK, "yahoo": "^HSI", "name": "恒生指数"},
    # 恒生科技无 yfiua 免费 CSV,用追踪 ETF(iShares 3067.HK)持仓代理
    "HSTECH": {
        "market": Market.HK,
        "yahoo": "^HSTECH",
        "name": "恒生科技",
        "etf": "3067.HK",
    },
    "SP500": {"market": Market.US, "yahoo": "^GSPC", "name": "标普500"},
    "NASDAQ100": {"market": Market.US, "yahoo": "^NDX", "name": "纳斯达克100"},
    "DOWJONES": {"market": Market.US, "yahoo": "^DJI", "name": "道琼斯工业指数"},
}


def normalize_cn_symbol(code: str) -> str:
    """akshare 纯 6 位代码 → djinn 标准 ``.SH/.SZ/.BJ`` 后缀形式。"""
    c = code.strip()
    if c.endswith((".SH", ".SZ", ".BJ")):
        return c
    if c.startswith(("60", "68", "9", "11", "13")):
        return f"{c}.SH"
    if c.startswith(("43", "83", "87", "88")):
        return f"{c}.BJ"
    return f"{c}.SZ"

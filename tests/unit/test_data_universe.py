"""股票池(universe)与缓存命中测试。

本地 mock / 缓存逻辑不依赖网络;真实 akshare 成分 / 行业拉取标 ``network``。
"""

from __future__ import annotations

import pandas as pd
import pytest

from djinn.data.cache import DataCache
from djinn.data.schema import Adjust, Market
from djinn.data.universe import UNIVERSE_INDEX_MAP, normalize_cn_symbol


def test_normalize_cn_symbol() -> None:
    assert normalize_cn_symbol("600519") == "600519.SH"
    assert normalize_cn_symbol("000001") == "000001.SZ"
    assert normalize_cn_symbol("430047") == "430047.BJ"
    assert normalize_cn_symbol("600519.SH") == "600519.SH"  # 幂等


def test_universe_index_map_cn() -> None:
    assert UNIVERSE_INDEX_MAP["CSI300"]["market"] is Market.CN
    assert UNIVERSE_INDEX_MAP["CSI300"]["akshare"] == "000300"


def test_cache_dtype_keys_distinct() -> None:
    """dtype 维度让 quote / fundamental / universe 键互不冲突。"""
    k_quote = DataCache.make_key("akshare", "X", Adjust.BACKWARD, dtype="quote")
    k_fund = DataCache.make_key("akshare", "X", dtype="fundamental")
    k_univ = DataCache.make_key("akshare", "X", dtype="universe")
    assert len({k_quote, k_fund, k_univ}) == 3


def test_cache_fundamentals_roundtrip(tmp_path) -> None:
    """基本面 / 股票池整帧读写,二次读取命中内存 LRU(不重读磁盘)。"""
    cache = DataCache(cache_dir=tmp_path)
    df = pd.DataFrame({"pe": [20.0]}, index=["000001.SZ"])
    cache.put_fundamentals("mock", "snap", df)
    hit = cache.get_fundamentals("mock", "snap")
    assert hit is not None and len(hit) == 1
    # 索引未被强制转 datetime(symbol 索引原样保留)
    assert hit.index[0] == "000001.SZ"


def test_cache_universe_roundtrip(tmp_path) -> None:
    cache = DataCache(cache_dir=tmp_path)
    df = pd.DataFrame({"symbol": ["000001.SZ", "600519.SH"]})
    cache.put_universe("mock", "cons", df)
    hit = cache.get_universe("mock", "cons")
    assert hit is not None and list(hit["symbol"]) == ["000001.SZ", "600519.SH"]


def test_cache_second_read_uses_memory(tmp_path, monkeypatch) -> None:
    """二次读取命中内存 LRU,不再调用 pd.read_parquet。"""
    cache = DataCache(cache_dir=tmp_path)
    df = pd.DataFrame({"symbol": ["000001.SZ"]})
    cache.put_universe("mock", "cons", df)
    calls = {"n": 0}
    real_read = pd.read_parquet

    def _counting(*a, **k):
        calls["n"] += 1
        return real_read(*a, **k)

    monkeypatch.setattr(pd, "read_parquet", _counting)
    cache.get_universe("mock", "cons")  # 内存已命中
    cache.get_universe("mock", "cons")
    assert calls["n"] == 0


# ── 真实 akshare(需网络)─────────────────────────────────


@pytest.mark.network
def test_akshare_index_components_csi300() -> None:
    pytest.importorskip("akshare")
    from djinn.data.providers.akshare import AkShareProvider

    p = AkShareProvider()
    if not p.supports("000300.SH", Market.CN):
        pytest.skip("akshare 不可用")
    cons = p.get_index_components("CSI300")
    assert len(cons) >= 280
    assert all(s.endswith((".SH", ".SZ", ".BJ")) for s in cons)


@pytest.mark.network
def test_akshare_stock_list_real() -> None:
    pytest.importorskip("akshare")
    from djinn.data.providers.akshare import AkShareProvider

    p = AkShareProvider()
    df = p.get_stock_list(Market.CN)
    assert len(df) > 3000  # 全 A 股数千只
    assert {"name", "market"} <= set(df.columns)

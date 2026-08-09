"""Yahoo 指数成分(HSI / SP500)单元测试。

mock HTTP(``urllib.request.urlopen``)避免触网;真实网络拉取标 ``network``
(见 test_data_universe.py 约定)。
"""

from __future__ import annotations

import io
import os
import time

import pandas as pd
import pytest

from djinn.data.cache import DataCache
from djinn.data.providers.yahoo import YahooProvider


def _csv_bytes(symbols: list[str]) -> bytes:
    """构造 ``Symbol,Name`` 的 CSV 字节(与 yfiua.github.io 格式一致)。"""
    body = "\n".join(f"{s},Name{i}" for i, s in enumerate(symbols))
    return ("Symbol,Name\n" + body).encode()


def _monkey_urlopen(monkeypatch, handler) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        handler,
    )


def test_yahoo_index_components_hsi(monkeypatch, tmp_path) -> None:
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))
    captured: dict[str, object] = {}

    def fake_urlopen(url: str, timeout: int) -> io.IOBase:
        captured["url"] = url
        captured["timeout"] = timeout
        return io.BytesIO(_csv_bytes(["0101.HK", "1024.HK", "1038.HK"]))

    _monkey_urlopen(monkeypatch, fake_urlopen)
    cons = p.get_index_components("HSI")
    assert cons == ["0101.HK", "1024.HK", "1038.HK"]
    assert str(captured["url"]).endswith("constituents-hsi.csv")
    assert captured["timeout"] == 20


def test_yahoo_index_components_sp500_dot_passthrough(monkeypatch, tmp_path) -> None:
    """US 带点符号(``BRK.B`` / ``BF.B``)在成分列表原样保留(转换在行情层)。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))

    def fake_urlopen(url: str, timeout: int) -> io.IOBase:
        return io.BytesIO(_csv_bytes(["NVDA", "BRK.B", "BF.B", "AAPL"]))

    _monkey_urlopen(monkeypatch, fake_urlopen)
    cons = p.get_index_components("SP500")
    assert "BRK.B" in cons
    assert "BF.B" in cons
    assert "NVDA" in cons


def test_yahoo_index_components_nasdaq100(monkeypatch, tmp_path) -> None:
    """NASDAQ100 走 nasdaq100 CSV(URL 由 index.lower() 派生)。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))
    captured: dict[str, object] = {}

    def fake_urlopen(url: str, timeout: int) -> io.IOBase:
        captured["url"] = url
        return io.BytesIO(_csv_bytes(["NVDA", "AAPL", "MSFT", "GOOGL"]))

    _monkey_urlopen(monkeypatch, fake_urlopen)
    cons = p.get_index_components("NASDAQ100")
    assert cons == ["NVDA", "AAPL", "MSFT", "GOOGL"]
    assert str(captured["url"]).endswith("constituents-nasdaq100.csv")


def test_yahoo_index_components_uses_cache(monkeypatch, tmp_path) -> None:
    """二次调用命中缓存,不再触网。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))
    calls = {"n": 0}

    def fake_urlopen(url: str, timeout: int) -> io.IOBase:
        calls["n"] += 1
        return io.BytesIO(_csv_bytes(["NVDA", "AAPL"]))

    _monkey_urlopen(monkeypatch, fake_urlopen)
    assert p.get_index_components("SP500") == ["NVDA", "AAPL"]
    assert p.get_index_components("SP500") == ["NVDA", "AAPL"]
    assert calls["n"] == 1


def test_yahoo_index_components_dedup_keep_order(monkeypatch, tmp_path) -> None:
    """重复符号去重且保持首次出现顺序。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))

    def fake_urlopen(url: str, timeout: int) -> io.IOBase:
        return io.BytesIO(_csv_bytes(["AAPL", "NVDA", "AAPL"]))

    _monkey_urlopen(monkeypatch, fake_urlopen)
    assert p.get_index_components("SP500") == ["AAPL", "NVDA"]


def test_yahoo_index_components_unknown_index_not_implemented(tmp_path) -> None:
    """A 股宽基 / 未知指数交给更前序 provider(akshare)。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))
    with pytest.raises(NotImplementedError):
        p.get_index_components("CSI300")
    with pytest.raises(NotImplementedError):
        p.get_index_components("UNKNOWN")


def test_yahoo_index_components_network_error(monkeypatch, tmp_path) -> None:
    """网络失败抛 ProviderError(继承自 DataError),不误报 NotImplementedError。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))

    def boom(url: str, timeout: int) -> io.IOBase:
        raise OSError("network down")

    _monkey_urlopen(monkeypatch, boom)
    with pytest.raises(Exception) as ei:
        p.get_index_components("HSI")
    assert "yahoo" in str(ei.value)


def test_yf_symbol_normalization() -> None:
    """行情层符号转换:美股带点 → 连字符;``.HK`` 后缀不改写。"""
    p = YahooProvider()
    assert p._yf_symbol("BRK.B") == "BRK-B"
    assert p._yf_symbol("BF.B") == "BF-B"
    assert p._yf_symbol("NVDA") == "NVDA"
    assert p._yf_symbol("0101.HK") == "0101.HK"
    assert p._yf_symbol("600519.SH") == "600519.SH"


def test_yahoo_index_components_dowjones(monkeypatch, tmp_path) -> None:
    """DOWJONES 走 dowjones CSV(URL 由 index.lower() 派生)。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))
    captured: dict[str, object] = {}

    def fake_urlopen(url: str, timeout: int) -> io.IOBase:
        captured["url"] = url
        return io.BytesIO(_csv_bytes(["GS", "CAT", "MSFT"]))

    _monkey_urlopen(monkeypatch, fake_urlopen)
    cons = p.get_index_components("DOWJONES")
    assert cons == ["GS", "CAT", "MSFT"]
    assert str(captured["url"]).endswith("constituents-dowjones.csv")


def test_yahoo_index_component_names(monkeypatch, tmp_path) -> None:
    """名称映射与符号同源:缓存 symbol+name,名称取自 CSV Name 列。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))

    def fake_urlopen(url: str, timeout: int) -> io.IOBase:
        return io.BytesIO(_csv_bytes(["NVDA", "BRK.B", "BF.B", "AAPL"]))

    _monkey_urlopen(monkeypatch, fake_urlopen)
    cons = p.get_index_components("SP500")
    assert cons == ["NVDA", "BRK.B", "BF.B", "AAPL"]
    names = p.get_index_component_names("SP500")
    assert names == {
        "NVDA": "Name0",
        "BRK.B": "Name1",
        "BF.B": "Name2",
        "AAPL": "Name3",
    }


def test_yahoo_index_component_names_not_implemented(tmp_path) -> None:
    """非 yahoo 指数抛 NotImplementedError(交给 akshare)。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))
    with pytest.raises(NotImplementedError):
        p.get_index_component_names("CSI300")
    with pytest.raises(NotImplementedError):
        p.get_index_component_names("UNKNOWN")


def test_yahoo_index_component_names_old_cache_self_heal(monkeypatch, tmp_path) -> None:
    """旧格式缓存(只有 symbol 列)自愈:重拉一次并重写为 symbol+name。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))
    # 预写旧格式缓存(只有 symbol 列)
    p.cache.put_universe(
        "yahoo", "index_cons_sp500", pd.DataFrame({"symbol": ["NVDA", "AAPL"]})
    )
    calls = {"n": 0}

    def fake_urlopen(url: str, timeout: int) -> io.IOBase:
        calls["n"] += 1
        return io.BytesIO(_csv_bytes(["NVDA", "AAPL"]))

    _monkey_urlopen(monkeypatch, fake_urlopen)
    # 旧格式被当作 miss → 重拉一次(不再是缓存命中)
    cons = p.get_index_components("SP500")
    assert cons == ["NVDA", "AAPL"]
    assert calls["n"] == 1
    # 名称方法返回新格式的名称
    names = p.get_index_component_names("SP500")
    assert names == {"NVDA": "Name0", "AAPL": "Name1"}
    # 二次调用不再触网
    p.get_index_components("SP500")
    assert calls["n"] == 1


def test_yahoo_index_components_ttl_refetch(monkeypatch, tmp_path) -> None:
    """缓存超龄(>30 天)时重拉,不再命中旧缓存。"""
    p = YahooProvider(cache=DataCache(cache_dir=tmp_path))
    # 先写入一份新格式缓存
    p.cache.put_universe(
        "yahoo",
        "index_cons_sp500",
        pd.DataFrame({"symbol": ["NVDA"], "name": ["Nvidia"]}),
    )
    # 把磁盘 mtime 改到 40 天前 → 超龄
    key = DataCache.make_key("yahoo", "index_cons_sp500", dtype="universe")
    path = p.cache._parquet_path(key)
    old = time.time() - 40 * 86400
    os.utime(path, (old, old))
    calls = {"n": 0}

    def fake_urlopen(url: str, timeout: int) -> io.IOBase:
        calls["n"] += 1
        return io.BytesIO(_csv_bytes(["AAPL", "NVDA"]))

    _monkey_urlopen(monkeypatch, fake_urlopen)
    # 超龄 → 重拉一次,拿到新成分
    cons = p.get_index_components("SP500")
    assert cons == ["AAPL", "NVDA"]
    assert calls["n"] == 1
    # 重拉后 mtime 更新,二次命中缓存不触网
    p.get_index_components("SP500")
    assert calls["n"] == 1


# ── 真实网络拉取(需网络,标 network)───────────────────────


@pytest.mark.network
def test_yahoo_index_components_hsi_real() -> None:
    pytest.importorskip("yfinance")
    p = YahooProvider()
    cons = p.get_index_components("HSI")
    assert len(cons) >= 80
    assert all(s.endswith(".HK") for s in cons)


@pytest.mark.network
def test_yahoo_index_components_sp500_real() -> None:
    pytest.importorskip("yfinance")
    p = YahooProvider()
    cons = p.get_index_components("SP500")
    assert len(cons) >= 490

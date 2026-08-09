"""Yahoo 指数成分(HSI / SP500)单元测试。

mock HTTP(``urllib.request.urlopen``)避免触网;真实网络拉取标 ``network``
(见 test_data_universe.py 约定)。
"""

from __future__ import annotations

import io

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

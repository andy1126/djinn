"""缓存内容查看端点测试:/data/cache/content(结构 + 首尾预览)。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from djinn.api.deps import get_cache
from djinn.api.main import app
from djinn.data.cache import DataCache
from djinn.data.schema import Adjust

_tmp = Path(tempfile.mkdtemp())
_cache = DataCache(cache_dir=_tmp)
app.dependency_overrides[get_cache] = lambda: _cache

client = TestClient(app)


def setup_module() -> None:
    # 跨文件跑时其它套件的 teardown 会清 override,这里重新注入
    app.dependency_overrides[get_cache] = lambda: _cache
    df = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    _cache.put("test", "TEST", Adjust.BACKWARD, df)


def teardown_module() -> None:
    app.dependency_overrides.clear()


def test_cache_content():
    file = "test::quote::TEST::backward.parquet"
    resp = client.get("/data/cache/content", params={"file": file})
    assert resp.status_code == 200
    d = resp.json()
    assert d["rows"] == 3
    assert d["index_type"] == "datetime"
    assert [c["name"] for c in d["columns"]] == ["close", "name"]
    assert d["head"][0]["_index"] == "2024-01-01"
    assert d["head"][0]["close"] == 1.0
    assert d["tail"][-1]["name"] == "c"


def test_cache_content_missing_404():
    resp = client.get("/data/cache/content", params={"file": "no-such.parquet"})
    assert resp.status_code == 404


def test_cache_content_path_traversal_rejected():
    resp = client.get("/data/cache/content", params={"file": "../../etc/passwd"})
    assert resp.status_code == 404

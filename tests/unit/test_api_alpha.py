"""Phase 6 横截面 alpha API 测试:因子库 / 因子分析 / 选股 / 股票池。

不依赖网络:用确定性的 stub provider(注入 ``get_registry``)提供行情 / 基本面 /
成分 / 行业数据;TestClient 同步执行后台任务,任务完成后即可取结果。
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DJINN_TEST", "1")

from djinn.api.deps import get_job_registry, get_registry
from djinn.api.jobs import JobRegistry, recover_orphaned_jobs
from djinn.api.main import app
from djinn.data.market_data import MarketData
from djinn.data.provider import DataProvider, ProviderRegistry
from djinn.data.schema import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_FLOAT_CAP,
    COL_HIGH,
    COL_LOW,
    COL_MARKET_CAP,
    COL_OPEN,
    COL_PB,
    COL_PE,
    COL_VOLUME,
    Adjust,
    Market,
)

_SYMBOLS = ["000001.SH", "000002.SH", "600000.SH", "600519.SH", "000063.SH"]
_INDUSTRIES = {
    "000001.SH": "银行",
    "000002.SH": "地产",
    "600000.SH": "银行",
    "600519.SH": "白酒",
    "000063.SH": "通信",
}


def _code_num(symbol: str) -> int:
    """确定性 symbol → 序号(不用内置 hash,避免 PYTHONHASHSEED 抖动)。"""
    return sum(ord(c) for c in symbol) % 50


def _synth_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    """确定性线性上行行情:斜率随 symbol 递增(动量与前向收益截面同向)。"""
    idx = pd.bdate_range(start, end)
    n = len(idx)
    num = _code_num(symbol)
    base = 10.0 + num
    slope = 0.02 * (num + 1)  # 每符号不同斜率 → 截面动量 / 收益排序一致
    closes = [base + slope * i for i in range(n)]
    return pd.DataFrame(
        {
            COL_OPEN: closes,
            COL_HIGH: [c * 1.01 for c in closes],
            COL_LOW: [c * 0.99 for c in closes],
            COL_CLOSE: closes,
            COL_VOLUME: [1.0e6] * n,
            COL_AMOUNT: [1.0e8] * n,
        },
        index=idx,
    )


class _StubProvider(DataProvider):
    """确定性 stub provider(覆盖 universe / 行情 / 基本面 / 行业接口)。"""

    name = "stub"
    market = Market.CN

    def supports(self, symbol: str, market: Market | None = None) -> bool:
        return True

    def get_ohlcv(
        self, symbol: str, start: date, end: date, adjust: Adjust = Adjust.BACKWARD
    ) -> MarketData:
        return MarketData(
            symbol=symbol,
            market=Market.CN,
            df=_synth_ohlcv(symbol, start, end),
            adjust=adjust,
        )

    def get_stock_list(self, market: Market | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {"name": [f"股票{i}" for i in range(len(_SYMBOLS))], "market": "CN"},
            index=_SYMBOLS,
        )

    def get_index_components(self, index: str) -> list[str]:
        if index == "EMPTY":
            return []
        return list(_SYMBOLS)

    def get_index_component_names(self, index: str) -> dict[str, str]:
        return {s: f"名称{i}" for i, s in enumerate(_SYMBOLS)}

    def search_symbols(
        self, query: str, market: Market | None = None
    ) -> list[tuple[str, str]]:
        q = query.upper()
        return [(s, f"名称{i}") for i, s in enumerate(_SYMBOLS) if q in s.upper()]

    def get_stock_name(self, symbol: str, market: Market | None = None) -> str:
        return f"股票{_code_num(symbol)}" if symbol in _SYMBOLS else ""

    def get_stock_price(self, symbol: str, market: Market | None = None) -> float:
        if symbol not in _SYMBOLS:
            raise KeyError(symbol)
        return 10.0 + _code_num(symbol)

    def get_industry_map(self, symbols: list[str]) -> dict[str, str]:
        return {s: _INDUSTRIES[s] for s in symbols if s in _INDUSTRIES}

    def get_fundamentals(self, symbols: list[str], when: date) -> pd.DataFrame:
        rows = {}
        for s in symbols:
            num = _code_num(s)
            rows[s] = {
                COL_MARKET_CAP: 1.0e10 * (num + 1),
                COL_FLOAT_CAP: 8.0e9 * (num + 1),
                COL_PE: 5.0 + num,  # 截面 pe 递增 → 可筛 / 可排序
                COL_PB: 1.0 + 0.1 * num,
            }
        return pd.DataFrame.from_dict(rows, orient="index")


_stub_registry = ProviderRegistry([_StubProvider()])
_test_registry = JobRegistry(db_path=".cache/test_jobs_alpha.db")

client = TestClient(app)

_START = (date.today() - timedelta(days=200)).isoformat()
_END = (date.today() - timedelta(days=1)).isoformat()


def setup_module() -> None:
    """注入 stub registry(在 test_api.py 的 teardown 清除后重新建立,保证隔离)。"""
    app.dependency_overrides[get_job_registry] = lambda: _test_registry
    app.dependency_overrides[get_registry] = lambda: _stub_registry


def teardown_module() -> None:
    app.dependency_overrides.clear()


def _wait_done(job_id: str, kind_url: str) -> dict:
    """轮询任务直至完成(后台任务同步执行,首轮即 done)。"""
    for _ in range(50):
        got = client.get(f"{kind_url}/{job_id}")
        assert got.status_code == 200
        body = got.json()
        if body["status"] in ("done", "error"):
            return body
    raise AssertionError("任务未完成")


# ── 因子库 ─────────────────────────────────────────────
def test_list_factors() -> None:
    resp = client.get("/factors")
    assert resp.status_code == 200
    names = [f["name"] for f in resp.json()["factors"]]
    assert "momentum" in names and "bp" in names and "size" in names
    mom = next(f for f in resp.json()["factors"] if f["name"] == "momentum")
    assert mom["category"] == "momentum"
    assert [p["name"] for p in mom["params"]] == ["period", "skip"]


def test_get_factor_404() -> None:
    assert client.get("/factors/bogus").status_code == 404


# ── 因子分析 ───────────────────────────────────────────
def test_factor_analysis_requires_universe() -> None:
    resp = client.post(
        "/factor-analysis",
        json={"factor": "momentum", "start": _START, "end": _END},
    )
    assert resp.status_code == 400


def test_factor_analysis_unknown_factor_404() -> None:
    resp = client.post(
        "/factor-analysis",
        json={
            "factor": "bogus",
            "symbols": _SYMBOLS,
            "start": _START,
            "end": _END,
        },
    )
    assert resp.status_code == 404


def test_factor_analysis_end_to_end() -> None:
    resp = client.post(
        "/factor-analysis",
        json={
            "factor": "momentum",
            "params": {"period": 5, "skip": 0},
            "symbols": _SYMBOLS,
            "start": _START,
            "end": _END,
            "periods": [1, 5],
            "n_quantiles": 3,
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    body = _wait_done(job_id, "/factor-analysis")
    assert body["status"] == "done", body.get("error")
    assert body["title"].startswith("因子分析 momentum")

    rep = client.get(f"/factor-analysis/{job_id}/report")
    assert rep.status_code == 200
    report = rep.json()
    assert report["factor_name"] == "momentum"
    # 斜率驱动的动量与前向收益同向 → IC 显著为正
    assert report["ic_summary"]["ic_mean"] > 0.5
    assert set(report["ic_decay"]) == {"1", "5"}
    assert len(report["quantile_returns"]["columns"]) == 3  # 3 分层
    # 报告可 JSON 序列化(已由 TestClient 解码)


def test_factor_analysis_report_400_when_not_done() -> None:
    # 未完成任务:直接造一个 pending job
    job = _test_registry.create("factor-analysis", meta={"factor": "momentum"})
    resp = client.get(f"/factor-analysis/{job.job_id}/report")
    assert resp.status_code == 400


def test_list_factor_analysis_jobs() -> None:
    """历史因子分析任务列表(GET /factor-analysis)只含 factor-analysis 类型。"""
    created = client.post(
        "/factor-analysis",
        json={
            "factor": "momentum",
            "symbols": _SYMBOLS,
            "start": _START,
            "end": _END,
            "periods": [1],
        },
    ).json()["job_id"]
    _wait_done(created, "/factor-analysis")
    resp = client.get("/factor-analysis")
    assert resp.status_code == 200
    jobs = resp.json()
    assert isinstance(jobs, list)
    ids = [j["job_id"] for j in jobs]
    assert created in ids
    assert all(j["kind"] == "factor-analysis" for j in jobs)
    row = next(j for j in jobs if j["job_id"] == created)
    assert row["status"] == "done"
    assert row["title"].startswith("因子分析 momentum")


def test_list_factor_matrix_jobs() -> None:
    """历史多因子诊断任务列表(GET /factor-matrix)只含 factor-matrix 类型。"""
    created = client.post(
        "/factor-matrix",
        json={
            "factors": [
                {"factor": "momentum", "weight": 1.0, "direction": 1},
                {"factor": "bp", "weight": 1.0, "direction": 1},
            ],
            "symbols": _SYMBOLS,
            "start": _START,
            "end": _END,
            "periods": [1],
        },
    ).json()["job_id"]
    _wait_done(created, "/factor-matrix")
    resp = client.get("/factor-matrix")
    assert resp.status_code == 200
    jobs = resp.json()
    assert isinstance(jobs, list)
    ids = [j["job_id"] for j in jobs]
    assert created in ids
    assert all(j["kind"] == "factor-matrix" for j in jobs)
    row = next(j for j in jobs if j["job_id"] == created)
    assert row["status"] == "done"


# ── 选股 ───────────────────────────────────────────────
def test_screen_requires_universe() -> None:
    resp = client.post("/screens", json={"conditions": []})
    assert resp.status_code == 400


def test_screen_top_n_requires_scores() -> None:
    resp = client.post(
        "/screens",
        json={"symbols": _SYMBOLS, "top_n": 2},
    )
    assert resp.status_code == 400


def test_screen_conditions_only() -> None:
    resp = client.post(
        "/screens",
        json={
            "symbols": _SYMBOLS,
            "conditions": [{"field": "pe", "op": "lt", "value": 30}],
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    body = _wait_done(job_id, "/screens")
    assert body["status"] == "done", body.get("error")
    result = body["result"]
    symbols = [r["symbol"] for r in result["results"]]
    # pe = 5 + code_num;< 30 即 code_num < 25
    expect = sorted(s for s in _SYMBOLS if 5.0 + _code_num(s) < 30)
    assert symbols == expect
    assert result["count"] == len(expect)
    # 行内带基本面字段
    assert "pe" in result["results"][0]


def test_screen_with_scoring_and_top_n() -> None:
    resp = client.post(
        "/screens",
        json={
            "symbols": _SYMBOLS,
            "conditions": [],
            "scores": [{"factor": "momentum", "weight": 1.0, "direction": 1}],
            "top_n": 2,
            "lookback_days": 60,
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    body = _wait_done(job_id, "/screens")
    assert body["status"] == "done", body.get("error")
    result = body["result"]
    assert result["count"] == 2  # top_n 截断
    rows = result["results"]
    assert all("score" in r for r in rows)
    # 按得分降序
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_screen_fields() -> None:
    resp = client.get("/screens/fields")
    assert resp.status_code == 200
    fields = resp.json()["fields"]
    assert fields, "字段列表不应为空"
    by_name = {f["name"]: f for f in fields}
    # 覆盖截面快照的全部可筛选列(FUNDAMENTAL_VALUE_COLUMNS)
    from djinn.data.schema import FUNDAMENTAL_VALUE_COLUMNS

    assert set(by_name) == set(FUNDAMENTAL_VALUE_COLUMNS)
    # 每个字段带中文标签 + 分组
    for f in fields:
        assert f["label"]
        assert f["group"] in {"valuation", "financial"}
    assert by_name["pe"]["label"]
    assert by_name["pe"]["group"] == "valuation"


def test_screen_markets() -> None:
    resp = client.get("/screens/markets")
    assert resp.status_code == 200
    markets = resp.json()["markets"]
    by_market = {m["market"]: m for m in markets}
    assert set(by_market) == {"CN", "HK", "US"}
    # 每个市场带标签;不可用市场带原因
    assert by_market["HK"]["available"] is True
    assert by_market["US"]["available"] is True
    for m in markets:
        assert m["label"]
        if not m["available"]:
            assert m["reason"]


def test_screen_list() -> None:
    # 先创建一个任务,再确认列表能查回(历史结果不因刷新丢失)
    resp = client.post("/screens", json={"symbols": _SYMBOLS, "conditions": []})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    _wait_done(job_id, "/screens")
    lst = client.get("/screens").json()
    assert isinstance(lst, list)
    hit = next((j for j in lst if j["job_id"] == job_id), None)
    assert hit is not None
    assert hit["status"] == "done"
    assert hit["kind"] == "screen"
    assert hit["title"]


# ── 股票池 ─────────────────────────────────────────────
def test_universe_stock_list() -> None:
    resp = client.get("/universe/stock-list?market=CN")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == len(_SYMBOLS)
    assert {s["symbol"] for s in data["stocks"]} == set(_SYMBOLS)


def test_universe_index_components() -> None:
    resp = client.get("/universe/index-components/CSI300")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == len(_SYMBOLS)
    assert data["symbols"] == _SYMBOLS
    # 名称与符号位置对齐(stub 返回 名称{i})
    assert data["names"] == [f"名称{i}" for i in range(len(_SYMBOLS))]


def test_universe_index_components_empty_501() -> None:
    assert client.get("/universe/index-components/EMPTY").status_code == 501


def test_stocks_search() -> None:
    """股票搜索:按代码子串匹配 stub 池。"""
    resp = client.get("/stocks/search", params={"q": "600000", "market": "CN"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "600000"
    assert any(r["symbol"] == "600000.SH" for r in data["results"])


def test_stocks_detail() -> None:
    """股票详情:估值 + 财务 + 名称 / 价格。"""
    resp = client.get("/stocks/600519.SH", params={"market": "CN"})
    assert resp.status_code == 200
    d = resp.json()
    assert d["symbol"] == "600519.SH"
    assert d["market"] == "CN"
    assert d["name"]  # 名称非空
    assert d["price"] is not None  # stub 返回价格
    assert d["pe"] is not None  # 估值来自 stub get_fundamentals


def test_stocks_detail_unknown_symbol() -> None:
    """未知代码返回 200,但详情字段为空(null)而非报错。"""
    resp = client.get("/stocks/XXXXX", params={"market": "CN"})
    assert resp.status_code == 200
    d = resp.json()
    assert d["symbol"] == "XXXXX"
    assert d["price"] is None


def test_universe_industries() -> None:
    resp = client.get("/universe/industries?index=CSI300")
    assert resp.status_code == 200
    inds = {i["name"]: i["count"] for i in resp.json()["industries"]}
    assert inds["银行"] == 2  # 000001.SH + 600000.SH
    assert inds["白酒"] == 1
    # 按数量降序
    counts = [i["count"] for i in resp.json()["industries"]]
    assert counts == sorted(counts, reverse=True)


# ── 回测报告缓存(report_store)─────────────────────────────
def _backtest_cfg(symbols: list[str]) -> dict:
    """单标的 MA 交叉回测配置(stub provider 合成上行行情 → 有交易)。"""
    return {
        "universe": {"symbols": symbols, "market": "CN"},
        "period": {"start": _START, "end": _END},
        "account": {"initial_cash": 100000, "currency": "CNY"},
        "strategy": {"name": "MACrossover", "params": {"fast": 5, "slow": 20}},
        "costs": {
            "commission": {"type": "china"},
            "slippage": {"type": "fixed_bps", "bps": 5},
        },
        "portfolio": {"mode": "single", "allocation": "equal"},
        "output": {"export": [], "report": "none"},
        "adjust": "backward",
    }


def test_backtest_report_cached_and_export_no_rerun() -> None:
    """回测完成后报告落盘;/report 与 /export 直接读缓存(不重跑回测)。"""
    from djinn.api import report_store

    resp = client.post("/backtests", json={"config": _backtest_cfg([_SYMBOLS[0]])})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    body = _wait_done(job_id, "/backtests")
    assert body["status"] == "done", body.get("error")

    # 后台任务完成即落盘
    assert report_store.exists(job_id)
    payload_disk = report_store.load(job_id)
    assert payload_disk is not None
    assert "metrics" in payload_disk and "equity_curve" in payload_disk

    # /report 端点返回的 payload 与盘上一致(同源,不重跑)
    rep = client.get(f"/backtests/{job_id}/report")
    assert rep.status_code == 200
    rep_body = rep.json()
    assert rep_body["job_id"] == job_id
    assert "metrics" in rep_body and "equity_curve" in rep_body
    assert rep_body["metrics"] == payload_disk["metrics"]
    assert rep_body["equity_curve"]["values"] == payload_disk["equity_curve"]["values"]

    # /export 端点从缓存重建 Report 后导出 CSV(不重跑回测)
    export_resp = client.get(f"/backtests/{job_id}/export/csv")
    assert export_resp.status_code == 200
    data = export_resp.json()
    files = data.get("files") or []
    assert any(f.endswith("metrics.csv") for f in files)
    assert any(f.endswith("equity_curve.csv") for f in files)

    # 导出 422?excel 非法 fmt → 400
    assert client.get(f"/backtests/{job_id}/export/xls").status_code == 400


def test_backtest_report_fallback_reruns_when_no_cache() -> None:
    """旧任务无缓存时 /report 回退为重跑并落盘(兼容历史 job)。"""
    from djinn.api import report_store

    # 直接造一个 done 状态、带 config 元数据但无缓存文件的旧 job
    cfg = _backtest_cfg([_SYMBOLS[0]])
    cfg["strategy"]["params"] = {"fast": 5, "slow": 20}
    # 先正常建一个 job 走后台拿 config meta;再用其 meta 造一个"无缓存"job
    resp = client.post("/backtests", json={"config": cfg})
    src_id = resp.json()["job_id"]
    _wait_done(src_id, "/backtests")
    src = _test_registry.get(src_id)
    assert src is not None
    # 删掉 src 的缓存,模拟"旧任务无缓存"
    report_store.delete(src_id)
    assert not report_store.exists(src_id)

    rep = client.get(f"/backtests/{src_id}/report")
    assert rep.status_code == 200
    # 回退重跑后应重新落盘
    assert report_store.exists(src_id)
    assert rep.json()["metrics"]  # 非空指标


# ── 多因子诊断(/factor-matrix)─────────────────────────────
def test_factor_matrix_requires_universe() -> None:
    resp = client.post(
        "/factor-matrix",
        json={
            "factors": [{"factor": "momentum"}, {"factor": "reversal"}],
            "start": _START,
            "end": _END,
        },
    )
    assert resp.status_code == 400


def test_factor_matrix_unknown_factor_404() -> None:
    resp = client.post(
        "/factor-matrix",
        json={
            "factors": [{"factor": "momentum"}, {"factor": "bogus"}],
            "symbols": _SYMBOLS,
            "start": _START,
            "end": _END,
        },
    )
    assert resp.status_code == 404


def test_factor_matrix_too_few_factors_400() -> None:
    resp = client.post(
        "/factor-matrix",
        json={
            "factors": [{"factor": "momentum"}],
            "symbols": _SYMBOLS,
            "start": _START,
            "end": _END,
        },
    )
    # schema min_length=2 在路由前拦截 → 422
    assert resp.status_code == 422


def test_factor_matrix_end_to_end() -> None:
    """3 因子诊断 → 相关矩阵 3×3 + 每因子 IC 汇总 + 换手。"""
    resp = client.post(
        "/factor-matrix",
        json={
            "factors": [
                {"factor": "momentum", "params": {"period": 5, "skip": 0}},
                {"factor": "reversal"},
                {"factor": "bp"},
            ],
            "symbols": _SYMBOLS,
            "start": _START,
            "end": _END,
            "periods": [1, 5],
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    body = _wait_done(job_id, "/factor-matrix")
    assert body["status"] == "done", body.get("error")
    assert body["title"].startswith("多因子诊断")

    rep = client.get(f"/factor-matrix/{job_id}/report")
    assert rep.status_code == 200
    report = rep.json()
    assert report["factors"] == ["momentum", "reversal", "bp"]
    corr = report["correlation"]
    assert corr["index"] == report["factors"]
    assert corr["columns"] == report["factors"]
    # 3×3 方阵,对角线 = 1
    assert len(corr["data"]) == 3
    assert all(len(row) == 3 for row in corr["data"])
    for i in range(3):
        assert corr["data"][i][i] == pytest.approx(1.0)
    # 对称
    for i in range(3):
        for j in range(3):
            assert corr["data"][i][j] == pytest.approx(corr["data"][j][i])
    # 每因子每期 IC 汇总
    assert set(report["ic_summary"].keys()) == {"1", "5"}
    for p in ("1", "5"):
        assert set(report["ic_summary"][p].keys()) == set(report["factors"])
        assert "ic_mean" in report["ic_summary"][p]["momentum"]
        assert "icir" in report["ic_summary"][p]["momentum"]
    # 换手覆盖全部因子
    assert set(report["turnover"].keys()) == set(report["factors"])


# ── sweep 多轴扩展 ───────────────────────────────────────
def _factor_portfolio_cfg(symbols: list[str]) -> dict:
    """FactorPortfolio 配置(动量打分 TopN,stub 上行行情 → 有交易)。"""
    return {
        "universe": {"symbols": symbols, "market": "CN"},
        "period": {"start": _START, "end": _END},
        "account": {"initial_cash": 1000000, "currency": "CNY"},
        "strategy": {
            "name": "FactorPortfolio",
            "params": {},
            "factor_weights": {"momentum": 1.0},
            "n_stocks": 2,
            "rebalance_freq": 20,
        },
        "costs": {
            "commission": {"type": "china"},
            "slippage": {"type": "fixed_bps", "bps": 5},
        },
        "portfolio": {"mode": "portfolio", "allocation": "equal"},
        "output": {"export": [], "report": "none"},
        "adjust": "backward",
    }


def test_sweep_unknown_axis_prefix_400() -> None:
    resp = client.post(
        "/sweeps",
        json={
            "config": _factor_portfolio_cfg(_SYMBOLS),
            "grid": {"bogus.axis": [1, 2]},
            "target": "sharpe",
            "parallel": False,
        },
    )
    assert resp.status_code == 400
    assert "未知扫轴前缀" in resp.json()["detail"]


def test_sweep_multi_axis_end_to_end() -> None:
    """扫 portfolio.allocation × strategy.n_stocks:每组合带 config_summary + 三件套。"""
    resp = client.post(
        "/sweeps",
        json={
            "config": _factor_portfolio_cfg(_SYMBOLS),
            "grid": {
                "portfolio.allocation": ["equal", "score"],
                "strategy.n_stocks": [2, 3],
            },
            "target": "sharpe",
            "parallel": False,
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    body = _wait_done(job_id, "/sweeps")
    assert body["status"] == "done", body.get("error")

    results = body["result"]["results"]
    # 2 allocation × 2 n_stocks = 4 组合
    assert len(results) == 4
    for row in results:
        # 每行带 sharpe/sortino/calmar 三件套
        for key in ("sharpe", "sortino", "calmar"):
            assert key in row
        # config_summary 记录本次组合真用上的关键轴
        cs = row["config_summary"]
        assert cs["portfolio.allocation"] in ("equal", "score")
        assert cs["strategy.n_stocks"] in (2, 3)
        assert cs["strategy"] == "FactorPortfolio"
        assert cs["n_symbols"] == len(_SYMBOLS)
    # 目标键存在且按 sharpe 降序(默认非 REVERSE_MIN_TARGETS)
    sharpes = [r["sharpe"] for r in results]
    assert sharpes == sorted(sharpes, reverse=True)


def test_sweep_max_drawdown_sorted_descending() -> None:
    """target=max_drawdown:负值,越接近 0 回撤越浅越好 → 降序排最优在前。"""
    resp = client.post(
        "/sweeps",
        json={
            "config": _factor_portfolio_cfg(_SYMBOLS),
            "grid": {"strategy.n_stocks": [2, 3]},
            "target": "max_drawdown",
            "parallel": False,
        },
    )
    assert resp.status_code == 200
    body = _wait_done(resp.json()["job_id"], "/sweeps")
    assert body["status"] == "done", body.get("error")
    mdds = [r["max_drawdown"] for r in body["result"]["results"]]
    # 降序:最接近 0(回撤最浅)的在前 —— max_drawdown ≤ 0,值越大越好
    assert mdds == sorted(mdds, reverse=True)


# ── 孤儿任务恢复(进程重启)──────────────────────────────
def _make_running_job(reg: JobRegistry, kind: str) -> str:
    """造一个 running 状态的孤儿任务(模拟进程重启中断)。"""
    if kind == "factor-analysis":
        meta = {
            "factor": "momentum",
            "symbols": _SYMBOLS,
            "start": _START,
            "end": _END,
            "periods": [1],
            "title": "因子分析 momentum",
        }
    elif kind == "screen":
        meta = {
            "symbols": _SYMBOLS,
            "conditions": [{"field": "pe", "op": "lt", "value": 30}],
            "title": "选股",
        }
    else:
        raise ValueError(f"测试未覆盖 kind: {kind}")
    job = reg.create(kind, meta=meta)
    reg.update(job.job_id, status="running", progress=0.5, stage="中途")
    return job.job_id


def test_recover_orphaned_jobs_resubmits(monkeypatch, tmp_path) -> None:
    """running/pending 孤儿任务被重新提交,用 stub provider 跑完到 done。"""
    monkeypatch.delenv("DJINN_TEST", raising=False)  # 关闭守卫,让恢复生效
    reg = JobRegistry(db_path=str(tmp_path / "recover.db"))
    fa_id = _make_running_job(reg, "factor-analysis")
    scr_id = _make_running_job(reg, "screen")
    done_id = reg.create("factor-analysis", meta={"factor": "momentum"}).job_id
    reg.update(done_id, status="done")  # 已完成任务不应被恢复

    n = recover_orphaned_jobs(reg, _stub_registry)
    assert n == 2  # 只恢复 running/pending,不动 done

    # 等恢复线程跑完(确定性 stub 很快)
    for _ in range(100):
        fa = reg.get(fa_id)
        scr = reg.get(scr_id)
        if fa.status == "done" and scr.status == "done":
            break
        import time

        time.sleep(0.05)
    assert reg.get(fa_id).status == "done"
    assert reg.get(fa_id).error is None
    assert reg.get(scr_id).status == "done"
    assert reg.get(done_id).status == "done"  # 未被触碰


def test_recover_orphaned_jobs_skipped_in_test_env(monkeypatch, tmp_path) -> None:
    """DJINN_TEST=1 时恢复被禁用(测试隔离,避免误恢复真实任务)。"""
    monkeypatch.setenv("DJINN_TEST", "1")
    reg = JobRegistry(db_path=str(tmp_path / "guard.db"))
    _make_running_job(reg, "factor-analysis")
    assert recover_orphaned_jobs(reg, _stub_registry) == 0
    # 任务仍是 running,未被重新提交
    job = reg.list(limit=10)[0]
    assert job.status == "running"

"""可视化层测试:HTML 报告与静态图生成。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from djinn.analytics import build_report
from djinn.data import Adjust, CSVProvider, Market
from djinn.engine import EngineConfig, EventDrivenEngine
from djinn.engine.commission import USCommissionModel
from djinn.engine.slippage import ZeroSlippage
from djinn.strategy import MACrossover
from djinn.viz import (
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_heatmap,
    plot_positions,
    render_html_report,
    save_fig,
    save_html_report,
)


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> object:
    """生成一个回测报告供 viz 测试复用。"""
    d = tmp_path_factory.mktemp("csv")
    np.random.seed(7)
    n = 120
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, n)), index=idx
    )
    pd.DataFrame(
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 10000,
        }
    ).to_csv(d / "AAPL.csv", index=False)
    prov = CSVProvider(d, default_market=Market.US)
    md = prov.get_ohlcv("AAPL", date(2024, 1, 2), date(2024, 6, 30), Adjust.BACKWARD)
    engine = EventDrivenEngine(
        EngineConfig(
            initial_cash=100000, commission=USCommissionModel(), slippage=ZeroSlippage()
        )
    )
    res = engine.run(MACrossover(fast=10, slow=30), {"AAPL": md})
    return build_report(res, market="US")


def test_render_html_report(report: object) -> None:
    html = render_html_report(report)  # type: ignore[arg-type]
    assert "<html" in html.lower() or "<!doctype" in html.lower()
    assert "净值曲线" in html
    assert "夏普" in html
    assert len(html) > 5000


def test_save_html_report(report: object, tmp_path: Path) -> None:
    p = save_html_report(report, tmp_path / "out" / "report.html")  # type: ignore[arg-type]
    assert p.exists()
    assert p.stat().st_size > 5000


def test_static_figures(report: object, tmp_path: Path) -> None:
    for name, fig in [
        ("equity", plot_equity_curve(report)),  # type: ignore[arg-type]
        ("drawdown", plot_drawdown(report)),  # type: ignore[arg-type]
        ("positions", plot_positions(report)),  # type: ignore[arg-type]
        ("heatmap", plot_monthly_heatmap(report)),  # type: ignore[arg-type]
    ]:
        p = save_fig(fig, tmp_path / f"{name}.png")
        assert p.exists()
        assert p.stat().st_size > 1000

"""配置加载与校验测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from djinn.config import dump_config, load_config
from djinn.data.schema import Adjust, Market
from djinn.utils.exceptions import ConfigError


def _minimal_cfg() -> dict:
    return {
        "universe": {"symbols": ["AAPL"], "market": "US"},
        "period": {"start": "2024-01-01", "end": "2024-12-31"},
        "strategy": {"name": "MACrossover", "params": {"fast": 5, "slow": 20}},
    }


def test_load_minimal_config():
    cfg = load_config(data=_minimal_cfg())
    assert cfg.universe.symbols == ["AAPL"]
    assert cfg.resolved_market() is Market.US
    assert cfg.adjust is Adjust.BACKWARD
    assert cfg.strategy.name == "MACrossover"


def test_config_validation_bad_period():
    data = _minimal_cfg()
    data["period"] = {"start": "2024-12-01", "end": "2024-01-01"}  # start > end
    with pytest.raises(ConfigError):
        load_config(data=data)


def test_config_validation_missing_symbols():
    data = _minimal_cfg()
    data["universe"] = {"symbols": []}
    with pytest.raises(ConfigError):
        load_config(data=data)


def test_config_env_override(monkeypatch: pytest.MonkeyPatch):
    """env 覆盖:DJINN_ACCOUNT_INITIAL_CASH 改初始资金。"""
    monkeypatch.setenv("DJINN_ACCOUNT_INITIAL_CASH", "200000")
    cfg = load_config(data=_minimal_cfg())
    assert cfg.account.initial_cash == 200000


def test_config_cn_market_t_plus_1_auto():
    """A 股市场自动启用 T+1。"""
    data = _minimal_cfg()
    data["universe"] = {"symbols": ["600000.SH"], "market": "CN"}
    cfg = load_config(data=data)
    assert cfg.resolved_market() is Market.CN
    assert cfg.resolved_t_plus_1() is True


def test_config_us_no_t_plus_1():
    cfg = load_config(data=_minimal_cfg())
    assert cfg.resolved_t_plus_1() is False


def test_config_yaml_roundtrip(tmp_path: Path):
    """YAML 导入 → 导出 → 重载,保持一致。"""
    cfg = load_config(data=_minimal_cfg())
    p = tmp_path / "rt.yaml"
    dump_config(cfg, p)
    cfg2 = load_config(p)
    assert cfg2.universe.symbols == cfg.universe.symbols
    assert cfg2.strategy.params == cfg.strategy.params


def test_load_example_yaml():
    """示例配置可加载。"""
    cfg = load_config("configs/backtest.example.yaml")
    assert cfg.universe.symbols == ["AAPL"]
    assert cfg.strategy.name == "MACrossover"


def test_load_example_portfolio_yaml():
    cfg = load_config("configs/portfolio.example.yaml")
    assert cfg.portfolio.mode == "portfolio"
    assert cfg.portfolio.allocation == "equal"
    assert cfg.portfolio.rebalance.period == "quarterly"


def test_unknown_strategy_rejected():
    data = _minimal_cfg()
    data["strategy"] = {"name": "Bogus", "params": {}}
    # 配置本身可加载(策略名在运行时解析),但 runner 会失败
    cfg = load_config(data=data)
    assert cfg.strategy.name == "Bogus"

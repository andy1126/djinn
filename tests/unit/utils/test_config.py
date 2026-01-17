"""
Unit tests for the config module.
"""

import pytest
import yaml
import json
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock
import tempfile
import os

from src.djinn.utils.config import (
    ConfigManager,
    MarketType,
    DataSourceType,
    IntervalType,
    PositionSizingMethod,
    RebalanceFrequency,
    AllocationMethod,
    OptimizationMethod,
    RiskConfig,
    BacktestConfig,
    Settings,
)
from src.djinn.utils.exceptions import ConfigurationError


class TestConfigEnums:
    """Test configuration enumerations."""

    def test_market_type_enum(self):
        """Test MarketType enum values."""
        assert MarketType.US == "US"
        assert MarketType.HK == "HK"
        assert MarketType.CN == "CN"

    def test_data_source_type_enum(self):
        """Test DataSourceType enum values."""
        assert DataSourceType.YAHOO_FINANCE == "yahoo_finance"
        assert DataSourceType.AKSHARE == "akshare"
        assert DataSourceType.TUSHARE == "tushare"
        assert DataSourceType.LOCAL == "local"

    def test_interval_type_enum(self):
        """Test IntervalType enum values."""
        assert IntervalType.DAILY == "1d"
        assert IntervalType.HOURLY == "1h"
        assert IntervalType.MINUTE_30 == "30m"
        assert IntervalType.MINUTE_15 == "15m"
        assert IntervalType.MINUTE_5 == "5m"
        assert IntervalType.MINUTE_1 == "1m"

    def test_position_sizing_method_enum(self):
        """Test PositionSizingMethod enum values."""
        assert PositionSizingMethod.FIXED_FRACTIONAL == "fixed_fractional"
        assert PositionSizingMethod.FIXED_UNITS == "fixed_units"
        assert PositionSizingMethod.PERCENT_RISK == "percent_risk"
        assert PositionSizingMethod.KELLY == "kelly"

    def test_rebalance_frequency_enum(self):
        """Test RebalanceFrequency enum values."""
        assert RebalanceFrequency.DAILY == "daily"
        assert RebalanceFrequency.WEEKLY == "weekly"
        assert RebalanceFrequency.MONTHLY == "monthly"
        assert RebalanceFrequency.QUARTERLY == "quarterly"
        assert RebalanceFrequency.YEARLY == "yearly"

    def test_allocation_method_enum(self):
        """Test AllocationMethod enum values."""
        assert AllocationMethod.EQUAL_WEIGHT == "equal_weight"
        assert AllocationMethod.MARKET_CAP == "market_cap"
        assert AllocationMethod.RISK_PARITY == "risk_parity"
        assert AllocationMethod.MIN_VARIANCE == "min_variance"
        assert AllocationMethod.MAX_SHARPE == "max_sharpe"

    def test_optimization_method_enum(self):
        """Test OptimizationMethod enum values."""
        assert OptimizationMethod.GRID_SEARCH == "grid_search"
        assert OptimizationMethod.RANDOM_SEARCH == "random_search"
        assert OptimizationMethod.BAYESIAN == "bayesian"
        assert OptimizationMethod.GENETIC == "genetic"


class TestRiskConfig:
    """Test RiskConfig model."""

    def test_risk_config_defaults(self):
        """Test RiskConfig with default values."""
        config = RiskConfig()

        assert config.max_position_size == 0.1
        assert config.stop_loss == 0.1
        assert config.take_profit == 0.2
        assert config.max_drawdown == 0.2
        assert config.volatility_limit == 0.3

    def test_risk_config_custom_values(self):
        """Test RiskConfig with custom values."""
        config = RiskConfig(
            max_position_size=0.2,
            stop_loss=0.05,
            take_profit=0.15,
            max_drawdown=0.15,
            volatility_limit=0.2,
        )

        assert config.max_position_size == 0.2
        assert config.stop_loss == 0.05
        assert config.take_profit == 0.15
        assert config.max_drawdown == 0.15
        assert config.volatility_limit == 0.2

    def test_risk_config_validation(self):
        """Test RiskConfig validation."""
        # Valid values should pass
        RiskConfig(max_position_size=0.5, stop_loss=0.2, take_profit=0.3)

        # Invalid values should raise validation error
        with pytest.raises(ValueError):
            RiskConfig(max_position_size=-0.1)  # Negative value

        with pytest.raises(ValueError):
            RiskConfig(max_position_size=1.5)  # Greater than 1.0


class TestBacktestConfig:
    """Test BacktestConfig model."""

    def test_backtest_config_defaults(self):
        """Test BacktestConfig with default values."""
        config = BacktestConfig(
            initial_capital=100000.0,
            start_date="2020-01-01",
            end_date="2023-12-31",
        )

        assert config.initial_capital == 100000.0
        assert config.start_date == "2020-01-01"
        assert config.end_date == "2023-12-31"
        assert config.commission == 0.001  # Default value
        assert config.slippage == 0.0005   # Default value
        assert config.tax_rate == 0.001    # Default value

    def test_backtest_config_custom_values(self):
        """Test BacktestConfig with custom values."""
        config = BacktestConfig(
            initial_capital=50000.0,
            start_date="2021-01-01",
            end_date="2022-12-31",
            commission=0.002,
            slippage=0.001,
            tax_rate=0.002,
            data_source=DataSourceType.YAHOO_FINANCE,
            interval=IntervalType.DAILY,
            adjust="adj",
            cache_enabled=True,
            cache_ttl=7200,
        )

        assert config.initial_capital == 50000.0
        assert config.commission == 0.002
        assert config.slippage == 0.001
        assert config.tax_rate == 0.002
        assert config.data_source == DataSourceType.YAHOO_FINANCE
        assert config.interval == IntervalType.DAILY
        assert config.adjust == "adj"
        assert config.cache_enabled is True
        assert config.cache_ttl == 7200

    def test_backtest_config_with_risk_config(self):
        """Test BacktestConfig with nested RiskConfig."""
        risk_config = RiskConfig(
            max_position_size=0.15,
            stop_loss=0.08,
            take_profit=0.25,
            max_drawdown=0.25,
            volatility_limit=0.25,
        )

        config = BacktestConfig(
            initial_capital=100000.0,
            start_date="2020-01-01",
            end_date="2023-12-31",
            risk=risk_config,
        )

        assert config.risk == risk_config
        assert config.risk.max_position_size == 0.15


class TestConfigManager:
    """Test ConfigManager class."""

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()

        assert isinstance(manager.config_dir, Path)
        assert manager._config_cache == {}
        assert isinstance(manager.settings, Settings)

    def test_load_config_yaml_success(self):
        """Test successful YAML config loading."""
        manager = ConfigManager()
        config_content = """
        backtest:
          initial_capital: 100000
          start_date: "2020-01-01"
          end_date: "2023-12-31"
        """

        with patch('builtins.open', mock_open(read_data=config_content)):
            with patch('yaml.safe_load', return_value=yaml.safe_load(config_content)):
                config = manager.load_config("test_config.yaml", config_type="yaml")

                assert "backtest" in config
                assert config["backtest"]["initial_capital"] == 100000
                assert config["backtest"]["start_date"] == "2020-01-01"
                assert config["backtest"]["end_date"] == "2023-12-31"

    def test_load_config_json_success(self):
        """Test successful JSON config loading."""
        manager = ConfigManager()
        config_content = {
            "backtest": {
                "initial_capital": 100000,
                "start_date": "2020-01-01",
                "end_date": "2023-12-31"
            }
        }

        with patch('builtins.open', mock_open(read_data=json.dumps(config_content))):
            with patch('json.load', return_value=config_content):
                config = manager.load_config("test_config.json", config_type="json")

                assert "backtest" in config
                assert config["backtest"]["initial_capital"] == 100000

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        manager = ConfigManager()

        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(ConfigurationError) as exc_info:
                manager.load_config("non_existent.yaml")

            assert "找不到配置文件" in str(exc_info.value)

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML config."""
        manager = ConfigManager()
        invalid_yaml = "invalid: [yaml: content"

        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with pytest.raises(ConfigurationError) as exc_info:
                manager.load_config("invalid.yaml")

            assert "解析配置文件失败" in str(exc_info.value)

    def test_load_config_cache(self):
        """Test config loading with caching."""
        manager = ConfigManager()
        config_content = """
        backtest:
          initial_capital: 100000
          start_date: "2020-01-01"
          end_date: "2023-12-31"
        """
        parsed_config = yaml.safe_load(config_content)

        with patch('builtins.open', mock_open(read_data=config_content)):
            with patch('yaml.safe_load', return_value=parsed_config):
                # First load - should read from file
                config1 = manager.load_config("test.yaml")

                # Second load - should use cache
                config2 = manager.load_config("test.yaml")

                assert config1 == config2
                # Verify cache was used (file should only be read once)
                # We can't easily verify this without more complex mocking

    def test_save_config_yaml(self):
        """Test saving config as YAML."""
        manager = ConfigManager()
        config_data = {
            "backtest": {
                "initial_capital": 100000,
                "start_date": "2020-01-01",
                "end_date": "2023-12-31"
            }
        }

        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch.object(Path, 'mkdir') as mock_mkdir:
                with patch('yaml.dump') as mock_yaml_dump:
                    manager.save_config(config_data, "test.yaml", config_type="yaml")

                    # Verify directory creation
                    mock_mkdir.assert_called_with(parents=True, exist_ok=True)

                    # Verify YAML dump was called
                    mock_yaml_dump.assert_called_once()

                    # Verify cache was updated
                    cache_key = "test.yaml:yaml"
                    assert cache_key in manager._config_cache
                    assert manager._config_cache[cache_key] == config_data

    def test_save_config_json(self):
        """Test saving config as JSON."""
        manager = ConfigManager()
        config_data = {"test": "data"}

        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch.object(Path, 'mkdir'):
                with patch('json.dump') as mock_json_dump:
                    manager.save_config(config_data, "test.json", config_type="json")

                    # Verify JSON dump was called
                    mock_json_dump.assert_called_once()

    def test_save_config_invalid_type(self):
        """Test saving config with invalid type."""
        manager = ConfigManager()
        config_data = {"test": "data"}

        with pytest.raises(ConfigurationError) as exc_info:
            manager.save_config(config_data, "test.xml", config_type="xml")

        assert "不支持的配置类型" in str(exc_info.value)

    def test_get_and_update_setting(self):
        """Test getting and updating settings."""
        manager = ConfigManager()

        # Test getting existing setting
        assert hasattr(manager.settings, 'log_level')

        # Test getting non-existent setting with default
        value = manager.get_setting("non_existent", "default_value")
        assert value == "default_value"

        # Test updating setting
        manager.update_setting("test_key", "test_value")
        assert manager.get_setting("test_key") == "test_value"

    def test_clear_cache(self):
        """Test clearing config cache."""
        manager = ConfigManager()

        # Add something to cache
        manager._config_cache["test"] = {"data": "value"}
        assert "test" in manager._config_cache

        # Clear cache
        manager.clear_cache()
        assert manager._config_cache == {}

    def test_get_config_path_absolute(self):
        """Test getting absolute config path."""
        manager = ConfigManager()

        absolute_path = "/absolute/path/config.yaml"
        result = manager._get_config_path(absolute_path, "yaml")

        assert str(result) == absolute_path

    def test_get_config_path_relative(self):
        """Test getting relative config path."""
        manager = ConfigManager()

        # Mock config_dir to predictable value
        manager.config_dir = Path("/mock/config/dir")

        result = manager._get_config_path("test", "yaml")

        assert result == Path("/mock/config/dir/test.yaml")

    def test_validate_config_success(self):
        """Test successful config validation."""
        manager = ConfigManager()

        config_data = {
            "backtest": {
                "start_date": "2020-01-01",
                "end_date": "2023-12-31"
            }
        }

        result = manager._validate_config(config_data, "test.yaml")
        assert result == config_data

    def test_validate_config_missing_fields(self):
        """Test config validation with missing required fields."""
        manager = ConfigManager()

        config_data = {
            "backtest": {
                # Missing start_date and end_date
                "initial_capital": 100000
            }
        }

        with pytest.raises(ConfigurationError) as exc_info:
            manager._validate_config(config_data, "test.yaml")

        assert "配置缺少必需字段" in str(exc_info.value)
        assert "backtest.start_date" in str(exc_info.value)
        assert "backtest.end_date" in str(exc_info.value)


class TestIntegration:
    """Integration tests for config module."""

    def test_load_real_yaml_config(self):
        """Test loading a real YAML config file."""
        manager = ConfigManager()

        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "backtest": {
                    "initial_capital": 50000,
                    "start_date": "2021-01-01",
                    "end_date": "2022-12-31",
                    "commission": 0.001,
                    "slippage": 0.0005,
                }
            }, f)
            temp_file = f.name

        try:
            # Load the config
            config = manager.load_config(temp_file)

            assert "backtest" in config
            assert config["backtest"]["initial_capital"] == 50000
            assert config["backtest"]["start_date"] == "2021-01-01"

        finally:
            # Clean up
            os.unlink(temp_file)

    def test_save_and_load_roundtrip(self):
        """Test save and load roundtrip."""
        manager = ConfigManager()

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_data = {
                "test": {
                    "value": 123,
                    "nested": {
                        "item": "test"
                    }
                }
            }

            # Save config
            manager.save_config(config_data, str(config_file))

            # Load config
            loaded_config = manager.load_config(str(config_file))

            # Verify roundtrip
            assert loaded_config == config_data
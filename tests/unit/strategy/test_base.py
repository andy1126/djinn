"""
Unit tests for the strategy base module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock

from src.djinn.core.strategy.base import (
    SignalType,
    PositionType,
    Signal,
    PositionSizing,
    Strategy,
)
from src.djinn.utils.exceptions import ValidationError, StrategyError
from src.djinn.data.market_data import MarketData


class TestSignalType:
    """Test SignalType enumeration."""

    def test_signal_type_values(self):
        """Test SignalType enum values."""
        assert SignalType.BUY == "BUY"
        assert SignalType.SELL == "SELL"
        assert SignalType.HOLD == "HOLD"
        assert SignalType.EXIT == "EXIT"


class TestPositionType:
    """Test PositionType enumeration."""

    def test_position_type_values(self):
        """Test PositionType enum values."""
        assert PositionType.LONG == "LONG"
        assert PositionType.SHORT == "SHORT"


class TestSignal:
    """Test Signal data class."""

    def test_signal_creation(self):
        """Test Signal creation with valid data."""
        timestamp = datetime(2023, 12, 31, 14, 30, 45)
        signal = Signal(
            timestamp=timestamp,
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            strength=0.8,
            reason="Crossing above MA",
            metadata={"indicator": "MA", "period": 20}
        )

        assert signal.timestamp == timestamp
        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.price == 150.0
        assert signal.strength == 0.8
        assert signal.reason == "Crossing above MA"
        assert signal.metadata == {"indicator": "MA", "period": 20}

    def test_signal_validation_strength_too_low(self):
        """Test Signal validation with strength below 0."""
        with pytest.raises(ValidationError) as exc_info:
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=150.0,
                strength=-0.1
            )

        assert "信号强度必须在 0.0 到 1.0 之间" in str(exc_info.value)
        assert exc_info.value.field == "strength"

    def test_signal_validation_strength_too_high(self):
        """Test Signal validation with strength above 1."""
        with pytest.raises(ValidationError) as exc_info:
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=150.0,
                strength=1.1
            )

        assert "信号强度必须在 0.0 到 1.0 之间" in str(exc_info.value)

    def test_signal_validation_invalid_price(self):
        """Test Signal validation with invalid price."""
        with pytest.raises(ValidationError) as exc_info:
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=0.0
            )

        assert "价格必须大于 0" in str(exc_info.value)
        assert exc_info.value.field == "price"

    def test_signal_to_dict(self):
        """Test Signal.to_dict method."""
        timestamp = datetime(2023, 12, 31, 14, 30, 45)
        signal = Signal(
            timestamp=timestamp,
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            strength=0.8,
            reason="Test reason",
            metadata={"key": "value"}
        )

        result = signal.to_dict()

        assert result["timestamp"] == timestamp
        assert result["symbol"] == "AAPL"
        assert result["signal_type"] == "BUY"
        assert result["price"] == 150.0
        assert result["strength"] == 0.8
        assert result["reason"] == "Test reason"
        assert result["metadata"] == {"key": "value"}

    def test_signal_from_dict(self):
        """Test Signal.from_dict method."""
        data = {
            "timestamp": datetime(2023, 12, 31, 14, 30, 45),
            "symbol": "AAPL",
            "signal_type": "BUY",
            "price": 150.0,
            "strength": 0.8,
            "reason": "Test reason",
            "metadata": {"key": "value"}
        }

        signal = Signal.from_dict(data)

        assert signal.timestamp == data["timestamp"]
        assert signal.symbol == data["symbol"]
        assert signal.signal_type == SignalType.BUY
        assert signal.price == data["price"]
        assert signal.strength == data["strength"]
        assert signal.reason == data["reason"]
        assert signal.metadata == data["metadata"]


class TestPositionSizing:
    """Test PositionSizing data class."""

    def test_position_sizing_creation(self):
        """Test PositionSizing creation with valid data."""
        sizing = PositionSizing(
            method="fixed_fractional",
            risk_per_trade=0.02,  # 2%
            max_risk=0.1,  # 10%
            fixed_size=100.0,
            max_position_size=0.1  # 10%
        )

        assert sizing.method == "fixed_fractional"
        assert sizing.risk_per_trade == 0.02
        assert sizing.max_risk == 0.1
        assert sizing.fixed_size == 100.0
        assert sizing.max_position_size == 0.1

    def test_position_sizing_validation_risk_per_trade_too_low(self):
        """Test PositionSizing validation with risk_per_trade <= 0."""
        with pytest.raises(ValidationError) as exc_info:
            PositionSizing(risk_per_trade=0.0)

        assert "每笔交易风险必须在 0.0 到 1.0 之间" in str(exc_info.value)
        assert exc_info.value.field == "risk_per_trade"

    def test_position_sizing_validation_risk_per_trade_too_high(self):
        """Test PositionSizing validation with risk_per_trade > 1."""
        with pytest.raises(ValidationError) as exc_info:
            PositionSizing(risk_per_trade=1.1)

        assert "每笔交易风险必须在 0.0 到 1.0 之间" in str(exc_info.value)

    def test_position_sizing_validation_max_risk_too_low(self):
        """Test PositionSizing validation with max_risk <= 0."""
        with pytest.raises(ValidationError) as exc_info:
            PositionSizing(max_risk=0.0)

        assert "最大总风险必须在 0.0 到 1.0 之间" in str(exc_info.value)
        assert exc_info.value.field == "max_risk"

    def test_position_sizing_validation_risk_per_trade_greater_than_max_risk(self):
        """Test PositionSizing validation when risk_per_trade > max_risk."""
        with pytest.raises(ValidationError) as exc_info:
            PositionSizing(risk_per_trade=0.2, max_risk=0.1)

        assert "每笔交易风险不能大于最大总风险" in str(exc_info.value)

    def test_position_sizing_validation_max_position_size_invalid(self):
        """Test PositionSizing validation with invalid max_position_size."""
        with pytest.raises(ValidationError) as exc_info:
            PositionSizing(max_position_size=0.0)

        assert "最大单仓位比例必须在 0.0 到 1.0 之间" in str(exc_info.value)

    def test_position_sizing_validation_fixed_size_invalid(self):
        """Test PositionSizing validation with invalid fixed_size."""
        with pytest.raises(ValidationError) as exc_info:
            PositionSizing(fixed_size=0.0)

        assert "固定仓位大小必须大于 0" in str(exc_info.value)


class TestConcreteStrategy(Strategy):
    """Concrete strategy implementation for testing abstract base class."""

    def _validate_parameters(self, parameters):
        """Simple parameter validation for testing."""
        if "invalid" in parameters:
            raise ValidationError("Invalid parameter")
        return parameters

    def initialize(self, data):
        """Test implementation of initialize."""
        self.initialized = True
        self.test_data = data

    def generate_signals(self, data):
        """Test implementation of generate_signals."""
        return [
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=100.0
            )
        ]

    def calculate_indicators(self, data):
        """Test implementation of calculate_indicators."""
        return {"test_indicator": pd.Series([1, 2, 3])}


class TestStrategyBaseClass:
    """Test Strategy abstract base class."""

    def test_strategy_initialization(self):
        """Test Strategy initialization."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={"param1": "value1", "param2": 123}
        )

        assert strategy.name == "TestStrategy"
        assert strategy.parameters == {"param1": "value1", "param2": 123}
        assert isinstance(strategy.position_sizing, PositionSizing)
        assert strategy.initialized is False
        assert strategy.signals == []
        assert strategy.indicators == {}
        assert "name" in strategy.metadata
        assert "parameters" in strategy.metadata
        assert "created_at" in strategy.metadata

    def test_strategy_initialization_with_position_sizing(self):
        """Test Strategy initialization with custom PositionSizing."""
        position_sizing = PositionSizing(
            method="fixed_units",
            fixed_size=100.0
        )

        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={},
            position_sizing=position_sizing
        )

        assert strategy.position_sizing == position_sizing
        assert strategy.position_sizing.method == "fixed_units"
        assert strategy.position_sizing.fixed_size == 100.0

    def test_strategy_parameter_validation(self):
        """Test Strategy parameter validation."""
        # Should pass with valid parameters
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={"valid": "parameter"}
        )
        assert strategy.parameters == {"valid": "parameter"}

        # Should raise error with invalid parameters
        with pytest.raises(ValidationError):
            TestConcreteStrategy(
                name="TestStrategy",
                parameters={"invalid": "parameter"}
            )

    def test_calculate_position_size_fixed_fractional(self):
        """Test calculate_position_size with fixed_fractional method."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )
        strategy.position_sizing.method = "fixed_fractional"
        strategy.position_sizing.risk_per_trade = 0.02  # 2%

        # Test without stop loss
        position_size = strategy.calculate_position_size(
            capital=10000.0,
            price=100.0
        )

        # Expected: risk_amount = 10000 * 0.02 = 200
        # risk_per_share = 100 * 0.01 = 1 (default 1% risk when no stop loss)
        # position_size = 200 / 1 = 200 shares
        assert position_size == 200

    def test_calculate_position_size_fixed_fractional_with_stop_loss(self):
        """Test calculate_position_size with fixed_fractional method and stop loss."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )
        strategy.position_sizing.method = "fixed_fractional"
        strategy.position_sizing.risk_per_trade = 0.02  # 2%

        # Test with stop loss
        position_size = strategy.calculate_position_size(
            capital=10000.0,
            price=100.0,
            stop_loss=95.0
        )

        # Expected: risk_amount = 10000 * 0.02 = 200
        # risk_per_share = 100 - 95 = 5
        # position_size = 200 / 5 = 40 shares
        assert position_size == 40

    def test_calculate_position_size_fixed_units(self):
        """Test calculate_position_size with fixed_units method."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )
        strategy.position_sizing.method = "fixed_units"
        strategy.position_sizing.fixed_size = 150.0

        position_size = strategy.calculate_position_size(
            capital=10000.0,
            price=100.0
        )

        # Should use fixed_size
        assert position_size == 150

    def test_calculate_position_size_fixed_units_default(self):
        """Test calculate_position_size with fixed_units method and no fixed_size."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )
        strategy.position_sizing.method = "fixed_units"
        strategy.position_sizing.fixed_size = None

        position_size = strategy.calculate_position_size(
            capital=10000.0,
            price=100.0
        )

        # Should use default (100 shares)
        assert position_size == 100

    def test_calculate_position_size_percent_risk(self):
        """Test calculate_position_size with percent_risk method."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )
        strategy.position_sizing.method = "percent_risk"
        strategy.position_sizing.risk_per_trade = 0.02  # 2%

        position_size = strategy.calculate_position_size(
            capital=10000.0,
            price=100.0
        )

        # Expected: position_value = 10000 * 0.02 = 200
        # position_size = 200 / 100 = 2 shares
        assert position_size == 2

    def test_calculate_position_size_percent_risk_with_custom_risk(self):
        """Test calculate_position_size with percent_risk method and custom risk."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )
        strategy.position_sizing.method = "percent_risk"

        # Provide custom risk amount
        position_size = strategy.calculate_position_size(
            capital=10000.0,
            price=100.0,
            risk=500.0  # $500 risk
        )

        # Expected: position_value = 500
        # position_size = 500 / 100 = 5 shares
        assert position_size == 5

    def test_calculate_position_size_kelly(self):
        """Test calculate_position_size with kelly method."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )
        strategy.position_sizing.method = "kelly"

        position_size = strategy.calculate_position_size(
            capital=10000.0,
            price=100.0
        )

        # Kelly formula uses default win_rate=0.5 and win_loss_ratio=2.0
        # f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        # f = (0.5*2 - 0.5) / 2 = (1 - 0.5) / 2 = 0.5 / 2 = 0.25
        # position_value = 10000 * 0.25 = 2500
        # position_size = 2500 / 100 = 25 shares
        assert position_size == 25

    def test_calculate_position_size_unknown_method(self):
        """Test calculate_position_size with unknown method."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )
        strategy.position_sizing.method = "unknown_method"

        with pytest.raises(StrategyError) as exc_info:
            strategy.calculate_position_size(
                capital=10000.0,
                price=100.0
            )

        assert "不支持的仓位管理方法" in str(exc_info.value)

    def test_calculate_position_size_zero_price(self):
        """Test calculate_position_size with zero price."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )

        with pytest.raises(StrategyError) as exc_info:
            strategy.calculate_position_size(
                capital=10000.0,
                price=0.0
            )

        assert "价格必须大于 0" in str(exc_info.value)

    def test_calculate_position_size_zero_capital(self):
        """Test calculate_position_size with zero capital."""
        strategy = TestConcreteStrategy(
            name="TestStrategy",
            parameters={}
        )

        position_size = strategy.calculate_position_size(
            capital=0.0,
            price=100.0
        )

        # Should return 0 shares when capital is 0
        assert position_size == 0

    def test_strategy_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Cannot instantiate abstract Strategy directly
        with pytest.raises(TypeError):
            Strategy("Test", {})

    def test_strategy_concrete_implementation(self):
        """Test concrete strategy implementation."""
        strategy = TestConcreteStrategy("Test", {})

        # Test initialize
        mock_data = Mock(spec=MarketData)
        strategy.initialize(mock_data)
        assert strategy.initialized is True
        assert strategy.test_data == mock_data

        # Test generate_signals
        signals = strategy.generate_signals(mock_data)
        assert len(signals) == 1
        assert isinstance(signals[0], Signal)
        assert signals[0].symbol == "TEST"

        # Test calculate_indicators
        indicators = strategy.calculate_indicators(mock_data)
        assert "test_indicator" in indicators
        assert isinstance(indicators["test_indicator"], pd.Series)
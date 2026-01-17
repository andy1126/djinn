"""
Unit tests for the technical indicators module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.djinn.core.strategy.indicators import (
    TechnicalIndicators,
    IndicatorType,
    IndicatorResult,
)
from src.djinn.utils.exceptions import IndicatorError, ValidationError


class TestIndicatorType:
    """Test IndicatorType enumeration."""

    def test_indicator_type_values(self):
        """Test IndicatorType enum values."""
        assert IndicatorType.TREND == "trend"
        assert IndicatorType.MOMENTUM == "momentum"
        assert IndicatorType.VOLATILITY == "volatility"
        assert IndicatorType.VOLUME == "volume"
        assert IndicatorType.OSCILLATOR == "oscillator"


class TestIndicatorResult:
    """Test IndicatorResult data class."""

    def test_indicator_result_creation(self):
        """Test IndicatorResult creation."""
        values = pd.Series([1.0, 2.0, 3.0])
        signals = pd.Series([1, -1, 0])
        metadata = {"type": "test", "window": 10}

        result = IndicatorResult(
            values=values,
            signals=signals,
            metadata=metadata
        )

        pd.testing.assert_series_equal(result.values, values)
        pd.testing.assert_series_equal(result.signals, signals)
        assert result.metadata == metadata

    def test_indicator_result_creation_minimal(self):
        """Test IndicatorResult creation with minimal parameters."""
        values = pd.Series([1.0, 2.0, 3.0])

        result = IndicatorResult(values=values)

        pd.testing.assert_series_equal(result.values, values)
        assert result.signals is None
        assert result.metadata is None


class TestTechnicalIndicators:
    """Test TechnicalIndicators class."""

    def setup_method(self):
        """Setup test data."""
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Generate prices with some trend and noise
        trend = np.linspace(100, 200, 100)
        noise = np.random.normal(0, 5, 100)
        self.prices = pd.Series(trend + noise, index=dates, name='close')

        # Volume data
        self.volume = pd.Series(
            np.random.randint(1000, 10000, 100),
            index=dates,
            name='volume'
        )

        # OHLC data for testing
        self.high = self.prices * 1.02  # High is 2% above close
        self.low = self.prices * 0.98   # Low is 2% below close
        self.open = self.prices * 0.99  # Open is 1% below close

    def test_simple_moving_average_basic(self):
        """Test simple moving average calculation."""
        result = TechnicalIndicators.simple_moving_average(
            self.prices,
            window=20
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)
        assert len(result.values) == len(self.prices)

        # Check first values are NaN (until window is filled)
        assert pd.isna(result.values.iloc[18])  # Index 18 should be NaN
        assert not pd.isna(result.values.iloc[19])  # Index 19 should have value

        # Check metadata
        assert result.metadata["type"] == IndicatorType.TREND
        assert result.metadata["window"] == 20
        assert result.metadata["min_periods"] == 20

    def test_simple_moving_average_custom_min_periods(self):
        """Test SMA with custom min_periods."""
        result = TechnicalIndicators.simple_moving_average(
            self.prices,
            window=20,
            min_periods=10
        )

        # With min_periods=10, we should have values starting at index 9
        assert pd.isna(result.values.iloc[8])  # Index 8 should be NaN
        assert not pd.isna(result.values.iloc[9])  # Index 9 should have value
        assert result.metadata["min_periods"] == 10

    def test_simple_moving_average_insufficient_data(self):
        """Test SMA with insufficient data."""
        short_prices = pd.Series([1.0, 2.0, 3.0])

        with pytest.raises(IndicatorError) as exc_info:
            TechnicalIndicators.simple_moving_average(short_prices, window=10)

        assert "Insufficient data for SMA" in str(exc_info.value)

    def test_exponential_moving_average_basic(self):
        """Test exponential moving average calculation."""
        result = TechnicalIndicators.exponential_moving_average(
            self.prices,
            span=20
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)
        assert len(result.values) == len(self.prices)

        # EMA should give more weight to recent prices
        # Check that it's different from SMA
        sma_result = TechnicalIndicators.simple_moving_average(
            self.prices,
            window=20
        )

        # EMA and SMA should be different (but might be close)
        assert not result.values.equals(sma_result.values)

        # Check metadata
        assert result.metadata["type"] == IndicatorType.TREND
        assert result.metadata["span"] == 20
        assert result.metadata["adjust"] is True

    def test_exponential_moving_average_no_adjust(self):
        """Test EMA with adjust=False."""
        result = TechnicalIndicators.exponential_moving_average(
            self.prices,
            span=20,
            adjust=False
        )

        assert result.metadata["adjust"] is False

    def test_moving_average_convergence_divergence_basic(self):
        """Test MACD calculation."""
        result = TechnicalIndicators.moving_average_convergence_divergence(
            self.prices
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.DataFrame)

        # Should contain MACD, signal, and histogram columns
        assert "macd" in result.values.columns
        assert "signal" in result.values.columns
        assert "histogram" in result.values.columns

        # Check metadata
        assert result.metadata["type"] == IndicatorType.OSCILLATOR
        assert "fast_period" in result.metadata
        assert "slow_period" in result.metadata
        assert "signal_period" in result.metadata

    def test_moving_average_convergence_divergence_custom_periods(self):
        """Test MACD with custom periods."""
        result = TechnicalIndicators.moving_average_convergence_divergence(
            self.prices,
            fast_period=12,
            slow_period=26,
            signal_period=9
        )

        assert result.metadata["fast_period"] == 12
        assert result.metadata["slow_period"] == 26
        assert result.metadata["signal_period"] == 9

    def test_relative_strength_index_basic(self):
        """Test RSI calculation."""
        result = TechnicalIndicators.relative_strength_index(
            self.prices,
            period=14
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)

        # RSI should be between 0 and 100
        assert result.values.min() >= 0
        assert result.values.max() <= 100

        # Check metadata
        assert result.metadata["type"] == IndicatorType.OSCILLATOR
        assert result.metadata["period"] == 14

    def test_relative_strength_index_custom_period(self):
        """Test RSI with custom period."""
        result = TechnicalIndicators.relative_strength_index(
            self.prices,
            period=20
        )

        assert result.metadata["period"] == 20

    def test_bollinger_bands_basic(self):
        """Test Bollinger Bands calculation."""
        result = TechnicalIndicators.bollinger_bands(
            self.prices,
            window=20,
            num_std=2.0
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.DataFrame)

        # Should contain upper, middle, lower bands
        assert "upper" in result.values.columns
        assert "middle" in result.values.columns
        assert "lower" in result.values.columns

        # Upper band should be above middle band
        assert (result.values["upper"] > result.values["middle"]).all()
        # Lower band should be below middle band
        assert (result.values["lower"] < result.values["middle"]).all()

        # Check metadata
        assert result.metadata["type"] == IndicatorType.VOLATILITY
        assert result.metadata["window"] == 20
        assert result.metadata["num_std"] == 2.0

    def test_average_true_range_basic(self):
        """Test Average True Range calculation."""
        result = TechnicalIndicators.average_true_range(
            high=self.high,
            low=self.low,
            close=self.prices,
            period=14
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)

        # ATR should be non-negative
        assert (result.values >= 0).all()

        # Check metadata
        assert result.metadata["type"] == IndicatorType.VOLATILITY
        assert result.metadata["period"] == 14

    def test_on_balance_volume_basic(self):
        """Test On-Balance Volume calculation."""
        result = TechnicalIndicators.on_balance_volume(
            close=self.prices,
            volume=self.volume
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)

        # OBV should be cumulative
        # (Not testing exact values since they depend on price changes)

        # Check metadata
        assert result.metadata["type"] == IndicatorType.VOLUME

    def test_stochastic_oscillator_basic(self):
        """Test Stochastic Oscillator calculation."""
        result = TechnicalIndicators.stochastic_oscillator(
            high=self.high,
            low=self.low,
            close=self.prices,
            k_period=14,
            d_period=3
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.DataFrame)

        # Should contain %K and %D columns
        assert "%K" in result.values.columns
        assert "%D" in result.values.columns

        # %K and %D should be between 0 and 100
        assert (result.values["%K"] >= 0).all() and (result.values["%K"] <= 100).all()
        assert (result.values["%D"] >= 0).all() and (result.values["%D"] <= 100).all()

        # Check metadata
        assert result.metadata["type"] == IndicatorType.OSCILLATOR
        assert result.metadata["k_period"] == 14
        assert result.metadata["d_period"] == 3

    def test_commodity_channel_index_basic(self):
        """Test Commodity Channel Index calculation."""
        result = TechnicalIndicators.commodity_channel_index(
            high=self.high,
            low=self.low,
            close=self.prices,
            period=20
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)

        # CCI can be positive or negative
        # Just check it's a Series
        assert len(result.values) == len(self.prices)

        # Check metadata
        assert result.metadata["type"] == IndicatorType.OSCILLATOR
        assert result.metadata["period"] == 20

    def test_average_directional_index_basic(self):
        """Test Average Directional Index calculation."""
        result = TechnicalIndicators.average_directional_index(
            high=self.high,
            low=self.low,
            close=self.prices,
            period=14
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.DataFrame)

        # Should contain ADX, +DI, -DI columns
        assert "adx" in result.values.columns
        assert "+di" in result.values.columns
        assert "-di" in result.values.columns

        # ADX should be between 0 and 100
        assert (result.values["adx"] >= 0).all() and (result.values["adx"] <= 100).all()

        # Check metadata
        assert result.metadata["type"] == IndicatorType.TREND
        assert result.metadata["period"] == 14

    def test_parabolic_sar_basic(self):
        """Test Parabolic SAR calculation."""
        result = TechnicalIndicators.parabolic_sar(
            high=self.high,
            low=self.low,
            acceleration=0.02,
            maximum=0.2
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)

        # SAR should be between high and low
        # (Might not be true for all points due to acceleration)

        # Check metadata
        assert result.metadata["type"] == IndicatorType.TREND
        assert result.metadata["acceleration"] == 0.02
        assert result.metadata["maximum"] == 0.2

    def test_ichimoku_cloud_basic(self):
        """Test Ichimoku Cloud calculation."""
        result = TechnicalIndicators.ichimoku_cloud(
            high=self.high,
            low=self.low,
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.DataFrame)

        # Should contain multiple Ichimoku lines
        expected_columns = [
            "tenkan_sen", "kijun_sen", "senkou_span_a",
            "senkou_span_b", "chikou_span"
        ]
        for col in expected_columns:
            assert col in result.values.columns

        # Check metadata
        assert result.metadata["type"] == IndicatorType.TREND
        assert "tenkan_period" in result.metadata

    def test_money_flow_index_basic(self):
        """Test Money Flow Index calculation."""
        result = TechnicalIndicators.money_flow_index(
            high=self.high,
            low=self.low,
            close=self.prices,
            volume=self.volume,
            period=14
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)

        # MFI should be between 0 and 100
        assert (result.values >= 0).all() and (result.values <= 100).all()

        # Check metadata
        assert result.metadata["type"] == IndicatorType.VOLUME
        assert result.metadata["period"] == 14

    def test_standard_deviation_basic(self):
        """Test Standard Deviation calculation."""
        result = TechnicalIndicators.standard_deviation(
            self.prices,
            window=20
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)

        # Standard deviation should be non-negative
        assert (result.values >= 0).all()

        # Check metadata
        assert result.metadata["type"] == IndicatorType.VOLATILITY
        assert result.metadata["window"] == 20

    def test_rate_of_change_basic(self):
        """Test Rate of Change calculation."""
        result = TechnicalIndicators.rate_of_change(
            self.prices,
            period=12
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)

        # ROC can be positive or negative
        assert len(result.values) == len(self.prices)

        # Check metadata
        assert result.metadata["type"] == IndicatorType.MOMENTUM
        assert result.metadata["period"] == 12

    def test_williams_percent_r_basic(self):
        """Test Williams %R calculation."""
        result = TechnicalIndicators.williams_percent_r(
            high=self.high,
            low=self.low,
            close=self.prices,
            period=14
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.Series)

        # Williams %R should be between -100 and 0
        assert (result.values >= -100).all() and (result.values <= 0).all()

        # Check metadata
        assert result.metadata["type"] == IndicatorType.OSCILLATOR
        assert result.metadata["period"] == 14

    def test_price_channels_basic(self):
        """Test Price Channels calculation."""
        result = TechnicalIndicators.price_channels(
            high=self.high,
            low=self.low,
            window=20
        )

        assert isinstance(result, IndicatorResult)
        assert isinstance(result.values, pd.DataFrame)

        # Should contain upper and lower channels
        assert "upper" in result.values.columns
        assert "lower" in result.values.columns

        # Upper channel should be >= lower channel
        assert (result.values["upper"] >= result.values["lower"]).all()

        # Check metadata
        assert result.metadata["type"] == IndicatorType.VOLATILITY
        assert result.metadata["window"] == 20

    def test_validate_series_parameters_valid(self):
        """Test validate_series_parameters with valid input."""
        # Should not raise exception
        TechnicalIndicators._validate_series_parameters(
            series=self.prices,
            min_length=10,
            allow_nan=False,
            allow_negative=False
        )

    def test_validate_series_parameters_too_short(self):
        """Test validate_series_parameters with too short series."""
        short_series = pd.Series([1.0, 2.0, 3.0])

        with pytest.raises(ValidationError) as exc_info:
            TechnicalIndicators._validate_series_parameters(
                series=short_series,
                min_length=10
            )

        assert "数据长度不足" in str(exc_info.value)

    def test_validate_series_parameters_with_nan(self):
        """Test validate_series_parameters with NaN values."""
        series_with_nan = self.prices.copy()
        series_with_nan.iloc[0] = np.nan

        with pytest.raises(ValidationError) as exc_info:
            TechnicalIndicators._validate_series_parameters(
                series=series_with_nan,
                allow_nan=False
            )

        assert "包含 NaN 值" in str(exc_info.value)

    def test_validate_series_parameters_with_negative(self):
        """Test validate_series_parameters with negative values."""
        series_with_negative = pd.Series([-1.0, 2.0, 3.0])

        with pytest.raises(ValidationError) as exc_info:
            TechnicalIndicators._validate_series_parameters(
                series=series_with_negative,
                allow_negative=False
            )

        assert "包含负值" in str(exc_info.value)

    def test_generate_signals_crossover(self):
        """Test generate_signals with crossover logic."""
        # Create two series that cross over
        series1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # Rising
        series2 = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])  # Falling

        # They cross at index 2 (both are 3.0)

        signals = TechnicalIndicators._generate_signals(
            series1=series1,
            series2=series2,
            signal_type="crossover"
        )

        # Should be a Series with signals
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(series1)

        # At crossover point (index 2), we might get a signal
        # Implementation dependent

    def test_generate_signals_threshold(self):
        """Test generate_signals with threshold logic."""
        series = pd.Series([0.3, 0.5, 0.7, 0.9, 0.2])

        signals = TechnicalIndicators._generate_signals(
            series=series,
            upper_threshold=0.8,
            lower_threshold=0.2,
            signal_type="threshold"
        )

        # Should generate signals based on thresholds
        # Values above 0.8: buy signal (1)
        # Values below 0.2: sell signal (-1)
        # Others: neutral (0)

        # Implementation dependent, but should have same length
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(series)
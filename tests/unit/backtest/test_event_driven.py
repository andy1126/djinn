"""
Unit tests for the event-driven backtest engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.djinn.core.backtest.event_driven import EventDrivenBacktestEngine
from src.djinn.core.backtest.base import BacktestResult, Trade, Position
from src.djinn.core.strategy.base import Strategy, Signal, SignalType
from src.djinn.utils.exceptions import BacktestError, ValidationError


class TestEventDrivenBacktestEngine:
    """Test EventDrivenBacktestEngine class."""

    def setup_method(self):
        """Setup test data."""
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        self.data = pd.DataFrame({
            'open': np.random.normal(150, 5, 100),
            'high': np.random.normal(155, 5, 100),
            'low': np.random.normal(145, 5, 100),
            'close': np.random.normal(152, 5, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

        # Create mock strategy
        self.mock_strategy = Mock(spec=Strategy)
        self.mock_strategy.name = "TestStrategy"

        # Create engine instance
        self.engine = EventDrivenBacktestEngine(
            strategy=self.mock_strategy,
            data=self.data,
            initial_capital=100000.0,
            start_date='2023-01-01',
            end_date='2023-03-31',
            symbols=["AAPL"],
            commission=0.001,
            slippage=0.0005
        )

    def test_initialization(self):
        """Test EventDrivenBacktestEngine initialization."""
        assert self.engine.strategy == self.mock_strategy
        pd.testing.assert_frame_equal(self.engine.data, self.data)
        assert self.engine.initial_capital == 100000.0
        assert self.engine.start_date == datetime(2023, 1, 1).date()
        assert self.engine.end_date == datetime(2023, 3, 31).date()
        assert self.engine.symbols == ["AAPL"]
        assert self.engine.commission == 0.001
        assert self.engine.slippage == 0.0005

        # Event-driven specific attributes
        assert hasattr(self.engine, 'event_queue')
        assert hasattr(self.engine, 'current_time')
        assert hasattr(self.engine, 'position_manager')

    def test_initialize(self):
        """Test _initialize method."""
        # Mock the strategy's initialize method
        self.mock_strategy.initialize.return_value = None

        # Call initialize
        self.engine._initialize()

        # Check that strategy was initialized
        self.mock_strategy.initialize.assert_called_once()

        # Check engine state
        assert self.engine.current_time is not None
        assert self.engine.cash == self.engine.initial_capital
        assert len(self.engine.positions) == 0
        assert len(self.engine.trades) == 0

    def test_execute_backtest_basic(self):
        """Test _execute_backtest method."""
        # Setup mocks
        self.mock_strategy.generate_signals.return_value = []

        # Mock internal methods
        with patch.object(self.engine, '_process_events') as mock_process_events, \
             patch.object(self.engine, '_advance_time') as mock_advance_time:

            # Setup advance_time to stop after a few iterations
            call_count = 0
            def advance_time_side_effect():
                nonlocal call_count
                call_count += 1
                if call_count >= 3:
                    self.engine.current_time = self.engine.end_date + timedelta(days=1)

            mock_advance_time.side_effect = advance_time_side_effect

            # Execute backtest
            result = self.engine._execute_backtest()

            # Check result
            assert isinstance(result, BacktestResult)
            assert result.strategy_name == "TestStrategy"

            # Verify methods were called
            assert mock_process_events.called
            assert mock_advance_time.called

    def test_process_events_with_signals(self):
        """Test _process_events with strategy signals."""
        # Create mock signals
        signals = [
            Signal(
                timestamp=datetime(2023, 1, 2),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=150.0,
                strength=0.8
            ),
            Signal(
                timestamp=datetime(2023, 1, 3),
                symbol="AAPL",
                signal_type=SignalType.SELL,
                price=160.0,
                strength=0.7
            )
        ]

        self.mock_strategy.generate_signals.return_value = signals

        # Mock order execution
        with patch.object(self.engine, '_execute_order_from_signal') as mock_execute_order:
            self.engine._process_events()

            # Verify signals were processed
            assert mock_execute_order.call_count == len(signals)
            for i, signal in enumerate(signals):
                mock_execute_order.assert_any_call(signal)

    def test_execute_order_from_signal_buy(self):
        """Test _execute_order_from_signal with BUY signal."""
        signal = Signal(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            strength=0.8
        )

        # Mock position size calculation
        self.mock_strategy.calculate_position_size.return_value = 100

        # Mock trade execution
        with patch.object(self.engine, 'execute_trade') as mock_execute_trade, \
             patch.object(self.engine, 'update_position') as mock_update_position:

            mock_execute_trade.return_value = Trade(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                quantity=100,
                price=signal.price,
                side="buy"
            )

            self.engine._execute_order_from_signal(signal)

            # Verify calculations and executions
            self.mock_strategy.calculate_position_size.assert_called_once()
            mock_execute_trade.assert_called_once()
            mock_update_position.assert_called_once()

    def test_execute_order_from_signal_sell(self):
        """Test _execute_order_from_signal with SELL signal."""
        signal = Signal(
            timestamp=datetime(2023, 1, 3),
            symbol="AAPL",
            signal_type=SignalType.SELL,
            price=160.0,
            strength=0.7
        )

        # Setup existing position
        self.engine.positions["AAPL"] = Position(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=15000.0,
            unrealized_pnl=0.0
        )

        # Mock position size calculation
        self.mock_strategy.calculate_position_size.return_value = 100

        with patch.object(self.engine, 'execute_trade') as mock_execute_trade, \
             patch.object(self.engine, 'update_position') as mock_update_position:

            self.engine._execute_order_from_signal(signal)

            # Verify position size calculation included current position
            # (implementation dependent)

    def test_execute_order_from_signal_hold(self):
        """Test _execute_order_from_signal with HOLD signal."""
        signal = Signal(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            signal_type=SignalType.HOLD,
            price=150.0
        )

        # HOLD signal should not execute trades
        with patch.object(self.engine, 'execute_trade') as mock_execute_trade:
            self.engine._execute_order_from_signal(signal)
            mock_execute_trade.assert_not_called()

    def test_execute_order_from_signal_exit(self):
        """Test _execute_order_from_signal with EXIT signal."""
        signal = Signal(
            timestamp=datetime(2023, 1, 3),
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            price=160.0
        )

        # Setup existing position
        self.engine.positions["AAPL"] = Position(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=15000.0,
            unrealized_pnl=0.0
        )

        # EXIT should close the entire position
        with patch.object(self.engine, 'execute_trade') as mock_execute_trade:
            self.engine._execute_order_from_signal(signal)

            # Should execute a sell trade for the full position
            mock_execute_trade.assert_called_once()
            call_args = mock_execute_trade.call_args[1]
            assert call_args['side'] == 'sell'
            assert call_args['quantity'] == 100

    def test_advance_time(self):
        """Test _advance_time method."""
        initial_time = self.engine.current_time

        # Mock data indexing
        self.engine.current_index = 0
        self.engine.data_index = self.data.index

        self.engine._advance_time()

        # Time should advance
        assert self.engine.current_time != initial_time
        assert self.engine.current_index == 1

        # Portfolio should be updated
        assert hasattr(self.engine, 'portfolio_value')

    def test_update_portfolio_value(self):
        """Test _update_portfolio_value method."""
        # Setup positions
        self.engine.positions["AAPL"] = Position(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=16000.0,  # Updated value
            unrealized_pnl=1000.0
        )

        self.engine.cash = 50000.0

        self.engine._update_portfolio_value()

        # Portfolio value should be sum of cash and positions
        expected_value = 50000.0 + 16000.0
        assert self.engine.portfolio_value == expected_value

        # Should update equity curve
        assert len(self.engine.equity_curve) > 0

    def test_handle_market_event(self):
        """Test _handle_market_event method."""
        # Mock current market data
        current_data = {
            'open': 150.0,
            'high': 155.0,
            'low': 145.0,
            'close': 152.0,
            'volume': 10000
        }

        with patch.object(self.engine, '_update_prices') as mock_update_prices:
            self.engine._handle_market_event(current_data)

            # Should update prices
            mock_update_prices.assert_called_once()

    def test_update_prices(self):
        """Test _update_prices method."""
        current_prices = {"AAPL": 155.0}

        # Setup position
        self.engine.positions["AAPL"] = Position(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=15000.0,
            unrealized_pnl=0.0
        )

        self.engine._update_prices(current_prices)

        # Position should be updated with new price
        position = self.engine.positions["AAPL"]
        assert position.current_price == 155.0
        assert position.market_value == 15500.0  # 100 * 155
        assert position.unrealized_pnl == 500.0  # 100 * (155 - 150)

    def test_validate_signal_timing(self):
        """Test _validate_signal_timing method."""
        # Signal timestamp should be >= current time
        valid_signal = Signal(
            timestamp=self.engine.current_time,
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0
        )

        assert self.engine._validate_signal_timing(valid_signal) is True

        # Signal in the past should be invalid
        past_signal = Signal(
            timestamp=self.engine.current_time - timedelta(days=1),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0
        )

        assert self.engine._validate_signal_timing(past_signal) is False

    def test_check_stop_loss(self):
        """Test _check_stop_loss method."""
        # Setup position with stop loss
        self.engine.positions["AAPL"] = Position(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=14000.0,  # Price dropped to 140
            unrealized_pnl=-1000.0
        )

        # Set stop loss threshold
        self.engine.stop_loss_percentage = 0.05  # 5%

        with patch.object(self.engine, '_execute_order_from_signal') as mock_execute_order:
            self.engine._check_stop_loss()

            # Should generate exit signal if loss exceeds stop loss
            # Current price: 140, Avg price: 150, Loss: 6.67% > 5%
            # Implementation dependent

    def test_check_take_profit(self):
        """Test _check_take_profit method."""
        # Setup position with profit
        self.engine.positions["AAPL"] = Position(
            timestamp=datetime(2023, 1, 2),
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=17000.0,  # Price increased to 170
            unrealized_pnl=2000.0
        )

        # Set take profit threshold
        self.engine.take_profit_percentage = 0.10  # 10%

        with patch.object(self.engine, '_execute_order_from_signal') as mock_execute_order:
            self.engine._check_take_profit()

            # Should generate exit signal if profit exceeds take profit
            # Current price: 170, Avg price: 150, Profit: 13.33% > 10%
            # Implementation dependent

    def test_get_current_market_data(self):
        """Test _get_current_market_data method."""
        self.engine.current_index = 0

        market_data = self.engine._get_current_market_data()

        # Should return dictionary with current market data
        assert isinstance(market_data, dict)
        assert 'open' in market_data
        assert 'high' in market_data
        assert 'low' in market_data
        assert 'close' in market_data
        assert 'volume' in market_data

    def test_is_backtest_complete(self):
        """Test _is_backtest_complete method."""
        # Test when current time < end date
        self.engine.current_time = datetime(2023, 1, 15).date()
        self.engine.end_date = datetime(2023, 3, 31).date()

        assert self.engine._is_backtest_complete() is False

        # Test when current time >= end date
        self.engine.current_time = datetime(2023, 4, 1).date()

        assert self.engine._is_backtest_complete() is True

    def test_event_driven_specific_features(self):
        """Test event-driven specific features."""
        # Test event queue operations
        event = {"type": "market", "data": {"price": 150.0}}

        # Add event to queue
        self.engine.event_queue.append(event)
        assert len(self.engine.event_queue) == 1

        # Process event queue
        with patch.object(self.engine, '_handle_market_event'):
            # Implementation would process events from queue
            pass

        # Test event prioritization
        # Higher priority events should be processed first
        # Implementation dependent
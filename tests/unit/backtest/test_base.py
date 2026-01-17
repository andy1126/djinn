"""
Unit tests for the backtest base module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.djinn.core.backtest.base import (
    BacktestMode,
    Trade,
    Position,
    BacktestResult,
    BacktestEngine,
)
from src.djinn.utils.exceptions import BacktestError, ValidationError
from src.djinn.core.strategy.base import Strategy, Signal, SignalType
from src.djinn.core.portfolio.base import Portfolio


class TestBacktestMode:
    """Test BacktestMode enumeration."""

    def test_backtest_mode_values(self):
        """Test BacktestMode enum values."""
        assert BacktestMode.EVENT_DRIVEN == "event_driven"
        assert BacktestMode.VECTORIZED == "vectorized"
        assert BacktestMode.HYBRID == "hybrid"


class TestTrade:
    """Test Trade data class."""

    def test_trade_creation(self):
        """Test Trade creation."""
        timestamp = datetime(2023, 12, 31, 14, 30, 45)
        trade = Trade(
            timestamp=timestamp,
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            side="buy",
            commission=1.0,
            slippage=0.5,
            trade_id="TRADE_001",
            strategy_name="TestStrategy"
        )

        assert trade.timestamp == timestamp
        assert trade.symbol == "AAPL"
        assert trade.quantity == 100.0
        assert trade.price == 150.0
        assert trade.side == "buy"
        assert trade.commission == 1.0
        assert trade.slippage == 0.5
        assert trade.trade_id == "TRADE_001"
        assert trade.strategy_name == "TestStrategy"

    def test_trade_creation_minimal(self):
        """Test Trade creation with minimal parameters."""
        timestamp = datetime.now()
        trade = Trade(
            timestamp=timestamp,
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            side="buy"
        )

        assert trade.timestamp == timestamp
        assert trade.symbol == "AAPL"
        assert trade.quantity == 100.0
        assert trade.price == 150.0
        assert trade.side == "buy"
        assert trade.commission == 0.0  # Default
        assert trade.slippage == 0.0    # Default
        assert trade.trade_id is None   # Default
        assert trade.strategy_name is None  # Default


class TestPosition:
    """Test Position data class."""

    def test_position_creation(self):
        """Test Position creation."""
        timestamp = datetime(2023, 12, 31, 16, 0, 0)
        position = Position(
            timestamp=timestamp,
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            market_value=16000.0,
            unrealized_pnl=1000.0,
            realized_pnl=500.0
        )

        assert position.timestamp == timestamp
        assert position.symbol == "AAPL"
        assert position.quantity == 100.0
        assert position.avg_price == 150.0
        assert position.market_value == 16000.0
        assert position.unrealized_pnl == 1000.0
        assert position.realized_pnl == 500.0

    def test_position_creation_minimal(self):
        """Test Position creation with minimal parameters."""
        timestamp = datetime.now()
        position = Position(
            timestamp=timestamp,
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            market_value=16000.0,
            unrealized_pnl=1000.0
        )

        assert position.timestamp == timestamp
        assert position.symbol == "AAPL"
        assert position.quantity == 100.0
        assert position.avg_price == 150.0
        assert position.market_value == 16000.0
        assert position.unrealized_pnl == 1000.0
        assert position.realized_pnl == 0.0  # Default


class TestBacktestResult:
    """Test BacktestResult data class."""

    def test_backtest_result_creation(self):
        """Test BacktestResult creation."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        equity_curve = pd.Series(np.linspace(100000, 110000, 100), index=dates)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        drawdown = pd.Series(np.random.uniform(-0.1, 0, 100), index=dates)

        # Create sample trades and positions
        trades = [
            Trade(
                timestamp=datetime(2023, 1, 15),
                symbol="AAPL",
                quantity=100.0,
                price=150.0,
                side="buy"
            )
        ]

        positions = [
            Position(
                timestamp=datetime(2023, 1, 15),
                symbol="AAPL",
                quantity=100.0,
                avg_price=150.0,
                market_value=16000.0,
                unrealized_pnl=1000.0
            )
        ]

        result = BacktestResult(
            # Performance metrics
            total_return=0.1,
            annual_return=0.12,
            sharpe_ratio=1.5,
            max_drawdown=-0.15,
            volatility=0.2,
            sortino_ratio=2.0,
            calmar_ratio=0.8,
            win_rate=0.6,
            profit_factor=1.8,
            avg_win=0.02,
            avg_loss=-0.01,

            # Trade statistics
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            avg_trade_return=0.005,
            avg_trade_duration=pd.Timedelta(days=5),
            max_consecutive_wins=8,
            max_consecutive_losses=5,

            # Portfolio statistics
            initial_capital=100000.0,
            final_capital=110000.0,
            peak_capital=115000.0,
            trough_capital=95000.0,
            total_commission=100.0,
            total_slippage=50.0,

            # Time series data
            equity_curve=equity_curve,
            returns=returns,
            drawdown=drawdown,
            positions=positions,
            trades=trades,

            # Additional metadata
            strategy_name="TestStrategy",
            symbols=["AAPL", "GOOGL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            parameters={"param1": "value1"},
            benchmark_returns=returns * 0.8,  # Benchmark underperforms
            risk_free_rate=0.02
        )

        # Test performance metrics
        assert result.total_return == 0.1
        assert result.annual_return == 0.12
        assert result.sharpe_ratio == 1.5
        assert result.max_drawdown == -0.15

        # Test trade statistics
        assert result.total_trades == 100
        assert result.winning_trades == 60
        assert result.losing_trades == 40
        assert result.avg_trade_return == 0.005

        # Test portfolio statistics
        assert result.initial_capital == 100000.0
        assert result.final_capital == 110000.0
        assert result.peak_capital == 115000.0
        assert result.trough_capital == 95000.0

        # Test time series data
        pd.testing.assert_series_equal(result.equity_curve, equity_curve)
        pd.testing.assert_series_equal(result.returns, returns)
        pd.testing.assert_series_equal(result.drawdown, drawdown)
        assert result.positions == positions
        assert result.trades == trades

        # Test metadata
        assert result.strategy_name == "TestStrategy"
        assert result.symbols == ["AAPL", "GOOGL"]
        assert result.start_date == datetime(2023, 1, 1)
        assert result.end_date == datetime(2023, 12, 31)

    def test_backtest_result_to_dataframe(self):
        """Test BacktestResult.to_dataframe method."""
        # Create minimal result
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        equity_curve = pd.Series(range(100000, 100010), index=dates)

        result = BacktestResult(
            total_return=0.1,
            annual_return=0.12,
            sharpe_ratio=1.5,
            max_drawdown=-0.15,
            volatility=0.2,
            sortino_ratio=2.0,
            calmar_ratio=0.8,
            win_rate=0.6,
            profit_factor=1.8,
            avg_win=0.02,
            avg_loss=-0.01,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            avg_trade_return=0.005,
            avg_trade_duration=pd.Timedelta(days=5),
            max_consecutive_wins=8,
            max_consecutive_losses=5,
            initial_capital=100000.0,
            final_capital=110000.0,
            peak_capital=115000.0,
            trough_capital=95000.0,
            total_commission=100.0,
            total_slippage=50.0,
            equity_curve=equity_curve,
            returns=pd.Series(np.zeros(10), index=dates),
            drawdown=pd.Series(np.zeros(10), index=dates),
            positions=[],
            trades=[],
            strategy_name="TestStrategy",
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10)
        )

        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Should contain key metrics
        assert "total_return" in df.columns
        assert "sharpe_ratio" in df.columns
        assert "max_drawdown" in df.columns

    def test_backtest_result_get_summary(self):
        """Test BacktestResult.get_summary method."""
        # Create minimal result
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        equity_curve = pd.Series(range(100000, 100010), index=dates)

        result = BacktestResult(
            total_return=0.1,
            annual_return=0.12,
            sharpe_ratio=1.5,
            max_drawdown=-0.15,
            volatility=0.2,
            sortino_ratio=2.0,
            calmar_ratio=0.8,
            win_rate=0.6,
            profit_factor=1.8,
            avg_win=0.02,
            avg_loss=-0.01,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            avg_trade_return=0.005,
            avg_trade_duration=pd.Timedelta(days=5),
            max_consecutive_wins=8,
            max_consecutive_losses=5,
            initial_capital=100000.0,
            final_capital=110000.0,
            peak_capital=115000.0,
            trough_capital=95000.0,
            total_commission=100.0,
            total_slippage=50.0,
            equity_curve=equity_curve,
            returns=pd.Series(np.zeros(10), index=dates),
            drawdown=pd.Series(np.zeros(10), index=dates),
            positions=[],
            trades=[],
            strategy_name="TestStrategy",
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10)
        )

        summary = result.get_summary()

        assert isinstance(summary, dict)
        # Should contain key metrics grouped by category
        assert "performance" in summary
        assert "trades" in summary
        assert "portfolio" in summary
        assert "metadata" in summary


class TestConcreteBacktestEngine(BacktestEngine):
    """Concrete backtest engine implementation for testing."""

    def _initialize(self):
        """Test implementation of _initialize."""
        self.initialized = True

    def _execute_backtest(self):
        """Test implementation of _execute_backtest."""
        # Create a simple result
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        equity_curve = pd.Series(
            np.linspace(self.initial_capital, self.initial_capital * 1.1, len(dates)),
            index=dates
        )

        return BacktestResult(
            total_return=0.1,
            annual_return=0.12,
            sharpe_ratio=1.5,
            max_drawdown=-0.15,
            volatility=0.2,
            sortino_ratio=2.0,
            calmar_ratio=0.8,
            win_rate=0.6,
            profit_factor=1.8,
            avg_win=0.02,
            avg_loss=-0.01,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            avg_trade_return=0.005,
            avg_trade_duration=pd.Timedelta(days=5),
            max_consecutive_wins=3,
            max_consecutive_losses=2,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital * 1.1,
            peak_capital=self.initial_capital * 1.15,
            trough_capital=self.initial_capital * 0.95,
            total_commission=10.0,
            total_slippage=5.0,
            equity_curve=equity_curve,
            returns=pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates),
            drawdown=pd.Series(np.random.uniform(-0.1, 0, len(dates)), index=dates),
            positions=[],
            trades=[],
            strategy_name=self.strategy.name if self.strategy else "TestStrategy",
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )


class TestBacktestEngineBaseClass:
    """Test BacktestEngine abstract base class."""

    def test_backtest_engine_initialization(self):
        """Test BacktestEngine initialization."""
        # Create mock strategy and data
        mock_strategy = Mock(spec=Strategy)
        mock_strategy.name = "TestStrategy"

        data = pd.DataFrame({
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))

        engine = TestConcreteBacktestEngine(
            strategy=mock_strategy,
            data=data,
            initial_capital=50000.0,
            start_date='2023-01-01',
            end_date='2023-03-31',
            symbols=["AAPL"],
            commission=0.001,
            slippage=0.0005,
            benchmark_symbol="SPY"
        )

        assert engine.strategy == mock_strategy
        pd.testing.assert_frame_equal(engine.data, data)
        assert engine.initial_capital == 50000.0
        assert engine.start_date == datetime(2023, 1, 1).date()
        assert engine.end_date == datetime(2023, 3, 31).date()
        assert engine.symbols == ["AAPL"]
        assert engine.commission == 0.001
        assert engine.slippage == 0.0005
        assert engine.benchmark_symbol == "SPY"
        assert engine.mode == BacktestMode.EVENT_DRIVEN  # Default
        assert isinstance(engine.trades, list)
        assert isinstance(engine.positions, list)
        assert isinstance(engine.equity_curve, pd.Series)

    def test_backtest_engine_default_initialization(self):
        """Test BacktestEngine initialization with defaults."""
        mock_strategy = Mock(spec=Strategy)
        data = pd.DataFrame({'close': [1, 2, 3]})

        engine = TestConcreteBacktestEngine(
            strategy=mock_strategy,
            data=data
        )

        assert engine.initial_capital == 100000.0  # Default
        assert engine.commission == 0.001  # Default
        assert engine.slippage == 0.0005  # Default
        assert engine.benchmark_symbol is None  # Default

    def test_run_backtest(self):
        """Test run_backtest method."""
        mock_strategy = Mock(spec=Strategy)
        mock_strategy.name = "TestStrategy"

        data = pd.DataFrame({
            'close': np.random.randn(100)
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))

        engine = TestConcreteBacktestEngine(
            strategy=mock_strategy,
            data=data,
            start_date='2023-01-01',
            end_date='2023-03-31'
        )

        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert result.total_return == 0.1
        assert result.strategy_name == "TestStrategy"
        assert engine.initialized is True

    def test_calculate_commission(self):
        """Test calculate_commission method."""
        engine = TestConcreteBacktestEngine(
            strategy=Mock(spec=Strategy),
            data=pd.DataFrame({'close': [1, 2, 3]})
        )
        engine.commission = 0.001  # 0.1%

        # Test fixed commission
        commission = engine.calculate_commission(
            quantity=100.0,
            price=150.0,
            commission_type="fixed"
        )

        assert commission == 0.001  # Fixed amount

        # Test percentage commission
        commission = engine.calculate_commission(
            quantity=100.0,
            price=150.0,
            commission_type="percentage"
        )

        # Expected: 100 * 150 * 0.001 = 15.0
        assert commission == 15.0

        # Test per_share commission
        engine.commission = 0.01  # $0.01 per share
        commission = engine.calculate_commission(
            quantity=100.0,
            price=150.0,
            commission_type="per_share"
        )

        # Expected: 100 * 0.01 = 1.0
        assert commission == 1.0

    def test_calculate_slippage(self):
        """Test calculate_slippage method."""
        engine = TestConcreteBacktestEngine(
            strategy=Mock(spec=Strategy),
            data=pd.DataFrame({'close': [1, 2, 3]})
        )
        engine.slippage = 0.0005  # 0.05%

        slippage = engine.calculate_slippage(
            quantity=100.0,
            price=150.0,
            side="buy"
        )

        # Expected: 100 * 150 * 0.0005 = 7.5
        assert slippage == 7.5

    def test_execute_trade(self):
        """Test execute_trade method."""
        engine = TestConcreteBacktestEngine(
            strategy=Mock(spec=Strategy),
            data=pd.DataFrame({'close': [1, 2, 3]})
        )
        engine.commission = 0.001
        engine.slippage = 0.0005

        trade = engine.execute_trade(
            timestamp=datetime.now(),
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            side="buy",
            strategy_name="TestStrategy"
        )

        assert isinstance(trade, Trade)
        assert trade.symbol == "AAPL"
        assert trade.quantity == 100.0
        assert trade.price == 150.0
        assert trade.side == "buy"
        assert trade.strategy_name == "TestStrategy"
        assert trade.commission > 0
        assert trade.slippage > 0

        # Should be added to trades list
        assert len(engine.trades) == 1
        assert engine.trades[0] == trade

    def test_update_position(self):
        """Test update_position method."""
        engine = TestConcreteBacktestEngine(
            strategy=Mock(spec=Strategy),
            data=pd.DataFrame({'close': [1, 2, 3]})
        )

        timestamp = datetime.now()

        # First position
        position1 = engine.update_position(
            timestamp=timestamp,
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            side="buy"
        )

        assert isinstance(position1, Position)
        assert position1.symbol == "AAPL"
        assert position1.quantity == 100.0
        assert position1.avg_price == 150.0

        # Update position (buy more)
        position2 = engine.update_position(
            timestamp=timestamp + timedelta(days=1),
            symbol="AAPL",
            quantity=50.0,
            price=160.0,
            side="buy"
        )

        # Average price should be weighted average
        # (100*150 + 50*160) / 150 = (15000 + 8000) / 150 = 23000 / 150 = 153.33
        expected_avg_price = (100.0 * 150.0 + 50.0 * 160.0) / 150.0
        assert abs(position2.avg_price - expected_avg_price) < 0.01
        assert position2.quantity == 150.0

        # Should be added to positions list
        assert len(engine.positions) == 2

    def test_update_equity_curve(self):
        """Test update_equity_curve method."""
        engine = TestConcreteBacktestEngine(
            strategy=Mock(spec=Strategy),
            data=pd.DataFrame({'close': [1, 2, 3]})
        )
        engine.initial_capital = 100000.0

        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        ]

        portfolio_values = [100000.0, 101000.0, 102000.0]

        for date, value in zip(dates, portfolio_values):
            engine.update_equity_curve(date, value)

        # Check equity curve
        assert len(engine.equity_curve) == 3
        assert engine.equity_curve.iloc[0] == 100000.0
        assert engine.equity_curve.iloc[1] == 101000.0
        assert engine.equity_curve.iloc[2] == 102000.0

    def test_validate_parameters_valid(self):
        """Test validate_parameters with valid input."""
        engine = TestConcreteBacktestEngine(
            strategy=Mock(spec=Strategy),
            data=pd.DataFrame({'close': [1, 2, 3]})
        )

        # Should not raise exception
        engine.validate_parameters()

    def test_validate_parameters_invalid_dates(self):
        """Test validate_parameters with invalid dates."""
        mock_strategy = Mock(spec=Strategy)
        data = pd.DataFrame({
            'close': np.random.randn(100)
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))

        # Start date after end date
        with pytest.raises(ValidationError) as exc_info:
            TestConcreteBacktestEngine(
                strategy=mock_strategy,
                data=data,
                start_date='2023-12-31',
                end_date='2023-01-01'
            )

        assert "开始日期不能晚于结束日期" in str(exc_info.value)

    def test_backtest_engine_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Cannot instantiate abstract BacktestEngine directly
        with pytest.raises(TypeError):
            BacktestEngine(
                strategy=Mock(spec=Strategy),
                data=pd.DataFrame({'close': [1, 2, 3]})
            )
"""
Unit tests for the portfolio base module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock

from src.djinn.core.portfolio.base import (
    PortfolioStatus,
    RebalancingFrequency,
    Asset,
    PortfolioAllocation,
    PortfolioHolding,
    PortfolioSnapshot,
    Portfolio,
)
from src.djinn.utils.exceptions import PortfolioError, ValidationError


class TestPortfolioStatus:
    """Test PortfolioStatus enumeration."""

    def test_portfolio_status_values(self):
        """Test PortfolioStatus enum values."""
        assert PortfolioStatus.ACTIVE == "active"
        assert PortfolioStatus.CLOSED == "closed"
        assert PortfolioStatus.SUSPENDED == "suspended"


class TestRebalancingFrequency:
    """Test RebalancingFrequency enumeration."""

    def test_rebalancing_frequency_values(self):
        """Test RebalancingFrequency enum values."""
        assert RebalancingFrequency.DAILY == "daily"
        assert RebalancingFrequency.WEEKLY == "weekly"
        assert RebalancingFrequency.MONTHLY == "monthly"
        assert RebalancingFrequency.QUARTERLY == "quarterly"
        assert RebalancingFrequency.YEARLY == "yearly"
        assert RebalancingFrequency.NEVER == "never"


class TestAsset:
    """Test Asset data class."""

    def test_asset_creation(self):
        """Test Asset creation."""
        asset = Asset(
            symbol="AAPL",
            name="Apple Inc.",
            asset_type="stock",
            currency="USD",
            exchange="NASDAQ",
            sector="Technology",
            country="USA"
        )

        assert asset.symbol == "AAPL"
        assert asset.name == "Apple Inc."
        assert asset.asset_type == "stock"
        assert asset.currency == "USD"
        assert asset.exchange == "NASDAQ"
        assert asset.sector == "Technology"
        assert asset.country == "USA"

    def test_asset_creation_minimal(self):
        """Test Asset creation with minimal parameters."""
        asset = Asset(
            symbol="AAPL",
            name="Apple Inc.",
            asset_type="stock"
        )

        assert asset.symbol == "AAPL"
        assert asset.name == "Apple Inc."
        assert asset.asset_type == "stock"
        assert asset.currency == "USD"  # Default
        assert asset.exchange is None
        assert asset.sector is None
        assert asset.country is None


class TestPortfolioAllocation:
    """Test PortfolioAllocation data class."""

    def test_portfolio_allocation_creation(self):
        """Test PortfolioAllocation creation."""
        allocation = PortfolioAllocation(
            symbol="AAPL",
            target_weight=0.1,  # 10%
            min_weight=0.05,    # 5%
            max_weight=0.15,    # 15%
            is_core=True
        )

        assert allocation.symbol == "AAPL"
        assert allocation.target_weight == 0.1
        assert allocation.min_weight == 0.05
        assert allocation.max_weight == 0.15
        assert allocation.is_core is True

    def test_portfolio_allocation_defaults(self):
        """Test PortfolioAllocation with default values."""
        allocation = PortfolioAllocation(
            symbol="AAPL",
            target_weight=0.1
        )

        assert allocation.symbol == "AAPL"
        assert allocation.target_weight == 0.1
        assert allocation.min_weight == 0.0  # Default
        assert allocation.max_weight == 1.0  # Default
        assert allocation.is_core is True  # Default


class TestPortfolioHolding:
    """Test PortfolioHolding data class."""

    def test_portfolio_holding_creation(self):
        """Test PortfolioHolding creation."""
        entry_date = datetime(2023, 1, 15)
        holding = PortfolioHolding(
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            current_price=160.0,
            market_value=16000.0,  # 100 * 160
            cost_basis=15000.0,    # 100 * 150
            unrealized_pnl=1000.0, # 16000 - 15000
            realized_pnl=500.0,
            entry_date=entry_date
        )

        assert holding.symbol == "AAPL"
        assert holding.quantity == 100.0
        assert holding.avg_price == 150.0
        assert holding.current_price == 160.0
        assert holding.market_value == 16000.0
        assert holding.cost_basis == 15000.0
        assert holding.unrealized_pnl == 1000.0
        assert holding.realized_pnl == 500.0
        assert holding.entry_date == entry_date

    def test_portfolio_holding_calculated_values(self):
        """Test PortfolioHolding with calculated values."""
        # Create holding with basic values, let market_value and unrealized_pnl be calculated
        # In actual implementation, these might be calculated properties
        holding = PortfolioHolding(
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            current_price=160.0,
            market_value=100.0 * 160.0,  # Calculated
            cost_basis=100.0 * 150.0,    # Calculated
            unrealized_pnl=100.0 * (160.0 - 150.0)  # Calculated
        )

        assert holding.market_value == 16000.0
        assert holding.cost_basis == 15000.0
        assert holding.unrealized_pnl == 1000.0


class TestPortfolioSnapshot:
    """Test PortfolioSnapshot data class."""

    def test_portfolio_snapshot_creation(self):
        """Test PortfolioSnapshot creation."""
        timestamp = datetime(2023, 12, 31, 16, 0, 0)

        # Create sample holdings
        holdings = {
            "AAPL": PortfolioHolding(
                symbol="AAPL",
                quantity=100.0,
                avg_price=150.0,
                current_price=160.0,
                market_value=16000.0,
                cost_basis=15000.0,
                unrealized_pnl=1000.0
            ),
            "GOOGL": PortfolioHolding(
                symbol="GOOGL",
                quantity=50.0,
                avg_price=2800.0,
                current_price=2900.0,
                market_value=145000.0,
                cost_basis=140000.0,
                unrealized_pnl=5000.0
            )
        }

        allocations = {"AAPL": 0.1, "GOOGL": 0.9}
        performance = {"return": 0.05, "sharpe": 1.2}

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=161000.0,  # 16000 + 145000
            cash=5000.0,
            holdings=holdings,
            allocations=allocations,
            performance=performance
        )

        assert snapshot.timestamp == timestamp
        assert snapshot.total_value == 161000.0
        assert snapshot.cash == 5000.0
        assert snapshot.holdings == holdings
        assert snapshot.allocations == allocations
        assert snapshot.performance == performance


class TestConcretePortfolio(Portfolio):
    """Concrete portfolio implementation for testing abstract base class."""

    def initialize(self):
        """Test implementation of initialize."""
        self.initialized = True

    def get_value(self):
        """Test implementation of get_value."""
        return 100000.0

    def get_positions(self):
        """Test implementation of get_positions."""
        return {"AAPL": PortfolioHolding(
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            current_price=160.0,
            market_value=16000.0,
            cost_basis=15000.0,
            unrealized_pnl=1000.0
        )}

    def execute_order(self, symbol, quantity, order_type="market"):
        """Test implementation of execute_order."""
        return {
            "symbol": symbol,
            "quantity": quantity,
            "order_type": order_type,
            "executed": True
        }

    def rebalance(self):
        """Test implementation of rebalance."""
        self.rebalanced = True


class TestPortfolioBaseClass:
    """Test Portfolio abstract base class."""

    def test_portfolio_initialization(self):
        """Test Portfolio initialization."""
        portfolio = TestConcretePortfolio(
            initial_capital=50000.0,
            name="Test Portfolio",
            currency="EUR",
            benchmark_symbol="SPY",
            rebalancing_frequency=RebalancingFrequency.MONTHLY
        )

        assert portfolio.initial_capital == 50000.0
        assert portfolio.name == "Test Portfolio"
        assert portfolio.currency == "EUR"
        assert portfolio.benchmark_symbol == "SPY"
        assert portfolio.rebalancing_frequency == RebalancingFrequency.MONTHLY
        assert portfolio.status == PortfolioStatus.ACTIVE
        assert isinstance(portfolio.holdings, dict)
        assert isinstance(portfolio.allocations, dict)
        assert isinstance(portfolio.performance_metrics, dict)
        assert isinstance(portfolio.trade_history, list)

    def test_portfolio_default_initialization(self):
        """Test Portfolio initialization with defaults."""
        portfolio = TestConcretePortfolio()

        assert portfolio.initial_capital == 100000.0  # Default
        assert portfolio.name == "Default Portfolio"  # Default
        assert portfolio.currency == "USD"  # Default
        assert portfolio.benchmark_symbol is None  # Default
        assert portfolio.rebalancing_frequency == RebalancingFrequency.MONTHLY  # Default

    def test_update_position_buy(self):
        """Test update_position for buy order."""
        portfolio = TestConcretePortfolio(initial_capital=100000.0)

        # Initial state
        assert portfolio.cash == 100000.0
        assert len(portfolio.holdings) == 0

        # Buy 100 shares of AAPL at $150
        portfolio.update_position(
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            side="buy"
        )

        # Check cash reduced
        expected_cash = 100000.0 - (100.0 * 150.0)
        assert portfolio.cash == expected_cash

        # Check holding created
        assert "AAPL" in portfolio.holdings
        holding = portfolio.holdings["AAPL"]
        assert holding.quantity == 100.0
        assert holding.avg_price == 150.0
        assert holding.market_value == 15000.0  # 100 * 150
        assert holding.cost_basis == 15000.0

    def test_update_position_sell(self):
        """Test update_position for sell order."""
        portfolio = TestConcretePortfolio(initial_capital=100000.0)

        # First buy some shares
        portfolio.update_position(
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            side="buy"
        )

        initial_cash = portfolio.cash
        initial_holding = portfolio.holdings["AAPL"]

        # Sell 50 shares at $160
        portfolio.update_position(
            symbol="AAPL",
            quantity=50.0,
            price=160.0,
            side="sell"
        )

        # Check cash increased
        expected_cash = initial_cash + (50.0 * 160.0)
        assert portfolio.cash == expected_cash

        # Check holding updated
        holding = portfolio.holdings["AAPL"]
        assert holding.quantity == 50.0  # 100 - 50
        # Average price should remain the same for remaining shares
        assert holding.avg_price == 150.0
        assert holding.market_value == 50.0 * 160.0  # 8000

    def test_update_position_sell_more_than_owned(self):
        """Test update_position trying to sell more shares than owned."""
        portfolio = TestConcretePortfolio()

        # Buy 100 shares
        portfolio.update_position("AAPL", 100.0, 150.0, "buy")

        # Try to sell 150 shares (more than owned)
        with pytest.raises(PortfolioError) as exc_info:
            portfolio.update_position("AAPL", 150.0, 160.0, "sell")

        assert "Insufficient position" in str(exc_info.value)

    def test_calculate_portfolio_value(self):
        """Test calculate_portfolio_value."""
        portfolio = TestConcretePortfolio(initial_capital=100000.0)

        # Add some positions
        portfolio.update_position("AAPL", 100.0, 150.0, "buy")
        portfolio.update_position("GOOGL", 50.0, 2800.0, "buy")

        # Update prices
        portfolio.update_prices({
            "AAPL": 160.0,
            "GOOGL": 2900.0
        })

        # Calculate value
        total_value = portfolio.calculate_portfolio_value()

        # Expected:
        # AAPL: 100 * 160 = 16000
        # GOOGL: 50 * 2900 = 145000
        # Cash: 100000 - (100*150 + 50*2800) = 100000 - (15000 + 140000) = 100000 - 155000 = -55000
        # Total value should be sum of market values
        expected_value = 16000.0 + 145000.0  # = 161000
        # Note: Negative cash means we borrowed, but portfolio value is still positive

        assert total_value == expected_value

    def test_update_prices(self):
        """Test update_prices."""
        portfolio = TestConcretePortfolio()

        # Add position
        portfolio.update_position("AAPL", 100.0, 150.0, "buy")

        # Update price
        portfolio.update_prices({"AAPL": 160.0})

        # Check holding updated
        holding = portfolio.holdings["AAPL"]
        assert holding.current_price == 160.0
        assert holding.market_value == 100.0 * 160.0  # 16000
        assert holding.unrealized_pnl == 100.0 * (160.0 - 150.0)  # 1000

    def test_update_prices_missing_symbol(self):
        """Test update_prices with symbol not in portfolio."""
        portfolio = TestConcretePortfolio()

        # Update price for symbol not in portfolio (should be ignored)
        portfolio.update_prices({"AAPL": 160.0})

        # Should not raise error, just ignore

    def test_get_holdings_summary(self):
        """Test get_holdings_summary."""
        portfolio = TestConcretePortfolio()

        # Add positions
        portfolio.update_position("AAPL", 100.0, 150.0, "buy")
        portfolio.update_position("GOOGL", 50.0, 2800.0, "buy")

        summary = portfolio.get_holdings_summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2  # AAPL and GOOGL
        assert "symbol" in summary.columns
        assert "quantity" in summary.columns
        assert "market_value" in summary.columns

    def test_get_performance_metrics(self):
        """Test get_performance_metrics."""
        portfolio = TestConcretePortfolio(initial_capital=100000.0)

        # Add some performance history
        portfolio.performance_history = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
            "portfolio_value": np.linspace(100000, 110000, 10)
        })

        metrics = portfolio.get_performance_metrics()

        assert isinstance(metrics, dict)
        # Should contain common metrics
        assert "total_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics

    def test_record_trade(self):
        """Test record_trade."""
        portfolio = TestConcretePortfolio()

        # Record a trade
        trade_data = {
            "timestamp": datetime.now(),
            "symbol": "AAPL",
            "quantity": 100.0,
            "price": 150.0,
            "side": "buy",
            "commission": 1.0
        }

        portfolio.record_trade(**trade_data)

        # Check trade history
        assert len(portfolio.trade_history) == 1
        trade = portfolio.trade_history[0]
        assert trade["symbol"] == "AAPL"
        assert trade["quantity"] == 100.0

    def test_validate_order_valid(self):
        """Test validate_order with valid order."""
        portfolio = TestConcretePortfolio(initial_capital=100000.0)

        # Valid buy order
        is_valid, message = portfolio.validate_order(
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            side="buy"
        )

        assert is_valid is True
        assert message == ""

    def test_validate_order_insufficient_cash(self):
        """Test validate_order with insufficient cash."""
        portfolio = TestConcretePortfolio(initial_capital=1000.0)

        # Try to buy more than we can afford
        is_valid, message = portfolio.validate_order(
            symbol="AAPL",
            quantity=100.0,
            price=150.0,  # Total = 15000 > 1000
            side="buy"
        )

        assert is_valid is False
        assert "insufficient cash" in message.lower()

    def test_validate_order_invalid_side(self):
        """Test validate_order with invalid side."""
        portfolio = TestConcretePortfolio()

        is_valid, message = portfolio.validate_order(
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            side="invalid_side"
        )

        assert is_valid is False
        assert "invalid side" in message.lower()

    def test_create_snapshot(self):
        """Test create_snapshot."""
        portfolio = TestConcretePortfolio(initial_capital=100000.0)

        # Add position
        portfolio.update_position("AAPL", 100.0, 150.0, "buy")
        portfolio.update_prices({"AAPL": 160.0})

        # Create snapshot
        snapshot = portfolio.create_snapshot()

        assert isinstance(snapshot, PortfolioSnapshot)
        assert snapshot.total_value > 0
        assert "AAPL" in snapshot.holdings
        assert isinstance(snapshot.allocations, dict)
        assert isinstance(snapshot.performance, dict)

    def test_portfolio_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Cannot instantiate abstract Portfolio directly
        with pytest.raises(TypeError):
            Portfolio()

    def test_portfolio_concrete_implementation(self):
        """Test concrete portfolio implementation."""
        portfolio = TestConcretePortfolio()

        # Test initialize
        portfolio.initialize()
        assert portfolio.initialized is True

        # Test get_value
        value = portfolio.get_value()
        assert value == 100000.0

        # Test get_positions
        positions = portfolio.get_positions()
        assert isinstance(positions, dict)
        assert "AAPL" in positions

        # Test execute_order
        result = portfolio.execute_order("AAPL", 100.0)
        assert result["symbol"] == "AAPL"
        assert result["quantity"] == 100.0
        assert result["executed"] is True

        # Test rebalance
        portfolio.rebalance()
        assert portfolio.rebalanced is True
#!/usr/bin/env python3
"""
Basic Backtest Example for Djinn Quantitative Backtesting Framework.

This example demonstrates how to:
1. Load market data using Yahoo Finance provider with local caching
2. Create a moving average crossover strategy
3. Run event-driven backtest
4. Analyze and visualize results
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import hashlib
import pickle

# Add parent directory to path to import djinn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.djinn.data.providers.yahoo_finance import YahooFinanceProvider
from src.djinn.data.market_data import AdjustmentType
from src.djinn.core.strategy import (
    MovingAverageCrossover,
    create_moving_average_crossover_strategy,
    PositionSizing
)
from src.djinn.core.backtest import (
    EventDrivenBacktestEngine,
    VectorizedBacktestEngine,
    BacktestMode
)
from src.djinn.utils.config import ConfigManager
from src.djinn.utils.logger import setup_logger

# Setup logging
setup_logger(level="INFO")

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')
CACHE_EXPIRY_DAYS = 1  # Cache expires after 1 day


def ensure_cache_dir():
    """Ensure cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def get_cache_key(symbol, start_date, end_date, interval="1d", adjustment="adj"):
    """
    Generate a unique cache key for the data request.

    Args:
        symbol: Stock symbol
        start_date: Start date (datetime or string)
        end_date: End date (datetime or string)
        interval: Data interval
        adjustment: Price adjustment type

    Returns:
        str: Unique cache key
    """
    # Convert dates to string if needed
    if isinstance(start_date, datetime):
        start_str = start_date.strftime("%Y-%m-%d")
    else:
        start_str = str(start_date)

    if isinstance(end_date, datetime):
        end_str = end_date.strftime("%Y-%m-%d")
    else:
        end_str = str(end_date)

    # Create a string representation of all parameters
    key_str = f"{symbol}_{start_str}_{end_str}_{interval}_{adjustment}"

    # Generate SHA256 hash for shorter filename
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def get_cache_filepath(symbol, start_date, end_date, interval="1d", adjustment="adj"):
    """
    Get cache file path for given parameters.

    Returns:
        str: Full path to cache file
    """
    ensure_cache_dir()
    cache_key = get_cache_key(symbol, start_date, end_date, interval, adjustment)
    filename = f"{symbol}_{cache_key}.csv"
    return os.path.join(CACHE_DIR, filename)


def is_cache_valid(filepath, max_age_days=CACHE_EXPIRY_DAYS):
    """
    Check if cache file exists and is not expired.

    Args:
        filepath: Path to cache file
        max_age_days: Maximum age in days before cache expires

    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not os.path.exists(filepath):
        return False

    # Check file age
    file_mtime = os.path.getmtime(filepath)
    file_age = (time.time() - file_mtime) / (60 * 60 * 24)  # Age in days

    return file_age <= max_age_days


def load_from_cache(filepath):
    """
    Load data from cache file.

    Args:
        filepath: Path to cache file

    Returns:
        pandas.DataFrame or None: Loaded data or None if loading fails
    """
    try:
        # Load CSV file, parse dates and use first column as index
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"  Loaded from cache: {filepath}")
        return data
    except Exception as e:
        print(f"  Warning: Failed to load cache {filepath}: {e}")
        return None


def save_to_cache(filepath, data):
    """
    Save data to cache file.

    Args:
        filepath: Path to cache file
        data: DataFrame to save
    """
    try:
        # Save DataFrame to CSV file with index (dates)
        data.to_csv(filepath, index=True)
        print(f"  Saved to cache: {filepath}")
    except Exception as e:
        print(f"  Warning: Failed to save cache {filepath}: {e}")


def get_symbol_data_with_cache(symbol, start_date, end_date, interval="1d", adjustment="adj"):
    """
    Get market data for a symbol with caching.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        interval: Data interval
        adjustment: Price adjustment type

    Returns:
        pandas.DataFrame: Market data for the symbol
    """
    print(f"  Getting data for {symbol}...")

    # Get cache file path
    cache_file = get_cache_filepath(symbol, start_date, end_date, interval, adjustment)

    # Try to load from cache first
    if is_cache_valid(cache_file):
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            return cached_data

    print(f"    Cache not available or expired, downloading from Yahoo Finance...")

    # Create Yahoo Finance provider
    provider = YahooFinanceProvider(
        cache_enabled=False,  # We handle our own caching
        cache_ttl=3600
    )

    try:
        # Download data
        market_data = provider.get_ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            adjustment=AdjustmentType(adjustment)
        )
        df = market_data.to_dataframe()

        # Save to cache
        if not df.empty:
            save_to_cache(cache_file, df)

        print(f"    Downloaded {len(df)} rows")
        return df

    except Exception as e:
        print(f"    Error downloading {symbol}: {e}")
        raise


def download_market_data():
    """Download sample market data from Yahoo Finance with local caching."""
    print("Getting market data (with local caching)...")

    # Define symbols and date range
    symbols = ["GOOGL"]
    # symbols = ["AAPL", "MSFT", "GOOGL"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 4)  # 4 years of data

    # Get data for each symbol
    data = {}
    for symbol in symbols:
        try:
            df = get_symbol_data_with_cache(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                adjustment="adj"
            )
            df.index = pd.to_datetime(df.index, utc=True)
            data[symbol] = df

        except Exception as e:
            print(f"    Error getting data for {symbol}: {e}")

    return data


def create_sample_data():
    """Create sample data for demonstration when internet is not available."""
    print("Creating sample market data...")

    # Generate sample date range
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')

    data = {}
    symbols = ["GOOGL"]

    for symbol in symbols:
        # Generate random walk for prices
        np.random.seed(42 + ord(symbol[0]))  # Different seed per symbol

        n_periods = len(dates)
        returns = np.random.normal(0.0005, 0.02, n_periods)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.01, n_periods)),
            'high': prices * (1 + np.random.normal(0.02, 0.02, n_periods)),
            'low': prices * (1 - np.random.normal(0.02, 0.02, n_periods)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_periods)
        }, index=dates)

        data[symbol] = df

        print(f"  Created sample data for {symbol}: {len(df)} rows")

    return data


def run_event_driven_backtest(data, strategy):
    """Run event-driven backtest."""
    print("\n" + "="*60)
    print("Running Event-Driven Backtest")
    print("="*60)

    # Create event-driven backtest engine
    backtest_engine = EventDrivenBacktestEngine(
        initial_capital=100000.0,
        commission=0.001,      # 0.1% commission
        slippage=0.0005,       # 0.05% slippage
        allow_short=False,
        max_position_size=0.5,  # 10% max position size
        stop_loss=0.2,          # 10% stop loss
        take_profit=0.4         # 20% take profit
    )

    # Run backtest on first symbol only for simplicity
    symbol = list(data.keys())[0]
    symbol_data = data[symbol]

    # Prepare data dictionary
    backtest_data = {symbol: symbol_data}

    # Define date range
    start_date = symbol_data.index[0]
    end_date = symbol_data.index[-1]

    print(f"Backtesting {symbol} from {start_date.date()} to {end_date.date()}")
    print(f"Data points: {len(symbol_data)}")
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.get_parameters_summary()}")

    # Run backtest
    result = backtest_engine.run(
        strategy=strategy,
        data=backtest_data,
        start_date=start_date,
        end_date=end_date,
        frequency='daily'
    )

    return result


def run_vectorized_backtest(data, strategy):
    """Run vectorized backtest."""
    print("\n" + "="*60)
    print("Running Vectorized Backtest")
    print("="*60)

    # Create vectorized backtest engine
    backtest_engine = VectorizedBacktestEngine(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005,
        allow_short=False,
        max_position_size=0.1,
        stop_loss=0.1,
        take_profit=0.2
    )

    # Run backtest on first symbol only for simplicity
    symbol = list(data.keys())[0]
    symbol_data = data[symbol]

    # Prepare data dictionary
    backtest_data = {symbol: symbol_data}

    # Define date range
    start_date = symbol_data.index[0]
    end_date = symbol_data.index[-1]

    print(f"Backtesting {symbol} from {start_date.date()} to {end_date.date()}")
    print(f"Data points: {len(symbol_data)}")
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.get_parameters_summary()}")

    # Run backtest
    result = backtest_engine.run(
        strategy=strategy,
        data=backtest_data,
        start_date=start_date,
        end_date=end_date,
        frequency='daily'
    )

    return result


def analyze_results(result):
    """Analyze and display backtest results."""
    print("\n" + "="*60)
    print("Backtest Results Analysis")
    print("="*60)

    # Basic performance metrics
    print("\nPerformance Metrics:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Annual Return: {result.annual_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Volatility: {result.volatility:.2%}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.3f}")
    print(f"  Calmar Ratio: {result.calmar_ratio:.3f}")

    # Trade statistics
    print("\nTrade Statistics:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Winning Trades: {result.winning_trades}")
    print(f"  Losing Trades: {result.losing_trades}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Profit Factor: {result.profit_factor:.3f}")
    print(f"  Avg Trade Return: {result.avg_trade_return:.2%}")

    # Portfolio statistics
    print("\nPortfolio Statistics:")
    print(f"  Initial Capital: ${result.initial_capital:,.2f}")
    print(f"  Final Capital: ${result.final_capital:,.2f}")
    print(f"  Peak Capital: ${result.peak_capital:,.2f}")
    print(f"  Trough Capital: ${result.trough_capital:,.2f}")
    print(f"  Total Commission: ${result.total_commission:,.2f}")
    print(f"  Total Slippage: ${result.total_slippage:,.2f}")

    # Risk metrics
    print("\nRisk Metrics:")
    if hasattr(result, 'value_at_risk'):
        print(f"  Value at Risk (95%): {result.value_at_risk:.2%}")
    if hasattr(result, 'conditional_var'):
        print(f"  Conditional VaR (95%): {result.conditional_var:.2%}")

    return result


def print_detailed_trades(result):
    """Print detailed information about all trades in a user-friendly format."""
    print("\n" + "="*60)
    print("Detailed Trade Information")
    print("="*60)

    if not result.trades:
        print("No trades executed during the backtest period.")
        return

    print(f"\nTotal Trades: {len(result.trades)}")
    print("-" * 100)

    # Print header
    print(f"{'Date':<12} {'Time':<8} {'Symbol':<8} {'Side':<6} {'Quantity':<10} {'Price':<10} {'Value':<12} {'Commission':<12} {'Slippage':<10} {'Total Cost':<12}")
    print("-" * 100)

    # Print each trade
    for i, trade in enumerate(result.trades, 1):
        # Calculate trade value (absolute value since quantity can be negative for sells)
        trade_value = abs(trade.quantity) * trade.price

        # Calculate total cost (commission + slippage)
        total_cost = trade.commission + trade.slippage

        # Format timestamp
        trade_date = trade.timestamp.strftime("%Y-%m-%d")
        trade_time = trade.timestamp.strftime("%H:%M:%S")

        # Format numeric values
        quantity_str = f"{trade.quantity:,.2f}"
        price_str = f"${trade.price:,.2f}"
        value_str = f"${trade_value:,.2f}"
        commission_str = f"${trade.commission:,.2f}"
        slippage_str = f"${trade.slippage:,.2f}"
        total_cost_str = f"${total_cost:,.2f}"

        # Print trade details
        print(f"{trade_date:<12} {trade_time:<8} {trade.symbol:<8} {trade.side:<6} {quantity_str:<10} {price_str:<10} {value_str:<12} {commission_str:<12} {slippage_str:<10} {total_cost_str:<12}")

    # Print summary
    print("-" * 100)

    # Calculate totals
    total_buys = sum(1 for t in result.trades if t.side.lower() == 'buy')
    total_sells = sum(1 for t in result.trades if t.side.lower() == 'sell')
    total_quantity = sum(t.quantity for t in result.trades)
    total_value = sum(abs(t.quantity) * t.price for t in result.trades)
    total_commission = sum(t.commission for t in result.trades)
    total_slippage = sum(t.slippage for t in result.trades)
    total_cost = total_commission + total_slippage

    print(f"\nTrade Summary:")
    print(f"  Total Buys: {total_buys}")
    print(f"  Total Sells: {total_sells}")
    print(f"  Net Quantity: {total_quantity:,.2f}")
    print(f"  Total Trade Value: ${total_value:,.2f}")
    print(f"  Total Commission: ${total_commission:,.2f}")
    print(f"  Total Slippage: ${total_slippage:,.2f}")
    print(f"  Total Transaction Costs: ${total_cost:,.2f}")

    # Additional analysis by symbol
    print("\nTrades by Symbol:")
    trades_by_symbol = {}
    for trade in result.trades:
        symbol = trade.symbol
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = {
                'buys': 0,
                'sells': 0,
                'total_quantity': 0,
                'total_value': 0,
                'total_commission': 0,
                'total_slippage': 0
            }

        trades_by_symbol[symbol]['total_quantity'] += trade.quantity
        trades_by_symbol[symbol]['total_value'] += abs(trade.quantity) * trade.price
        trades_by_symbol[symbol]['total_commission'] += trade.commission
        trades_by_symbol[symbol]['total_slippage'] += trade.slippage

        if trade.side.lower() == 'buy':
            trades_by_symbol[symbol]['buys'] += 1
        else:
            trades_by_symbol[symbol]['sells'] += 1

    for symbol, stats in trades_by_symbol.items():
        print(f"  {symbol}:")
        print(f"    Buys: {stats['buys']}, Sells: {stats['sells']}, Net Quantity: {stats['total_quantity']:,.2f}")
        print(f"    Total Value: ${stats['total_value']:,.2f}, Total Costs: ${stats['total_commission'] + stats['total_slippage']:,.2f}")


def plot_candlestick(data, symbol=None, title="Candlestick Chart"):
    """Plot candlestick chart for given OHLCV data.

    Args:
        data: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        symbol: Stock symbol for title (optional)
        title: Chart title
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle

        if symbol is None:
            symbol = list(data.keys())[0] if isinstance(data, dict) else "Unknown"

        # Extract data based on input type
        if isinstance(data, dict):
            df = data[symbol] if symbol in data else list(data.values())[0]
        else:
            df = data

        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns. Available: {list(df.columns)}")
            return

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Prepare data for candlestick plot
        dates = mdates.date2num(df.index.to_pydatetime())
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Determine width of candles (based on date range)
        if len(dates) > 1:
            width = (dates[1] - dates[0]) * 0.8
        else:
            width = 0.8

        # Plot candlesticks
        for i, (date, open_price, high, low, close) in enumerate(zip(dates, opens, highs, lows, closes)):
            color = 'green' if close >= open_price else 'red'

            # Plot high-low line
            ax1.plot([date, date], [low, high], color='black', linewidth=1)

            # Plot open-close rectangle
            rect = Rectangle((date - width/2, min(open_price, close)),
                            width, abs(close - open_price),
                            facecolor=color, edgecolor='black', linewidth=1)
            ax1.add_patch(rect)

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

        # Set labels and title
        chart_title = f"{symbol} - {title}" if symbol else title
        ax1.set_title(chart_title)
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)

        # Plot volume as bar chart
        if 'volume' in df.columns:
            volume_colors = ['green' if close >= open_price else 'red'
                            for open_price, close in zip(opens, closes)]
            ax2.bar(dates, df['volume'].values, width=width,
                   color=volume_colors, edgecolor='black', linewidth=0.5)
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout()

        # Save figure
        # filename = f'candlestick_{symbol}.png'
        # plt.savefig(filename, dpi=150, bbox_inches='tight')
        # print(f"Saved candlestick chart to '{filename}'")

        plt.show(block=False)

    except Exception as e:
        print(f"Warning: Could not create candlestick chart: {e}")
        print("Matplotlib may not be installed correctly.")

def plot_results(result, title="Backtest Results"):
    """Plot backtest results."""
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Equity curve
        ax1 = axes[0]
        result.equity_curve.plot(ax=ax1, title=f'{title} - Equity Curve', color='blue')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        result.drawdown.plot(ax=ax2, title='Drawdown', color='red')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # Returns distribution
        ax3 = axes[2]
        result.returns.hist(ax=ax3, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
        print("\nSaved plot to 'backtest_results.png'")
        plt.show()

    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
        print("Matplotlib may not be installed. Install with: pip install matplotlib")


def main():
    """Main function to run the example."""
    print("="*60)
    print("Djinn Quantitative Backtesting Framework - Basic Example")
    print("="*60)

    # Get market data
    print("\n2. Getting market data...")
    try:
        data = download_market_data()
        if not data:
            print("   Warning: No data downloaded, using sample data")
            raise Exception("No data downloaded")

    except Exception as e:
        print(f"   Error downloading data: {e}")
        print("   Using sample data instead")
        raise Exception("Error downloading data")

    # Plot candlestick chart
    print("\n2.5. Plotting candlestick chart...")
    try:
        # Get first symbol from data dictionary
        symbol = list(data.keys())[0]
        # plot_candlestick(data, symbol=symbol, title=f"{symbol} Candlestick Chart")
    except Exception as e:
        print(f"   Warning: Could not plot candlestick chart: {e}")

    # Create strategy
    print("\n3. Creating trading strategy...")
    strategy = create_moving_average_crossover_strategy(
        fast_period=10,
        slow_period=30,
        ma_type='sma',
        use_volume=False,
        min_crossover_strength=0.01,
        require_confirmation=True,
        confirmation_periods=2,
        position_sizing_params={
            'method': 'fixed_fractional',
            'risk_per_trade': 0.02,
            'max_risk': 0.1,
            'max_position_size': 0.1
        }
    )
    print(f"   Strategy created: {strategy.name}")

    # Run event-driven backtest
    event_driven_result = run_event_driven_backtest(data, strategy)
    analyze_results(event_driven_result)

    # Print detailed trade information
    print_detailed_trades(event_driven_result)

    # # Run vectorized backtest
    # vectorized_result = run_vectorized_backtest(data, strategy)
    # analyze_results(vectorized_result)

    # # Compare results
    # print("\n" + "="*60)
    # print("Comparison of Backtest Engines")
    # print("="*60)
    # print(f"{'Metric':<25} {'Event-Driven':<15} {'Vectorized':<15}")
    # print("-"*55)
    # print(f"{'Total Return':<25} {event_driven_result.total_return:>14.2%} {vectorized_result.total_return:>14.2%}")
    # print(f"{'Sharpe Ratio':<25} {event_driven_result.sharpe_ratio:>14.3f} {vectorized_result.sharpe_ratio:>14.3f}")
    # print(f"{'Max Drawdown':<25} {event_driven_result.max_drawdown:>14.2%} {vectorized_result.max_drawdown:>14.2%}")
    # print(f"{'Total Trades':<25} {event_driven_result.total_trades:>14} {vectorized_result.total_trades:>14}")
    # print(f"{'Win Rate':<25} {event_driven_result.win_rate:>14.2%} {vectorized_result.win_rate:>14.2%}")

    # Plot results
    print("\n5. Generating plots...")
    try:
        plot_results(event_driven_result, title="Event-Driven Backtest")
    except Exception as e:
        print(f"   Warning: Could not generate plots: {e}")

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)

    return {
        'event_driven': event_driven_result,
        # 'vectorized': vectorized_result,
        'data': data,
        'strategy': strategy
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\nResults available in 'results' dictionary")
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment (using uv)
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests with coverage (pytest.ini configures defaults)
pytest

# Run specific test file
pytest tests/test_simple_strategy.py

# Run tests with specific marker
pytest -m "not slow"

# Run tests in parallel
pytest -n auto

# Generate coverage report
pytest --cov=src/djinn --cov-report=html:coverage_html
```

### Code Quality
```bash
# Format code with black
black src/djinn

# Lint and auto-fix with ruff
ruff check src/djinn --fix

# Type checking with mypy (strict mode)
mypy src/djinn

# Run pre-commit hooks on all files
pre-commit run --all-files
```

### Building and Distribution
```bash
# Build wheel and sdist
hatch build

# Clean build artifacts
hatch clean
```

### Running Examples
```bash
# Examples are in the examples/ directory
python examples/ma_crossover_simple_example.py
```

## High-level Architecture

Djinn is a multi-market quantitative backtesting framework with a layered architecture:

### 1. Data Layer (`src/djinn/data/`)
- **DataProvider**: Abstract base class for market data providers (Yahoo Finance, AKShare)
- **MarketData**: Unified data structure for OHLCV and fundamental data
- **CacheManager**: Multi-level caching system with configurable TTL
- **DataCleaner**: Data validation and normalization

### 2. Strategy Layer (`src/djinn/core/strategy/`)

#### Simplified Strategy Framework (Recommended)
- **SimpleStrategy**: Simplified base class for quick strategy development (~15 lines)
- **param()**: Declarative parameter definition with automatic validation
- **Parameter**: Parameter descriptor class with validation rules

#### Pre-built Strategies (`src/djinn/core/strategy/impl/`)
- **RSIStrategy**: RSI overbought/oversold strategy
- **BollingerBandsStrategy**: Bollinger Bands breakout strategy
- **MACDStrategy**: MACD crossover strategy
- **MeanReversionStrategy**: Mean reversion to moving average

#### Legacy Strategy Framework
- **Strategy**: Abstract base class with full lifecycle (advanced users)
- **Signal**: Data class for trade signals with type, strength, and metadata
- **PositionSizing**: Configurable position sizing methods
- **Indicators**: Technical indicator calculations

### 3. Backtest Engine (`src/djinn/core/backtest/`)
- **EventDrivenBacktestEngine**: Event-driven backtesting (precise simulation)
- **VectorizedBacktestEngine**: Vectorized backtesting (high performance)
- **BacktestResult**: Comprehensive result container with 30+ performance metrics
- **Trade/Position**: Data classes representing executed trades and positions
- **Fee Models**: Commission, slippage, tax calculations

### 4. Portfolio Management (`src/djinn/core/portfolio/`)
- **Portfolio**: Manages holdings, cash, and portfolio-level risk
- **Rebalancer**: Implements periodic and threshold-based rebalancing
- **RiskManager**: Position limits, stop-loss, drawdown controls
- **PortfolioBuilder**: Construction methods (equal weight, market cap, risk parity)

### 5. Visualization (`src/djinn/visualization/`)
- **Plotly-based charts**: Interactive equity curves, drawdowns, performance heatmaps
- **Report generation**: HTML/PDF reports with performance summaries

### 6. Utilities (`src/djinn/utils/`)
- **Logger**: Structured logging with loguru
- **ConfigManager**: Centralized configuration from YAML and environment variables
- **DateUtils**: Market calendar and holiday handling
- **Validator**: Data validation with Pydantic integration

## Key Patterns and Conventions

### 1. Strategy Development (SimpleStrategy - Recommended)

Use `SimpleStrategy` for new strategies:

```python
from djinn import SimpleStrategy, param
import pandas as pd
import numpy as np

class MyStrategy(SimpleStrategy):
    # Declare parameters with validation
    fast = param(10, min=2, max=100, description="Fast MA period")
    slow = param(30, min=5, max=200, description="Slow MA period")

    def signals(self, data):
        """Generate trading signals."""
        fast_ma = data['close'].rolling(self.params.fast).mean()
        slow_ma = data['close'].rolling(self.params.slow).mean()
        return pd.Series(np.where(fast_ma > slow_ma, 1, -1), index=data.index)
```

### 2. Strategy Development (Legacy Strategy ABC)

For advanced features, extend the full `Strategy` ABC:

```python
from djinn.core.strategy import Strategy

class AdvancedStrategy(Strategy):
    def initialize(self, market_data):
        # Custom initialization
        pass

    def calculate_indicators(self, data):
        # Calculate and cache indicators
        pass

    def generate_signals(self, data):
        # Generate signals with metadata
        pass
```

### 3. Data Validation
- Extensive use of Pydantic for data validation in `utils/validation.py`
- Custom exceptions (`ValidationError`, `StrategyError`, `BacktestError`) provide context

### 4. Caching Strategy
- Multi-level caching: memory, disk, Redis (optional)
- Cache keys incorporate request parameters and timestamps for freshness

### 5. Configuration Management
- Primary configuration in `configs/backtest_config.yaml`
- Environment variables override YAML settings (see `.env.example`)
- ConfigManager provides type-safe access to settings

### 6. Logging
- Structured logging with loguru throughout
- Log level configurable via environment variable `LOG_LEVEL`

### 7. Error Handling
- Domain-specific exceptions with rich context (symbol, parameters, error details)
- Retry logic with exponential backoff for network operations

## Configuration Files

### `configs/backtest_config.yaml`
Main configuration for backtest parameters:
- `initial_capital`: Starting capital
- `commission`: Commission rate per trade
- `slippage`: Slippage rate
- `risk`: Position limits, stop-loss, max drawdown controls

### `pyproject.toml`
- Project metadata and dependencies
- Tool configurations: black (line-length=88), ruff, mypy (strict mode)
- Hatch build system configuration

### `.pre-commit-config.yaml`
Pre-commit hooks for code quality:
- black formatting
- ruff linting and formatting
- mypy type checking
- bandit security scanning
- detect-secrets for credential detection

## Development Notes

- Python 3.13+ required (type hints, new language features)
- All new code should pass `mypy --strict` with no errors
- Use `loguru` for logging instead of standard `logging` module
- Data providers should extend `DataProvider` ABC and implement caching
- **New strategies should extend `SimpleStrategy`** for simplicity
- Use `param()` for parameter declaration with automatic validation
- Pre-built strategies go in `src/djinn/core/strategy/impl/`
- Backtest engines should extend `BacktestEngine` ABC and support multiple modes

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
pytest tests/unit/test_strategy.py

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
python examples/basic_backtest.py
```

## High-level Architecture

Djinn is a multi-market quantitative backtesting framework with a layered architecture:

### 1. Data Layer (`src/djinn/data/`)
- **DataProvider**: Abstract base class for market data providers (Yahoo Finance, AKShare)
- **MarketData**: Unified data structure for OHLCV and fundamental data
- **CacheManager**: Multi-level caching system with configurable TTL
- **DataCleaner**: Data validation and normalization

### 2. Strategy Layer (`src/djinn/core/strategy/`)
- **Strategy**: Abstract base class defining `initialize()`, `generate_signals()`, `calculate_indicators()`
- **Signal**: Data class for trade signals with type, strength, and metadata
- **PositionSizing**: Configurable position sizing methods (fixed fractional, Kelly, etc.)
- **Indicators**: Technical indicator calculations (MA, MACD, RSI, etc.)

### 3. Backtest Engine (`src/djinn/core/backtest/`)
- **BacktestEngine**: Abstract base class supporting multiple execution modes (event-driven, vectorized, hybrid)
- **BacktestResult**: Comprehensive result container with 50+ performance metrics
- **Trade/Position**: Data classes representing executed trades and portfolio positions
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
- **ConfigManager**: Centralized configuration from YAML files and environment variables
- **DateUtils**: Market calendar and holiday handling
- **Validator**: Data validation with Pydantic integration

## Key Patterns and Conventions

### 1. Abstract Base Classes
- Core components use ABCs to define interfaces (DataProvider, Strategy, BacktestEngine)
- Implementations extend these ABCs (YahooFinanceProvider, MovingAverageCrossover)

### 2. Data Validation
- Extensive use of Pydantic for data validation in `utils/validation.py`
- Custom exceptions (`ValidationError`, `StrategyError`, `BacktestError`) provide context

### 3. Caching Strategy
- Multi-level caching: memory, disk, Redis (optional)
- Cache keys incorporate request parameters and timestamps for freshness

### 4. Configuration Management
- Primary configuration in `configs/backtest_config.yaml`
- Environment variables override YAML settings (see `.env.example`)
- ConfigManager provides type-safe access to settings

### 5. Logging
- Structured logging with loguru throughout
- Log level configurable via environment variable `LOG_LEVEL`

### 6. Error Handling
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
- Strategies should extend `Strategy` ABC and provide parameter validation
- Backtest engines should extend `BacktestEngine` ABC and support multiple modes
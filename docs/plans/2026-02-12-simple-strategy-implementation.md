# SimpleStrategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a simplified strategy framework that allows simple strategies (like dual moving average) to be written in under 20 lines of code, while maintaining extensibility for complex strategies.

**Architecture:** Create a `SimpleStrategy` base class with declarative parameter system using `param()` function. The class auto-collects parameters, validates them, and provides a single `signals()` method for subclasses to implement. Adapters bridge SimpleStrategy to both vectorized and event-driven backtest engines.

**Tech Stack:** Python 3.13+, pandas, numpy, pytest, existing Djinn codebase

---

## Prerequisites

Before starting, ensure you are in the correct worktree and have the virtual environment activated:

```bash
# Check current directory
pwd  # Should be in the worktree directory

# Activate virtual environment
source .venv/bin/activate

# Verify tests run
pytest --version
```

---

## Phase 1: Parameter System

### Task 1: Create Parameter dataclass

**Files:**
- Create: `src/djinn/core/strategy/parameter.py`
- Test: `tests/test_parameter.py`

**Step 1: Write the failing test**

```python
# tests/test_parameter.py
def test_parameter_creation():
    from djinn.core.strategy.parameter import Parameter, param

    p = param(default=10, min=2, max=100, description="Test param")

    assert p.default == 10
    assert p.min == 2
    assert p.max == 100
    assert p.description == "Test param"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_parameter.py::test_parameter_creation -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'djinn.core.strategy.parameter'"

**Step 3: Write minimal implementation**

```python
# src/djinn/core/strategy/parameter.py
from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass(frozen=True)
class Parameter:
    """Parameter declaration with validation rules."""
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    description: Optional[str] = None
    choices: Optional[List[Any]] = None


def param(default, *, min=None, max=None, description=None, choices=None):
    """Create a parameter declaration.

    Args:
        default: Default value for the parameter
        min: Minimum allowed value (for numeric params)
        max: Maximum allowed value (for numeric params)
        description: Human-readable description
        choices: List of allowed values

    Returns:
        Parameter instance
    """
    return Parameter(
        default=default,
        min=min,
        max=max,
        description=description,
        choices=choices
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_parameter.py::test_parameter_creation -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_parameter.py src/djinn/core/strategy/parameter.py
git commit -m "feat: add Parameter dataclass and param() function"
```

---

### Task 2: Add Parameter validation

**Files:**
- Modify: `src/djinn/core/strategy/parameter.py`
- Test: `tests/test_parameter.py`

**Step 1: Write the failing test**

```python
# tests/test_parameter.py
def test_parameter_validation():
    from djinn.core.strategy.parameter import Parameter, param

    p = param(default=10, min=2, max=100)

    # Valid value
    assert p.validate(50) == 50

    # Boundary values
    assert p.validate(2) == 2
    assert p.validate(100) == 100

    # Invalid values should raise
    try:
        p.validate(1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        p.validate(101)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_parameter.py::test_parameter_validation -v`

Expected: FAIL with "AttributeError: 'Parameter' object has no attribute 'validate'"

**Step 3: Write minimal implementation**

```python
# src/djinn/core/strategy/parameter.py
from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass(frozen=True)
class Parameter:
    """Parameter declaration with validation rules."""
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    description: Optional[str] = None
    choices: Optional[List[Any]] = None

    def validate(self, value):
        """Validate a value against this parameter's constraints.

        Args:
            value: Value to validate

        Returns:
            The validated value

        Raises:
            ValueError: If value violates constraints
        """
        # Check choices
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Value {value} not in allowed choices: {self.choices}"
            )

        # Check numeric constraints
        if self.min is not None and value < self.min:
            raise ValueError(
                f"Value {value} is below minimum {self.min}"
            )

        if self.max is not None and value > self.max:
            raise ValueError(
                f"Value {value} is above maximum {self.max}"
            )

        return value


def param(default, *, min=None, max=None, description=None, choices=None):
    """Create a parameter declaration.

    Args:
        default: Default value for the parameter
        min: Minimum allowed value (for numeric params)
        max: Maximum allowed value (for numeric params)
        description: Human-readable description
        choices: List of allowed values

    Returns:
        Parameter instance
    """
    return Parameter(
        default=default,
        min=min,
        max=max,
        description=description,
        choices=choices
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_parameter.py::test_parameter_validation -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_parameter.py src/djinn/core/strategy/parameter.py
git commit -m "feat: add Parameter.validate() method"
```

---

## Phase 2: SimpleStrategy Base Class

### Task 3: Create SimpleStrategy with parameter collection

**Files:**
- Create: `src/djinn/core/strategy/simple.py`
- Test: `tests/test_simple_strategy.py`

**Step 1: Write the failing test**

```python
# tests/test_simple_strategy.py
def test_simple_strategy_parameter_collection():
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10, min=2, max=100)
        slow = param(30, min=5, max=200)

    strategy = TestStrategy()

    assert strategy.params.fast == 10
    assert strategy.params.slow == 30
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simple_strategy.py::test_simple_strategy_parameter_collection -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'djinn.core.strategy.simple'"

**Step 3: Write minimal implementation**

```python
# src/djinn/core/strategy/simple.py
from types import SimpleNamespace
from typing import Any

from .parameter import Parameter


class SimpleStrategy:
    """Simplified strategy base class.

    Subclasses declare parameters as class attributes using param(),
    and implement the signals() method.

    Example:
        class MACrossover(SimpleStrategy):
            fast = param(10, min=2, max=100)
            slow = param(30, min=5, max=200)

            def signals(self, data):
                fast_ma = data.close.rolling(self.params.fast).mean()
                slow_ma = data.close.rolling(self.params.slow).mean()
                return np.where(fast_ma > slow_ma, 1, -1)
    """

    def __init__(self, **kwargs):
        """Initialize strategy with parameters.

        Args:
            **kwargs: Parameter values to override defaults
        """
        # Collect declared parameters
        self._param_definitions = self._collect_param_definitions()

        # Build params namespace with defaults
        params_dict = {
            name: param.default
            for name, param in self._param_definitions.items()
        }

        # Override with provided values
        for key, value in kwargs.items():
            if key in params_dict:
                params_dict[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")

        # Validate all parameters
        self.params = SimpleNamespace(**self._validate_params(params_dict))

    def _collect_param_definitions(self):
        """Collect Parameter declarations from class attributes."""
        definitions = {}
        for name in dir(self.__class__):
            if name.startswith('_'):
                continue
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, Parameter):
                definitions[name] = attr
        return definitions

    def _validate_params(self, params_dict):
        """Validate all parameters against their definitions."""
        validated = {}
        for name, value in params_dict.items():
            param_def = self._param_definitions[name]
            validated[name] = param_def.validate(value)
        return validated

    def signals(self, data):
        """Generate trading signals.

        Subclasses must implement this method.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            pd.Series with signal values:
                1 = buy/hold long
                -1 = sell/hold short (or exit)
                0 = no signal
        """
        raise NotImplementedError(
            "Subclasses must implement signals() method"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simple_strategy.py::test_simple_strategy_parameter_collection -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_simple_strategy.py src/djinn/core/strategy/simple.py
git commit -m "feat: add SimpleStrategy base class with parameter collection"
```

---

### Task 4: Add parameter override and validation in SimpleStrategy

**Files:**
- Modify: `src/djinn/core/strategy/simple.py`
- Test: `tests/test_simple_strategy.py`

**Step 1: Write the failing test**

```python
# tests/test_simple_strategy.py
def test_simple_strategy_parameter_override():
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10, min=2, max=100)
        slow = param(30, min=5, max=200)

    strategy = TestStrategy(fast=20, slow=50)

    assert strategy.params.fast == 20
    assert strategy.params.slow == 50


def test_simple_strategy_parameter_validation():
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10, min=2, max=100)

    # Valid override
    strategy = TestStrategy(fast=50)
    assert strategy.params.fast == 50

    # Invalid value should raise
    try:
        strategy = TestStrategy(fast=1)  # Below min
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        strategy = TestStrategy(fast=101)  # Above max
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simple_strategy.py::test_simple_strategy_parameter_override -v`

Expected: FAIL (if implementation doesn't support override yet)

Run: `pytest tests/test_simple_strategy.py::test_simple_strategy_parameter_validation -v`

Expected: FAIL (if validation not implemented)

**Step 3: Verify implementation already supports this**

The implementation from Task 3 should already support this. Run tests to confirm.

**Step 4: Run all tests**

Run: `pytest tests/test_simple_strategy.py -v`

Expected: All PASS

**Step 5: Commit (if any changes)**

If tests pass without changes, no commit needed. Otherwise:

```bash
git add tests/test_simple_strategy.py src/djinn/core/strategy/simple.py
git commit -m "feat: add parameter override and validation tests"
```

---

### Task 5: Add signals() method implementation test

**Files:**
- Test: `tests/test_simple_strategy.py`

**Step 1: Write the failing test**

```python
# tests/test_simple_strategy.py
import pandas as pd
import numpy as np


def test_simple_strategy_signals():
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class MACrossover(SimpleStrategy):
        fast = param(10)
        slow = param(30)

        def signals(self, data):
            fast_ma = data['close'].rolling(self.params.fast).mean()
            slow_ma = data['close'].rolling(self.params.slow).mean()
            return np.where(fast_ma > slow_ma, 1, -1)

    # Create test data
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'close': range(50)  # Steady uptrend
    }, index=dates)

    strategy = MACrossover()
    signals = strategy.signals(data)

    assert isinstance(signals, (pd.Series, np.ndarray))
    assert len(signals) == 50
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_simple_strategy.py::test_simple_strategy_signals -v`

Expected: PASS (implementation should already work)

**Step 3: Commit**

```bash
git add tests/test_simple_strategy.py
git commit -m "test: add SimpleStrategy signals() implementation test"
```

---

## Phase 3: Backtest Engine Integration

### Task 6: Add adapter methods for vectorized backtest

**Files:**
- Modify: `src/djinn/core/strategy/simple.py`
- Test: `tests/test_simple_strategy.py`

**Step 1: Write the failing test**

```python
# tests/test_simple_strategy.py
def test_simple_strategy_vectorized_adapter():
    """Test that SimpleStrategy can be used with vectorized backtest."""
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10)

        def signals(self, data):
            return np.where(
                data['close'].rolling(self.params.fast).mean() > data['close'],
                1, -1
            )

    strategy = TestStrategy()

    # Test calculate_signals_vectorized adapter
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = pd.DataFrame({'close': range(50)}, index=dates)

    signals = strategy.calculate_signals_vectorized(data)
    assert isinstance(signals, (pd.Series, np.ndarray))
    assert len(signals) == 50
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simple_strategy.py::test_simple_strategy_vectorized_adapter -v`

Expected: FAIL with "AttributeError: 'TestStrategy' object has no attribute 'calculate_signals_vectorized'"

**Step 3: Write minimal implementation**

```python
# src/djinn/core/strategy/simple.py
from types import SimpleNamespace
from typing import Any
from datetime import datetime

import pandas as pd

from .parameter import Parameter


class SimpleStrategy:
    """Simplified strategy base class.

    Subclasses declare parameters as class attributes using param(),
    and implement the signals() method.

    Example:
        class MACrossover(SimpleStrategy):
            fast = param(10, min=2, max=100)
            slow = param(30, min=5, max=200)

            def signals(self, data):
                fast_ma = data.close.rolling(self.params.fast).mean()
                slow_ma = data.close.rolling(self.params.slow).mean()
                return np.where(fast_ma > slow_ma, 1, -1)
    """

    def __init__(self, **kwargs):
        """Initialize strategy with parameters.

        Args:
            **kwargs: Parameter values to override defaults
        """
        # Collect declared parameters
        self._param_definitions = self._collect_param_definitions()

        # Build params namespace with defaults
        params_dict = {
            name: param.default
            for name, param in self._param_definitions.items()
        }

        # Override with provided values
        for key, value in kwargs.items():
            if key in params_dict:
                params_dict[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")

        # Validate all parameters
        self.params = SimpleNamespace(**self._validate_params(params_dict))

    def _collect_param_definitions(self):
        """Collect Parameter declarations from class attributes."""
        definitions = {}
        for name in dir(self.__class__):
            if name.startswith('_'):
                continue
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, Parameter):
                definitions[name] = attr
        return definitions

    def _validate_params(self, params_dict):
        """Validate all parameters against their definitions."""
        validated = {}
        for name, value in params_dict.items():
            param_def = self._param_definitions[name]
            validated[name] = param_def.validate(value)
        return validated

    def signals(self, data):
        """Generate trading signals.

        Subclasses must implement this method.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            pd.Series with signal values:
                1 = buy/hold long
                -1 = sell/hold short (or exit)
                0 = no signal
        """
        raise NotImplementedError(
            "Subclasses must implement signals() method"
        )

    # Adapter methods for backtest engine compatibility

    def calculate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Adapter for vectorized backtest engine.

        Delegates to signals() method.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            pd.Series with signal values
        """
        result = self.signals(data)
        if isinstance(result, pd.Series):
            return result
        return pd.Series(result, index=data.index)

    def calculate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_date: datetime
    ) -> float:
        """Adapter for event-driven backtest engine.

        Args:
            symbol: Symbol to calculate signal for
            data: Historical data up to current_date
            current_date: Current date for signal calculation

        Returns:
            float: Signal value for the current date
        """
        signals = self.signals(data)

        if isinstance(signals, pd.Series):
            if current_date in signals.index:
                return float(signals.loc[current_date])
            # Find last available signal
            mask = signals.index <= current_date
            if mask.any():
                return float(signals[mask].iloc[-1])
        else:
            # numpy array - return last value
            if len(signals) > 0:
                return float(signals[-1])

        return 0.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simple_strategy.py::test_simple_strategy_vectorized_adapter -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_simple_strategy.py src/djinn/core/strategy/simple.py
git commit -m "feat: add adapter methods for backtest engine integration"
```

---

### Task 7: Add event-driven adapter test

**Files:**
- Test: `tests/test_simple_strategy.py`

**Step 1: Write the failing test**

```python
# tests/test_simple_strategy.py
from datetime import datetime


def test_simple_strategy_event_driven_adapter():
    """Test that SimpleStrategy can be used with event-driven backtest."""
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10)

        def signals(self, data):
            return np.where(
                data['close'].rolling(self.params.fast).mean() > data['close'],
                1, -1
            )

    strategy = TestStrategy()

    # Test calculate_signal adapter
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = pd.DataFrame({'close': range(50)}, index=dates)

    # Get signal for specific date
    current_date = dates[30]
    signal = strategy.calculate_signal('AAPL', data, current_date)

    assert isinstance(signal, float)
    assert signal in [-1.0, 0.0, 1.0]
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_simple_strategy.py::test_simple_strategy_event_driven_adapter -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_simple_strategy.py
git commit -m "test: add event-driven adapter test"
```

---

## Phase 4: Module Integration

### Task 8: Export from strategy module

**Files:**
- Modify: `src/djinn/core/strategy/__init__.py`

**Step 1: Update strategy module exports**

```python
# src/djinn/core/strategy/__init__.py
# Add to existing imports

from .parameter import (
    Parameter,
    param
)

from .simple import (
    SimpleStrategy
)

# Add to __all__ list
__all__ = [
    # ... existing exports ...

    # Parameter system
    'Parameter',
    'param',

    # Simple strategy
    'SimpleStrategy',

    # ... rest of existing exports ...
]
```

**Step 2: Write test for exports**

```python
# tests/test_strategy_exports.py
def test_strategy_module_exports():
    """Test that all expected classes can be imported from strategy module."""
    from djinn.core.strategy import (
        Parameter,
        param,
        SimpleStrategy,
    )

    # Verify they are the correct types
    assert callable(param)
    assert isinstance(Parameter, type)
    assert isinstance(SimpleStrategy, type)
```

**Step 3: Run test**

Run: `pytest tests/test_strategy_exports.py -v`

Expected: PASS

**Step 4: Commit**

```bash
git add tests/test_strategy_exports.py src/djinn/core/strategy/__init__.py
git commit -m "feat: export Parameter, param, and SimpleStrategy from strategy module"
```

---

### Task 9: Export from main djinn package

**Files:**
- Modify: `src/djinn/__init__.py`

**Step 1: Check current djinn package exports**

Read: `src/djinn/__init__.py`

**Step 2: Add SimpleStrategy exports**

```python
# src/djinn/__init__.py
# Add imports for simplified strategy framework

from djinn.core.strategy import (
    SimpleStrategy,
    param,
    Parameter,
)

# Add to __all__ if it exists
```

**Step 3: Write test for package-level exports**

```python
# tests/test_djinn_exports.py
def test_djinn_package_exports():
    """Test that key classes can be imported directly from djinn package."""
    import djinn

    # Verify key exports exist
    assert hasattr(djinn, 'SimpleStrategy')
    assert hasattr(djinn, 'param')
    assert hasattr(djinn, 'Parameter')
```

**Step 4: Run test**

Run: `pytest tests/test_djinn_exports.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_djinn_exports.py src/djinn/__init__.py
git commit -m "feat: export SimpleStrategy and param from main djinn package"
```

---

## Phase 5: Example Strategy

### Task 10: Create example MACrossover strategy

**Files:**
- Create: `examples/simple_strategy_example.py`

**Step 1: Create example file**

```python
# examples/simple_strategy_example.py
"""
Example of using SimpleStrategy to implement a moving average crossover strategy.

This demonstrates how a strategy that previously required ~500 lines can now
be written in ~15 lines.
"""

import numpy as np
import pandas as pd

from djinn import SimpleStrategy, param


class MACrossover(SimpleStrategy):
    """Moving Average Crossover Strategy.

    Generates buy signals when fast MA crosses above slow MA,
    sell signals when fast MA crosses below slow MA.
    """
    fast = param(10, min=2, max=100, description="Fast MA period")
    slow = param(30, min=5, max=200, description="Slow MA period")

    def signals(self, data):
        """Generate trading signals based on MA crossover.

        Args:
            data: DataFrame with 'close' column

        Returns:
            pd.Series: 1 for buy, -1 for sell
        """
        fast_ma = data['close'].rolling(self.params.fast).mean()
        slow_ma = data['close'].rolling(self.params.slow).mean()

        return np.where(fast_ma > slow_ma, 1, -1)


def main():
    """Run example backtest."""
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    data = pd.DataFrame({'close': prices}, index=dates)

    # Create strategy with custom parameters
    strategy = MACrossover(fast=5, slow=20)

    # Generate signals
    signals = strategy.signals(data)

    print(f"Strategy: {strategy.__class__.__name__}")
    print(f"Parameters: fast={strategy.params.fast}, slow={strategy.params.slow}")
    print(f"Signals generated: {len(signals)}")
    print(f"Buy signals: {(signals == 1).sum()}")
    print(f"Sell signals: {(signals == -1).sum()}")


if __name__ == '__main__':
    main()
```

**Step 2: Verify example runs**

Run: `python examples/simple_strategy_example.py`

Expected: Output showing strategy parameters and signal counts

**Step 3: Commit**

```bash
git add examples/simple_strategy_example.py
git commit -m "feat: add SimpleStrategy MACrossover example"
```

---

## Phase 6: Final Integration Test

### Task 11: Create end-to-end integration test

**Files:**
- Create: `tests/test_simple_strategy_integration.py`

**Step 1: Create integration test**

```python
# tests/test_simple_strategy_integration.py
"""
Integration tests for SimpleStrategy with backtest components.
"""
import numpy as np
import pandas as pd

from djinn import SimpleStrategy, param


class RSIStrategy(SimpleStrategy):
    """Simple RSI-based strategy for testing."""
    period = param(14, min=5, max=50)
    overbought = param(70, min=50, max=90)
    oversold = param(30, min=10, max=50)

    def signals(self, data):
        # Simple RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.params.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.params.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return np.where(
            rsi < self.params.oversold, 1,
            np.where(rsi > self.params.overbought, -1, 0)
        )


class BollingerBandsStrategy(SimpleStrategy):
    """Bollinger Bands mean reversion strategy."""
    period = param(20, min=5, max=100)
    std_dev = param(2.0, min=0.5, max=4.0)

    def signals(self, data):
        ma = data['close'].rolling(self.params.period).mean()
        std = data['close'].rolling(self.params.period).std()
        upper = ma + self.params.std_dev * std
        lower = ma - self.params.std_dev * std

        return np.where(
            data['close'] < lower, 1,  # Buy when below lower band
            np.where(data['close'] > upper, -1, 0)  # Sell when above upper band
        )


def test_rsi_strategy():
    """Test RSI strategy generates correct signals."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    # Create oscillating price for RSI testing
    prices = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, 100))
    data = pd.DataFrame({'close': prices}, index=dates)

    strategy = RSIStrategy()
    signals = strategy.signals(data)

    assert len(signals) == 100
    assert isinstance(signals, (pd.Series, np.ndarray))


def test_bollinger_bands_strategy():
    """Test Bollinger Bands strategy."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    data = pd.DataFrame({'close': prices}, index=dates)

    strategy = BollingerBandsStrategy(period=10, std_dev=1.5)
    signals = strategy.signals(data)

    assert strategy.params.period == 10
    assert strategy.params.std_dev == 1.5
    assert len(signals) == 100


def test_strategy_parameter_validation_integration():
    """Test parameter validation in complex scenarios."""
    # Valid parameters
    strategy = RSIStrategy(period=20, overbought=75, oversold=25)
    assert strategy.params.period == 20

    # Invalid period (below min)
    try:
        strategy = RSIStrategy(period=3)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # Invalid overbought (above max)
    try:
        strategy = RSIStrategy(overbought=95)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_multiple_strategies_same_data():
    """Test that multiple strategies can be run on same data."""
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(50) * 0.5)
    }, index=dates)

    ma_strategy = MACrossover(fast=5, slow=10)
    rsi_strategy = RSIStrategy(period=10)
    bb_strategy = BollingerBandsStrategy(period=10)

    ma_signals = ma_strategy.signals(data)
    rsi_signals = rsi_strategy.signals(data)
    bb_signals = bb_strategy.signals(data)

    assert len(ma_signals) == len(rsi_signals) == len(bb_signals) == 50
```

**Step 2: Run integration tests**

Run: `pytest tests/test_simple_strategy_integration.py -v`

Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_simple_strategy_integration.py
git commit -m "test: add comprehensive integration tests for SimpleStrategy"
```

---

## Phase 7: Documentation

### Task 12: Add docstrings and type hints

**Files:**
- Modify: `src/djinn/core/strategy/parameter.py`
- Modify: `src/djinn/core/strategy/simple.py`

**Step 1: Review and improve docstrings**

Ensure all public methods have:
- Description
- Args section
- Returns section
- Raises section (if applicable)

**Step 2: Add type hints where missing**

Example:
```python
def param(
    default: Any,
    *,
    min: Optional[float] = None,
    max: Optional[float] = None,
    description: Optional[str] = None,
    choices: Optional[List[Any]] = None
) -> Parameter:
```

**Step 3: Run type checker**

Run: `mypy src/djinn/core/strategy/parameter.py src/djinn/core/strategy/simple.py`

Expected: No errors (or only pre-existing errors)

**Step 4: Commit**

```bash
git add src/djinn/core/strategy/parameter.py src/djinn/core/strategy/simple.py
git commit -m "docs: add comprehensive docstrings and type hints"
```

---

## Final Verification

### Task 13: Run full test suite

**Step 1: Run all new tests**

```bash
pytest tests/test_parameter.py tests/test_simple_strategy.py tests/test_strategy_exports.py tests/test_djinn_exports.py tests/test_simple_strategy_integration.py -v
```

Expected: All PASS

**Step 2: Run existing tests to ensure no regressions**

```bash
pytest tests/ -v --ignore=tests/test_backtest.py  # Skip if backtest tests require complex setup
```

Expected: No new failures

**Step 3: Run example**

```bash
python examples/simple_strategy_example.py
```

Expected: Clean output

**Step 4: Final commit (if any changes)**

```bash
git add .
git commit -m "feat: complete SimpleStrategy implementation" || echo "No changes to commit"
```

---

## Summary

This implementation plan creates:

1. **Parameter System** (`parameter.py`)
   - `Parameter` dataclass with validation
   - `param()` helper function

2. **SimpleStrategy Base Class** (`simple.py`)
   - Automatic parameter collection from class attributes
   - Parameter validation
   - Single `signals()` method to implement
   - Adapter methods for both vectorized and event-driven backtest engines

3. **Module Integration**
   - Exported from `djinn.core.strategy`
   - Exported from main `djinn` package

4. **Examples and Tests**
   - MACrossover example
   - Comprehensive unit and integration tests

**Usage:**
```python
from djinn import SimpleStrategy, param

class MACrossover(SimpleStrategy):
    fast = param(10, min=2, max=100)
    slow = param(30, min=5, max=200)

    def signals(self, data):
        fast_ma = data.close.rolling(self.params.fast).mean()
        slow_ma = data.close.rolling(self.params.slow).mean()
        return np.where(fast_ma > slow_ma, 1, -1)
```

---

**Plan complete and saved to `docs/plans/2026-02-12-simple-strategy-implementation.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**

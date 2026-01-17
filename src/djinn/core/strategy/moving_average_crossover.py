"""
移动平均线交叉策略模块。

本模块实现基于快速和慢速移动平均线交叉的交易策略，生成买入/卖出信号。

目的：
1. 提供经典的移动平均线交叉策略实现，作为策略开发示例
2. 演示如何继承Strategy基类实现具体交易策略
3. 展示技术指标计算、信号生成、参数配置等完整流程

实现方案：
1. 继承Strategy基类，实现initialize、calculate_indicators、generate_signals方法
2. 支持简单移动平均线（SMA）和指数移动平均线（EMA）两种类型
3. 提供参数配置：快慢线周期、移动平均类型、成交量确认、交叉强度阈值等
4. 支持信号确认机制和向量化信号计算

使用方法：
1. 使用create_moving_average_crossover_strategy便捷函数创建策略实例
2. 配置自定义参数：strategy = MovingAverageCrossover(parameters={'fast_period': 10, 'slow_period': 30})
3. 在回测引擎中使用策略进行回测
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from .base import Strategy, Signal, SignalType, PositionSizing
from .indicators import TechnicalIndicators
from ...data.market_data import MarketData
from ...utils.exceptions import StrategyError
from ...utils.logger import logger


class MovingAverageCrossover(Strategy):
    """
    移动平均线交叉策略类。

    基于快速和慢速移动平均线的交叉生成交易信号，当快线上穿慢线时产生买入信号，下穿时产生卖出信号。

    目的：
    1. 实现经典的移动平均线交叉策略，验证趋势跟踪策略效果
    2. 提供可配置的参数系统，支持策略优化和参数调优
    3. 展示完整的策略生命周期管理：初始化、指标计算、信号生成

    实现方案：
    1. 计算快速和慢速移动平均线（支持SMA和EMA）
    2. 检测交叉点：快线上穿慢线（金叉）为看涨，下穿（死叉）为看跌
    3. 支持信号确认机制：要求交叉信号持续多个周期才生效
    4. 支持交叉强度阈值：过滤微弱的交叉信号
    5. 可选成交量确认：结合成交量指标增强信号可靠性

    使用方法：
    1. 创建策略实例：strategy = MovingAverageCrossover(parameters={'fast_period': 10, 'slow_period': 30})
    2. 初始化策略：strategy.initialize(market_data)
    3. 更新策略并获取信号：signals = strategy.update(market_data)
    """

    def __init__(
        self,
        parameters: Dict[str, Any],
        position_sizing: Optional[PositionSizing] = None,
    ):
        """
        初始化移动平均线交叉策略。

        目的：
        1. 设置策略参数和默认值，合并用户提供的参数
        2. 调用父类构造函数，完成基础策略初始化
        3. 初始化策略状态变量，设置最小数据点要求

        参数：
            parameters: 策略参数字典，可包含以下键：
                - fast_period: 快速移动平均线周期，默认10
                - slow_period: 慢速移动平均线周期，默认30
                - ma_type: 移动平均线类型，'sma'（简单移动平均）或'ema'（指数移动平均），默认'sma'
                - use_volume: 是否使用成交量确认，默认False
                - min_crossover_strength: 最小交叉强度阈值，默认0.1
                - require_confirmation: 是否需要信号确认，默认True
                - confirmation_periods: 信号确认周期数，默认2
            position_sizing: 仓位管理配置对象，可选

        实现方案：
        1. 设置默认参数字典
        2. 合并用户提供的参数（用户参数优先）
        3. 调用父类Strategy.__init__方法完成基础初始化
        4. 初始化策略状态变量：fast_ma、slow_ma、crossover_signals、symbol
        5. 根据slow_period设置最小数据点要求，增加10个周期的缓冲

        使用方法：
        1. 直接实例化：strategy = MovingAverageCrossover(parameters={'fast_period': 10, 'slow_period': 30})
        2. 使用便捷函数：strategy = create_moving_average_crossover_strategy(fast_period=10, slow_period=30)
        """
        # Set default parameters
        self.market_data = None
        default_params = {
            'fast_period': 10,
            'slow_period': 30,
            'ma_type': 'sma',  # 'sma' or 'ema'
            'use_volume': False,
            'min_crossover_strength': 0.1,
            'require_confirmation': True,  # Require confirmation period
            'confirmation_periods': 2,  # Number of periods for confirmation
        }

        # Merge with provided parameters
        merged_params = {**default_params, **parameters}

        super().__init__(
            name="MovingAverageCrossover",
            parameters=merged_params,
            position_sizing=position_sizing
        )

        # Strategy state
        self.fast_ma: Optional[pd.Series] = None
        self.slow_ma: Optional[pd.Series] = None
        self.crossover_signals: Optional[pd.Series] = None
        self.symbol: Optional[str] = None

        # Set minimum data points required for this strategy
        self.min_data_points = merged_params['slow_period'] + 10  # Slow period plus buffer

        logger.info(f"Initialized MovingAverageCrossover strategy with parameters: {merged_params}")

    def initialize(self, data: MarketData) -> None:
        """
        使用市场数据初始化策略。

        目的：
        1. 从市场数据中提取交易品种符号，设置策略交易标的
        2. 验证数据充足性，确保有足够的数据点计算指标
        3. 标记策略为已初始化状态，准备计算指标和生成信号

        参数：
            data: MarketData对象，包含OHLCV市场数据

        实现方案：
        1. 从数据对象中提取符号（支持多种数据格式：MarketData对象、DataFrame、字典等）
        2. 存储市场数据引用供后续使用
        3. 调用_validate_data_sufficiency方法验证数据充足性
        4. 设置self.initialized = True标记初始化完成

        使用方法：
        1. 在策略使用前必须调用：strategy.initialize(market_data)
        2. 初始化后即可调用update方法生成信号
        """
        try:
            # For simplicity, assume single symbol for now
            # In real implementation, you'd handle multiple symbols
            # Get symbol from data (handle both .symbol and .symbols attributes)
            if hasattr(data, 'symbol'):
                self.symbol = data.symbol
            elif hasattr(data, 'symbols') and data.symbols:
                self.symbol = data.symbols[0]
            elif isinstance(data, dict) and len(data) > 0:
                self.symbol = next(iter(data))
            else:
                raise StrategyError("No symbols found in market data")

            logger.info(f"Initializing strategy for symbol: {self.symbol}")

            # Store market data reference for later use
            self.market_data = data

            # Validate that we have enough data for indicator calculation
            self._validate_data_sufficiency(data)

            # Mark as initialized
            self.initialized = True

        except Exception as e:
            raise StrategyError(
                "Failed to initialize MovingAverageCrossover strategy",
                strategy_name=self.name,
                parameters=self.parameters,
                details={"error": str(e)}
            )

    def calculate_indicators(self, data: MarketData) -> Dict[str, pd.Series]:
        """
        计算技术指标。

        目的：
        1. 根据配置计算快速和慢速移动平均线
        2. 基于移动平均线计算交叉信号
        3. 准备指标字典供信号生成和策略分析使用

        参数：
            data: MarketData对象，包含OHLCV市场数据

        返回：
            Dict[str, pd.Series]: 指标字典，键为指标名称，值为pandas Series

        实现方案：
        1. 从市场数据中提取指定符号的收盘价数据
        2. 根据ma_type参数计算SMA或EMA：使用TechnicalIndicators工具类
        3. 调用_calculate_crossover_signals方法计算交叉信号
        4. 构建指标字典包含：fast_ma、slow_ma、crossover_signal
        5. 如果配置use_volume=True，额外计算成交量移动平均线

        使用方法：
        1. 通常通过update方法自动调用，无需手动调用
        2. 计算结果存储在self.fast_ma、self.slow_ma、self.crossover_signals中
        3. 可通过self.indicators访问最近计算的指标字典
        """
        try:
            if not self.symbol:
                raise StrategyError("Strategy not initialized - no symbol set")

            # Get symbol data from market data object
            symbol_data = self._get_symbol_dataframe(data, self.symbol)
            if symbol_data is None or symbol_data.empty:
                raise StrategyError(f"No data found for symbol: {self.symbol}")

            close_prices = symbol_data['close']

            # Calculate moving averages based on configured type
            ma_type = self.parameters['ma_type']
            fast_period = int(self.parameters['fast_period'])
            slow_period = int(self.parameters['slow_period'])

            if ma_type == 'sma':
                # Calculate SMAs
                fast_ma_result = TechnicalIndicators.simple_moving_average(
                    close_prices, window=fast_period
                )
                slow_ma_result = TechnicalIndicators.simple_moving_average(
                    close_prices, window=slow_period
                )

                self.fast_ma = fast_ma_result.values
                self.slow_ma = slow_ma_result.values

            elif ma_type == 'ema':
                # Calculate EMAs
                fast_ma_result = TechnicalIndicators.exponential_moving_average(
                    close_prices, span=fast_period
                )
                slow_ma_result = TechnicalIndicators.exponential_moving_average(
                    close_prices, span=slow_period
                )

                self.fast_ma = fast_ma_result.values
                self.slow_ma = slow_ma_result.values

            else:
                raise StrategyError(
                    f"Unsupported moving average type: {ma_type}",
                    strategy_name=self.name,
                    parameters=self.parameters
                )

            # Calculate crossover signals
            self._calculate_crossover_signals()

            # Prepare indicators dictionary
            indicators = {
                'fast_ma': self.fast_ma,
                'slow_ma': self.slow_ma,
                'crossover_signal': self.crossover_signals
            }

            # Add volume indicator if configured
            if self.parameters.get('use_volume', False):
                volume = symbol_data['volume']
                # Calculate volume moving average
                volume_ma = volume.rolling(window=20).mean()
                indicators['volume_ma'] = volume_ma

            logger.debug(f"Calculated indicators for {self.symbol}")
            return indicators

        except Exception as e:
            raise StrategyError(
                "Failed to calculate indicators for MovingAverageCrossover strategy",
                strategy_name=self.name,
                parameters=self.parameters,
                details={"error": str(e)}
            )

    def generate_signals(self, data: MarketData) -> List[Signal]:
        """
        基于移动平均线交叉生成交易信号。

        目的：
        1. 分析移动平均线交叉信号，生成具体的交易指令
        2. 应用信号确认机制和强度阈值过滤，提高信号质量
        3. 封装信号信息为Signal对象，便于回测引擎处理

        参数：
            data: MarketData对象，包含OHLCV市场数据

        返回：
            List[Signal]: Signal对象列表，可能为空列表（无信号）

        实现方案：
        1. 检查指标是否已计算，如未计算则先调用calculate_indicators
        2. 获取最新的交叉信号值
        3. 如果配置require_confirmation=True，验证信号是否持续了confirmation_periods个周期
        4. 应用min_crossover_strength阈值过滤微弱信号
        5. 根据信号值生成买入或卖出Signal对象
        6. 基于快慢线差值计算信号强度

        使用方法：
        1. 通常通过update方法自动调用，无需手动调用
        2. 返回的信号列表可直接传递给回测引擎执行
        """
        try:
            if not self.symbol:
                raise StrategyError("Strategy not initialized - no symbol set")

            if self.fast_ma is None or self.slow_ma is None or self.crossover_signals is None:
                # Calculate indicators first
                self.calculate_indicators(data)

            symbol_data = self._get_symbol_dataframe(data, self.symbol)
            if symbol_data is None or symbol_data.empty:
                raise StrategyError(f"No data found for symbol: {self.symbol}")

            signals = []
            close_prices = symbol_data['close']

            # Get the latest data point
            latest_date = close_prices.index[-1]
            latest_close = close_prices.iloc[-1]

            # Get latest crossover signal
            latest_signal = self.crossover_signals.iloc[-1] if not self.crossover_signals.empty else 0

            # Apply confirmation requirement if configured
            if self.parameters.get('require_confirmation', True):
                confirmation_periods = self.parameters.get('confirmation_periods', 2)

                if len(self.crossover_signals) >= confirmation_periods:
                    # Check if signal has been consistent for confirmation periods
                    recent_signals = self.crossover_signals.iloc[-confirmation_periods:]
                    if not (recent_signals == latest_signal).all():
                        # Signal not confirmed, return no signal
                        return signals

            # Apply minimum crossover strength threshold
            if abs(latest_signal) < self.parameters.get('min_crossover_strength', 0.1):
                return signals

            # Generate signal based on crossover
            if latest_signal > 0:
                # Bullish crossover: fast MA crosses above slow MA
                signal_type = SignalType.BUY
                reason = f"Bullish MA crossover: fast MA ({self.parameters['fast_period']}) crossed above slow MA ({self.parameters['slow_period']})"

            elif latest_signal < 0:
                # Bearish crossover: fast MA crosses below slow MA
                signal_type = SignalType.SELL
                reason = f"Bearish MA crossover: fast MA ({self.parameters['fast_period']}) crossed below slow MA ({self.parameters['slow_period']})"

            else:
                # No crossover
                return signals

            # Calculate signal strength based on crossover magnitude
            if self.fast_ma is not None and self.slow_ma is not None:
                fast_val = self.fast_ma.iloc[-1]
                slow_val = self.slow_ma.iloc[-1]

                if slow_val != 0:
                    # Strength based on percentage difference
                    strength = abs(fast_val - slow_val) / slow_val
                    strength = min(strength, 1.0)  # Cap at 1.0
                else:
                    strength = 0.5  # Default strength
            else:
                strength = 0.5

            # Create signal
            signal = self._create_signal(
                timestamp=latest_date,
                symbol=self.symbol,
                signal_type=signal_type,
                price=latest_close,
                strength=strength,
                reason=reason,
                metadata={
                    'fast_period': self.parameters['fast_period'],
                    'slow_period': self.parameters['slow_period'],
                    'ma_type': self.parameters['ma_type'],
                    'crossover_value': latest_signal
                }
            )

            signals.append(signal)
            logger.info(f"Generated {signal_type.value} signal for {self.symbol}: {reason}")

            return signals

        except Exception as e:
            raise StrategyError(
                "Failed to generate signals for MovingAverageCrossover strategy",
                strategy_name=self.name,
                parameters=self.parameters,
                details={"error": str(e)}
            )

    def calculate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate signals using vectorized operations for vectorized backtesting.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)

        Raises:
            StrategyError: If vectorized signal calculation fails
        """
        try:
            if data.empty:
                return pd.Series(dtype=float)

            close_prices = data['close']

            # Calculate moving averages
            fast_period = int(self.parameters['fast_period'])
            slow_period = int(self.parameters['slow_period'])
            ma_type = self.parameters['ma_type']

            if ma_type == 'sma':
                fast_ma = close_prices.rolling(window=fast_period).mean()
                slow_ma = close_prices.rolling(window=slow_period).mean()
            elif ma_type == 'ema':
                fast_ma = close_prices.ewm(span=fast_period, adjust=False).mean()
                slow_ma = close_prices.ewm(span=slow_period, adjust=False).mean()
            else:
                raise StrategyError(
                    f"Unsupported moving average type: {ma_type}",
                    strategy_name=self.name,
                    parameters=self.parameters
                )

            # Calculate crossover signals
            signals = pd.Series(0, index=close_prices.index)

            # Bullish crossover: fast MA crosses above slow MA
            bullish = (fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())
            signals[bullish] = 1

            # Bearish crossover: fast MA crosses below slow MA
            bearish = (fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())
            signals[bearish] = -1

            # Apply confirmation requirement if configured
            if self.parameters.get('require_confirmation', True):
                confirmation_periods = self.parameters.get('confirmation_periods', 2)

                # Create confirmed signals
                confirmed_signals = pd.Series(0, index=signals.index)

                for i in range(confirmation_periods, len(signals)):
                    window = signals.iloc[i-confirmation_periods:i+1]

                    # Check if all signals in window are the same and non-zero
                    if window.nunique() == 1 and window.iloc[0] != 0:
                        confirmed_signals.iloc[i] = window.iloc[0]

                signals = confirmed_signals

            # Apply minimum crossover strength threshold
            if self.parameters.get('min_crossover_strength', 0.1) > 0:
                # Calculate crossover strength
                crossover_strength = (fast_ma - slow_ma).abs() / slow_ma.abs()

                # Filter signals by strength
                strength_threshold = self.parameters['min_crossover_strength']
                weak_signals = crossover_strength < strength_threshold
                signals[weak_signals] = 0

            logger.debug(f"Calculated vectorized signals: {signals[signals != 0].count()} non-zero signals")
            return signals

        except Exception as e:
            raise StrategyError(
                "Failed to calculate vectorized signals for MovingAverageCrossover strategy",
                strategy_name=self.name,
                parameters=self.parameters,
                details={"error": str(e)}
            )

    def calculate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_date: datetime,
    ) -> float:
        """
        Calculate trading signal for a specific symbol and date.

        Args:
            symbol: Symbol to calculate signal for
            data: Historical data DataFrame up to current_date
            current_date: Current date for signal calculation

        Returns:
            float: Signal value (positive for buy, negative for sell, 0 for no signal)
        """
        try:
            # Check if we have enough data
            if len(data) < self.min_data_points:
                return 0.0

            # Use vectorized signal calculation for efficiency
            signals = self.calculate_signals_vectorized(data)

            # Get the signal for the current date
            # If current_date is not in the index, use the last available signal
            if current_date in signals.index:
                signal = signals.loc[current_date]
            else:
                # Find the last date before or equal to current_date
                mask = signals.index <= current_date
                if mask.any():
                    signal = signals[mask].iloc[-1]
                else:
                    signal = 0.0

            return float(signal)

        except Exception as e:
            logger.warning(f"Error calculating signal for {symbol}: {e}")
            return 0.0

    def _calculate_crossover_signals(self) -> None:
        """Calculate crossover signals from fast and slow moving averages."""
        if self.fast_ma is None or self.slow_ma is None:
            raise StrategyError(
                "Moving averages not calculated",
                strategy_name=self.name,
                parameters=self.parameters
            )

        # Initialize signals series
        self.crossover_signals = pd.Series(0, index=self.fast_ma.index)

        # Calculate where fast MA crosses above slow MA (bullish)
        bullish_cross = (self.fast_ma > self.slow_ma) & (self.fast_ma.shift() <= self.slow_ma.shift())

        # Calculate where fast MA crosses below slow MA (bearish)
        bearish_cross = (self.fast_ma < self.slow_ma) & (self.fast_ma.shift() >= self.slow_ma.shift())

        # Assign signal values
        # Use continuous values instead of binary for signal strength
        ma_diff = self.fast_ma - self.slow_ma
        normalized_diff = ma_diff / self.slow_ma.abs()  # Percentage difference

        # Apply to crossover points
        self.crossover_signals[bullish_cross] = normalized_diff[bullish_cross]
        self.crossover_signals[bearish_cross] = normalized_diff[bearish_cross]

        # Optional: Add some persistence to signals between crossovers
        # For simplicity, we keep signals only at crossover points

        logger.debug(f"Crossover signals calculated: {self.crossover_signals[self.crossover_signals != 0].count()} crossovers")

    def _validate_data_sufficiency(self, data: MarketData) -> None:
        """
        Validate that we have enough data for indicator calculation.

        Args:
            data: MarketData object containing OHLCV data

        Raises:
            StrategyError: If data is insufficient
        """
        try:
            # Get the data length based on data type
            if hasattr(data, 'length'):
                data_length = data.length
            elif isinstance(data, pd.DataFrame):
                data_length = len(data)
            elif isinstance(data, dict) and self.symbol in data:
                data_length = len(data[self.symbol])
            elif hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()
                data_length = len(df)
            else:
                # If we can't determine length, assume it's sufficient
                # Actual validation will happen during indicator calculation
                return

            # We need at least slow_period data points
            slow_period = self.parameters['slow_period']
            if data_length < slow_period:
                raise StrategyError(
                    f"Insufficient data for strategy initialization. "
                    f"Need at least {slow_period} data points, got {data_length}.",
                    strategy_name=self.name,
                    parameters=self.parameters,
                    details={
                        "required_period": slow_period,
                        "available_data": data_length,
                        "symbol": self.symbol
                    }
                )

            logger.debug(f"Data sufficiency validated: {data_length} data points available for symbol {self.symbol}")

        except Exception as e:
            if isinstance(e, StrategyError):
                raise
            # If validation fails, log warning but continue
            # Actual indicator calculation will raise appropriate error
            logger.warning(f"Data sufficiency validation failed: {e}")

    def _get_symbol_dataframe(self, data: MarketData, symbol: str) -> pd.DataFrame:
        """
        Extract DataFrame for a specific symbol from market data.

        Args:
            data: MarketData object containing OHLCV data
            symbol: Symbol to extract data for

        Returns:
            DataFrame with OHLCV data for the symbol

        Raises:
            StrategyError: If data cannot be extracted
        """
        try:
            # Handle different data types
            if hasattr(data, 'get_symbol_data'):
                # Data has get_symbol_data method
                return data.get_symbol_data(symbol)
            elif hasattr(data, 'to_dataframe'):
                # MarketData object with to_dataframe method
                df = data.to_dataframe()
                # Check if the symbol matches
                if hasattr(data, 'symbol') and data.symbol == symbol:
                    return df
                else:
                    # If symbol doesn't match, still return the dataframe
                    # (assuming single symbol data)
                    return df
            elif isinstance(data, pd.DataFrame):
                # DataFrame directly - assume it's for the requested symbol
                return data
            elif isinstance(data, dict) and symbol in data:
                # Dictionary mapping symbols to dataframes
                return data[symbol]
            else:
                raise StrategyError(
                    f"Cannot extract data for symbol {symbol} from data object",
                    strategy_name=self.name,
                    parameters=self.parameters,
                    details={"data_type": type(data).__name__, "symbol": symbol}
                )
        except Exception as e:
            raise StrategyError(
                f"Failed to extract data for symbol {symbol}",
                strategy_name=self.name,
                parameters=self.parameters,
                details={"error": str(e), "symbol": symbol}
            )

    def get_parameters_summary(self) -> Dict[str, Any]:
        """
        Get detailed parameters summary.

        Returns:
            Dictionary with parameters summary
        """
        return {
            'strategy_name': self.name,
            'fast_period': self.parameters['fast_period'],
            'slow_period': self.parameters['slow_period'],
            'ma_type': self.parameters['ma_type'],
            'use_volume': self.parameters.get('use_volume', False),
            'min_crossover_strength': self.parameters.get('min_crossover_strength', 0.1),
            'require_confirmation': self.parameters.get('require_confirmation', True),
            'confirmation_periods': self.parameters.get('confirmation_periods', 2),
            'position_sizing': {
                'method': self.position_sizing.method,
                'risk_per_trade': self.position_sizing.risk_per_trade,
                'max_risk': self.position_sizing.max_risk,
                'max_position_size': self.position_sizing.max_position_size
            }
        }


# Convenience function for creating strategy
def create_moving_average_crossover_strategy(
    fast_period: int = 10,
    slow_period: int = 30,
    ma_type: str = 'sma',
    use_volume: bool = False,
    min_crossover_strength: float = 0.1,
    require_confirmation: bool = True,
    confirmation_periods: int = 2,
    position_sizing_params: Optional[Dict[str, Any]] = None
) -> MovingAverageCrossover:
    """
    Create a Moving Average Crossover strategy with specified parameters.

    Args:
        fast_period: Fast moving average period
        slow_period: Slow moving average period
        ma_type: Type of moving average ('sma' or 'ema')
        use_volume: Whether to use volume confirmation
        min_crossover_strength: Minimum crossover strength threshold
        require_confirmation: Whether to require confirmation periods
        confirmation_periods: Number of periods for confirmation
        position_sizing_params: Position sizing parameters

    Returns:
        Configured MovingAverageCrossover strategy
    """
    # Strategy parameters
    strategy_params = {
        'fast_period': fast_period,
        'slow_period': slow_period,
        'ma_type': ma_type,
        'use_volume': use_volume,
        'min_crossover_strength': min_crossover_strength,
        'require_confirmation': require_confirmation,
        'confirmation_periods': confirmation_periods
    }

    # Position sizing
    if position_sizing_params:
        position_sizing = PositionSizing(**position_sizing_params)
    else:
        position_sizing = None

    return MovingAverageCrossover(
        parameters=strategy_params,
        position_sizing=position_sizing
    )


# Export
__all__ = [
    'MovingAverageCrossover',
    'create_moving_average_crossover_strategy'
]
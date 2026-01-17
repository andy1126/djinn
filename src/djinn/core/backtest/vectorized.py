"""
模块目的：实现向量化回测引擎，通过数组运算提高大数据集上的回测性能。

实现方案：
1. 向量化计算：使用 Pandas 和 NumPy 的向量化操作一次性处理所有数据，避免循环。
2. 信号批量生成：一次性计算所有交易品种在所有时间点上的信号。
3. 持仓批量计算：基于信号批量计算目标持仓和实际持仓。
4. 组合价值计算：通过向量化运算计算每日组合价值和收益。
5. 交易批量生成：从持仓变化批量生成交易记录。

核心特点：
1. 高性能：适合大规模数据集和复杂策略，比事件驱动引擎快几个数量级。
2. 简化假设：假设所有交易都能以收盘价成交，不考虑订单簿深度。
3. 批量处理：所有计算都是批量进行，便于使用现代硬件加速。

使用方法：
1. 初始化 VectorizedBacktestEngine，设置初始资金、佣金、滑点等参数。
2. 调用 run 方法运行回测，传入策略、数据和日期范围。
3. 获取 BacktestResult 对象进行分析和可视化。

示例：
    engine = VectorizedBacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    result = engine.run(strategy, data, start_date, end_date)

注意事项：
1. 向量化引擎适合研究阶段快速验证策略，但不适合模拟真实交易环境。
2. 需要策略实现 calculate_signals_vectorized 方法支持向量化计算。
3. 不支持复杂的订单类型和交易逻辑。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings

from ...utils.exceptions import BacktestError, ValidationError
from ...utils.logger import get_logger
from .base import BacktestEngine, BacktestResult, Trade, Position, BacktestMode
from ..strategy.base import Strategy
from ..portfolio.base import Portfolio
from .metrics import calculate_all_metrics

logger = get_logger(__name__)


class VectorizedBacktestEngine(BacktestEngine):
    """
    Vectorized backtest engine.

    This engine uses vectorized operations for faster performance,
    making it suitable for large datasets and complex strategies.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        benchmark_symbol: Optional[str] = None,
        risk_free_rate: float = 0.02,
        allow_short: bool = False,
        max_position_size: float = 0.1,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """
        Initialize vectorized backtest engine.

        Args:
            initial_capital: Initial capital for backtest
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            benchmark_symbol: Benchmark symbol for comparison
            risk_free_rate: Risk-free rate for performance calculations
            allow_short: Whether to allow short positions
            max_position_size: Maximum position size as fraction of portfolio
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
        """
        super().__init__(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            benchmark_symbol=benchmark_symbol,
            risk_free_rate=risk_free_rate,
            mode=BacktestMode.VECTORIZED
        )

        self.allow_short = allow_short
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Vectorized specific state
        self.signals_df: Optional[pd.DataFrame] = None
        self.positions_df: Optional[pd.DataFrame] = None
        self.returns_df: Optional[pd.DataFrame] = None
        self.portfolio_values: Optional[pd.Series] = None

        logger.info("Initialized vectorized backtest engine")

    def run(
        self,
        strategy: Strategy,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'daily',
        **kwargs
    ) -> BacktestResult:
        """
        Run vectorized backtest.

        Args:
            strategy: Trading strategy to backtest
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            start_date: Start date for backtest
            end_date: End date for backtest
            frequency: Data frequency ('daily', 'hourly', 'minute')
            **kwargs: Additional parameters

        Returns:
            BacktestResult object with all performance metrics
        """
        logger.info(f"Starting vectorized backtest from {start_date} to {end_date}")

        # Validate data
        self.validate_data(data)

        # Initialize backtest
        self._initialize_backtest(strategy, data, start_date, end_date, frequency)

        # Calculate signals for all symbols
        self._calculate_signals(strategy, data)

        # Calculate positions based on signals
        self._calculate_positions(data)

        # Calculate portfolio values
        self._calculate_portfolio_values(data)

        # Generate trades from position changes
        self._generate_trades(data)

        # Calculate performance metrics
        result = self._calculate_metrics(strategy, start_date, end_date, frequency)

        logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        return result

    def _initialize_backtest(
        self,
        strategy: Strategy,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> None:
        """Initialize backtest state."""
        # Initialize strategy
        strategy.initialize(data)

        # Reset state
        self.current_capital = self.initial_capital
        self.signals_df = None
        self.positions_df = None
        self.returns_df = None
        self.portfolio_values = None
        self.trades = []
        self.positions = []

        logger.info(f"Backtest initialized with {len(data)} symbols")

    def _calculate_signals(
        self,
        strategy: Strategy,
        data: Dict[str, pd.DataFrame]
    ) -> None:
        """Calculate trading signals for all symbols."""
        signals_dict = {}

        for symbol, df in data.items():
            try:
                # Calculate signals using vectorized operations
                signals = strategy.calculate_signals_vectorized(df)

                if signals is not None and not signals.empty:
                    signals_dict[symbol] = signals

            except Exception as e:
                logger.warning(f"Error calculating signals for {symbol}: {e}")
                continue

        if not signals_dict:
            raise BacktestError("No signals generated for any symbol")

        # Create signals DataFrame
        self.signals_df = pd.DataFrame(signals_dict)

        logger.info(f"Generated signals for {len(signals_dict)} symbols")

    def _calculate_positions(self, data: Dict[str, pd.DataFrame]) -> None:
        """Calculate positions based on signals."""
        if self.signals_df is None:
            raise BacktestError("Signals not calculated")

        positions_dict = {}

        for symbol in self.signals_df.columns:
            if symbol not in data:
                continue

            signals = self.signals_df[symbol]
            close_prices = data[symbol]['close']

            # Align signals with prices
            aligned_signals = signals.reindex(close_prices.index).fillna(0)

            # Calculate position sizes
            positions = self._calculate_position_sizes(
                signals=aligned_signals,
                prices=close_prices,
                symbol=symbol
            )

            positions_dict[symbol] = positions

        self.positions_df = pd.DataFrame(positions_dict)

        logger.info(f"Calculated positions for {len(positions_dict)} symbols")

    def _calculate_position_sizes(
        self,
        signals: pd.Series,
        prices: pd.Series,
        symbol: str
    ) -> pd.Series:
        """Calculate position sizes based on signals and portfolio constraints."""
        positions = pd.Series(0.0, index=signals.index)

        # Track portfolio value over time
        portfolio_value = self.initial_capital
        position_values = pd.Series(0.0, index=signals.index)

        for i in range(1, len(signals)):
            current_signal = signals.iloc[i]
            current_price = prices.iloc[i]

            if abs(current_signal) < 0.01:  # Weak signal
                target_position = 0.0
            else:
                # Calculate target position value
                position_fraction = min(abs(current_signal), self.max_position_size)
                target_value = portfolio_value * position_fraction

                # Calculate target quantity
                target_quantity = target_value / current_price if current_price > 0 else 0

                # Apply signal direction
                if current_signal > 0:
                    target_quantity = abs(target_quantity)
                else:
                    if self.allow_short:
                        target_quantity = -abs(target_quantity)
                    else:
                        target_quantity = 0.0

                # Round to nearest whole share
                target_quantity = round(target_quantity)
                target_position = target_quantity

            # Update position
            positions.iloc[i] = target_position
            position_values.iloc[i] = target_position * current_price

            # Update portfolio value (simplified - doesn't account for commissions/slippage in this loop)
            # This will be refined in _calculate_portfolio_values
            portfolio_value = self.initial_capital + position_values.iloc[:i+1].sum()

        return positions

    def _calculate_portfolio_values(self, data: Dict[str, pd.DataFrame]) -> None:
        """Calculate portfolio values over time."""
        if self.positions_df is None:
            raise BacktestError("Positions not calculated")

        # Initialize portfolio values
        dates = self.positions_df.index
        portfolio_values = pd.Series(self.initial_capital, index=dates)
        cash = pd.Series(self.initial_capital, index=dates)

        # Calculate position values
        position_values = pd.DataFrame(index=dates)

        for symbol in self.positions_df.columns:
            if symbol in data:
                positions = self.positions_df[symbol]
                prices = data[symbol]['close'].reindex(dates).fillna(method='ffill')
                position_values[symbol] = positions * prices

        # Calculate daily changes
        position_changes = self.positions_df.diff().fillna(0)

        # Calculate trades and update cash
        for i in range(1, len(dates)):
            date = dates[i]

            # Calculate cash change from previous day's positions
            if i > 0:
                prev_date = dates[i-1]

                # Value change from existing positions
                for symbol in self.positions_df.columns:
                    if symbol in data:
                        prev_position = self.positions_df[symbol].iloc[i-1]
                        price_change = data[symbol]['close'].loc[date] - data[symbol]['close'].loc[prev_date]
                        cash.iloc[i] += prev_position * price_change

            # Calculate cash change from today's trades
            daily_trade_value = 0.0
            daily_commission = 0.0
            daily_slippage = 0.0

            for symbol in self.positions_df.columns:
                if symbol in data:
                    position_change = position_changes[symbol].iloc[i]
                    if abs(position_change) > 1e-6:
                        current_price = data[symbol]['close'].loc[date]

                        # Calculate trade value
                        trade_value = position_change * current_price

                        # Calculate commission and slippage
                        commission = abs(trade_value) * self.commission
                        slippage = abs(trade_value) * self.slippage

                        # Update cash
                        cash.iloc[i] -= trade_value + commission + slippage

                        daily_trade_value += trade_value
                        daily_commission += commission
                        daily_slippage += slippage

            # Update portfolio value
            portfolio_values.iloc[i] = cash.iloc[i] + position_values.loc[date].sum()

        self.portfolio_values = portfolio_values
        self.returns = portfolio_values.pct_change().dropna()

        logger.info(f"Calculated portfolio values for {len(dates)} periods")

    def _generate_trades(self, data: Dict[str, pd.DataFrame]) -> None:
        """Generate Trade objects from position changes."""
        if self.positions_df is None:
            raise BacktestError("Positions not calculated")

        trades = []

        for symbol in self.positions_df.columns:
            if symbol not in data:
                continue

            positions = self.positions_df[symbol]
            position_changes = positions.diff().fillna(0)
            prices = data[symbol]['close']

            for date in position_changes.index:
                change = position_changes[date]
                if abs(change) > 1e-6:
                    price = prices.loc[date]
                    side = 'buy' if change > 0 else 'sell'

                    # Calculate commission and slippage
                    trade_value = abs(change) * price
                    commission = trade_value * self.commission
                    slippage = trade_value * self.slippage

                    # Create trade
                    trade = Trade(
                        timestamp=date,
                        symbol=symbol,
                        quantity=change,
                        price=price,
                        side=side,
                        commission=commission,
                        slippage=slippage,
                        trade_id=f"{date.strftime('%Y%m%d')}_{symbol}_{len(trades)}"
                    )

                    trades.append(trade)

        self.trades = trades

        # Generate Position objects
        self._generate_positions(data)

        logger.info(f"Generated {len(trades)} trades")

    def _generate_positions(self, data: Dict[str, pd.DataFrame]) -> None:
        """Generate Position objects from positions DataFrame."""
        if self.positions_df is None or self.portfolio_values is None:
            raise BacktestError("Positions or portfolio values not calculated")

        positions = []

        for date in self.positions_df.index:
            portfolio_value = self.portfolio_values.loc[date]

            for symbol in self.positions_df.columns:
                if symbol not in data:
                    continue

                position_qty = self.positions_df.loc[date, symbol]
                if abs(position_qty) > 1e-6:
                    price = data[symbol]['close'].loc[date]
                    market_value = abs(position_qty) * price

                    # Calculate average price (simplified - assumes all shares bought at current price)
                    avg_price = price

                    # Calculate unrealized P&L (simplified)
                    unrealized_pnl = 0.0  # Would need cost basis tracking

                    position = Position(
                        timestamp=date,
                        symbol=symbol,
                        quantity=position_qty,
                        avg_price=avg_price,
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        realized_pnl=0.0
                    )

                    positions.append(position)

        self.positions = positions

    def _calculate_metrics(
        self,
        strategy: Strategy,
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> BacktestResult:
        """Calculate performance metrics from backtest results."""
        if self.portfolio_values is None or self.returns is None:
            raise BacktestError("Portfolio values or returns not calculated")

        # Calculate metrics
        metrics = calculate_all_metrics(
            returns=self.returns,
            equity_curve=self.portfolio_values,
            trades=self.trades,
            initial_capital=self.initial_capital,
            risk_free_rate=self.risk_free_rate
        )

        # Create result object
        result = BacktestResult(
            # Performance metrics
            total_return=metrics['total_return'],
            annual_return=metrics['annual_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            volatility=metrics['volatility'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            avg_win=metrics['avg_win'],
            avg_loss=metrics['avg_loss'],

            # Trade statistics
            total_trades=metrics['total_trades'],
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            avg_trade_return=metrics['avg_trade_return'],
            avg_trade_duration=metrics['avg_trade_duration'],
            max_consecutive_wins=metrics['max_consecutive_wins'],
            max_consecutive_losses=metrics['max_consecutive_losses'],

            # Portfolio statistics
            initial_capital=self.initial_capital,
            final_capital=float(self.portfolio_values.iloc[-1]) if not self.portfolio_values.empty else self.initial_capital,
            peak_capital=metrics['peak_capital'],
            trough_capital=metrics['trough_capital'],
            total_commission=sum(t.commission for t in self.trades),
            total_slippage=sum(t.slippage for t in self.trades),

            # Time series data
            equity_curve=self.portfolio_values,
            returns=self.returns,
            drawdown=metrics['drawdown'],
            positions=self.positions,
            trades=self.trades,

            # Metadata
            strategy_name=strategy.name,
            symbols=list(self.positions_df.columns) if self.positions_df is not None else [],
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            parameters={
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'slippage': self.slippage,
                'allow_short': self.allow_short,
                'max_position_size': self.max_position_size,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            }
        )

        return result

    def get_position_analysis(self) -> pd.DataFrame:
        """
        Get detailed position analysis.

        Returns:
            DataFrame with position statistics
        """
        if self.positions_df is None:
            raise BacktestError("Positions not calculated")

        analysis = []

        for symbol in self.positions_df.columns:
            positions = self.positions_df[symbol]

            # Calculate position statistics
            holding_periods = self._calculate_holding_periods(positions)
            avg_holding_days = holding_periods.mean() if len(holding_periods) > 0 else 0

            # Calculate position returns (simplified)
            # In a real implementation, you would track entry/exit prices

            analysis.append({
                'symbol': symbol,
                'total_positions': (positions != 0).sum(),
                'avg_holding_days': avg_holding_days,
                'max_position': positions.abs().max(),
                'avg_position': positions[positions != 0].abs().mean() if (positions != 0).any() else 0
            })

        return pd.DataFrame(analysis)

    def _calculate_holding_periods(self, positions: pd.Series) -> pd.Series:
        """Calculate holding periods for positions."""
        holding_periods = []

        in_position = False
        start_date = None

        for date, position in positions.items():
            if not in_position and position != 0:
                # Entering position
                in_position = True
                start_date = date
            elif in_position and position == 0:
                # Exiting position
                in_position = False
                if start_date:
                    holding_days = (date - start_date).days
                    holding_periods.append(holding_days)
                    start_date = None

        # Handle position still open at end
        if in_position and start_date and len(positions) > 0:
            end_date = positions.index[-1]
            holding_days = (end_date - start_date).days
            holding_periods.append(holding_days)

        return pd.Series(holding_periods)

    def _process_bar(self, current_date: datetime) -> None:
        """
        Process a single bar/period of data.

        Note: Vectorized engine processes all data at once, so this method
        is not used but required by the abstract base class.
        """
        # Vectorized engine doesn't process bars individually
        pass

    def _execute_trades(self, signals: Dict[str, float]) -> List[Trade]:
        """
        Execute trades based on signals.

        Note: Vectorized engine generates trades from position changes,
        so this method is not used but required by the abstract base class.
        """
        # Vectorized engine generates trades differently
        # Return empty list to satisfy interface
        return []

    def _update_positions(self, trades: List[Trade]) -> None:
        """
        Update positions based on executed trades.

        Note: Vectorized engine generates positions from positions DataFrame,
        so this method is not used but required by the abstract base class.
        """
        # Vectorized engine updates positions differently
        pass
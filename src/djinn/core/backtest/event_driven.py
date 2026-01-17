"""
模块目的：实现事件驱动回测引擎，通过逐笔数据处理模拟真实交易环境。

实现方案：
1. 逐笔处理：按时间顺序逐个交易日处理数据，模拟真实交易流程。
2. 实时信号生成：每个交易日根据最新数据生成交易信号。
3. 订单执行：模拟订单执行过程，包括价格获取、佣金和滑点计算。
4. 持仓管理：实时跟踪持仓状态，计算未实现盈亏和已实现盈亏。
5. 风控检查：支持止损、止盈等风控条件的实时检查。

核心特点：
1. 高真实性：模拟真实交易环境，考虑订单执行、资金约束等实际因素。
2. 支持复杂逻辑：支持止损、止盈、仓位限制等高级交易功能。
3. 灵活性强：可以模拟各种订单类型和交易规则。
4. 资源消耗大：相比向量化引擎，计算速度较慢。

使用方法：
1. 初始化 EventDrivenBacktestEngine，设置初始资金、佣金、滑点、止损止盈等参数。
2. 调用 run 方法运行回测，传入策略、数据和日期范围。
3. 获取 BacktestResult 对象进行分析和可视化。

示例：
    engine = EventDrivenBacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        stop_loss=0.05,
        take_profit=0.10
    )
    result = engine.run(strategy, data, start_date, end_date)

适用场景：
1. 需要高真实性模拟的交易策略验证。
2. 包含复杂订单逻辑和风控规则的策略。
3. 需要评估交易成本和滑点影响的策略。
4. 需要模拟真实资金管理和仓位控制的策略。
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from ...utils.exceptions import BacktestError
from ...utils.logger import get_logger
from ...data.market_data import MarketData, MarketDataType, AdjustmentType
from .base import BacktestEngine, BacktestResult, Trade, Position, BacktestMode
from ..strategy.base import Strategy
from .metrics import calculate_all_metrics

logger = get_logger(__name__)


class EventDrivenBacktestEngine(BacktestEngine):
    """
    Event-driven backtest engine.

    This engine processes data bar-by-bar, simulating real-time trading
    with order execution, position updates, and portfolio rebalancing.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        benchmark_symbol: Optional[str] = None,
        risk_free_rate: float = 0.02,
        allow_short: bool = False,
        max_position_size: float = 0.1,  # 10% of portfolio per position
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """
        Initialize event-driven backtest engine.

        Args:
            initial_capital: Initial capital for backtest
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            benchmark_symbol: Benchmark symbol for comparison
            risk_free_rate: Risk-free rate for performance calculations
            allow_short: Whether to allow short positions
            max_position_size: Maximum position size as fraction of portfolio
            stop_loss: Stop loss percentage (e.g., 0.05 for 5%)
            take_profit: Take profit percentage (e.g., 0.10 for 10%)
        """
        super().__init__(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            benchmark_symbol=benchmark_symbol,
            risk_free_rate=risk_free_rate,
            mode=BacktestMode.EVENT_DRIVEN
        )

        self.allow_short = allow_short
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Event-driven specific state
        self.current_prices: Dict[str, float] = {}
        self.open_positions: Dict[str, Position] = {}
        self.pending_orders: List[Dict] = []
        self.daily_returns = pd.Series(dtype=float)
        self.daily_equity = pd.Series(dtype=float)

        # Performance tracking
        self.peak_equity = initial_capital
        self.trough_equity = initial_capital
        self.max_drawdown = 0.0

        logger.info("Initialized event-driven backtest engine")

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
        Run event-driven backtest.

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
        logger.info(f"Starting event-driven backtest from {start_date} to {end_date}")

        # Validate data
        self.validate_data(data)

        # Initialize backtest
        self._initialize_backtest(strategy, data, start_date, end_date, frequency)

        # Get trading days
        trading_days = self._get_trading_days(data, start_date, end_date, frequency)

        # Main backtest loop
        self.is_running = True
        for i, current_date in enumerate(trading_days):
            self.current_date = current_date

            # Update current prices
            self._update_current_prices(current_date, data)

            # Check stop loss and take profit
            self._check_position_triggers()

            # Process strategy signals
            signals = self._get_strategy_signals(strategy, current_date, data)

            # Execute trades
            trades = self._execute_trades(signals)

            # Update positions
            self._update_positions(trades)

            # Update portfolio value
            self._update_portfolio_value(current_date)

            # Update performance metrics
            self._update_performance_metrics(current_date)

            # Log progress
            if i % max(1, len(trading_days) // 10) == 0:
                progress = (i + 1) / len(trading_days) * 100
                logger.info(f"Backtest progress: {progress:.1f}%")

        self.is_running = False

        # Calculate final metrics
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
        # Create MarketData object for strategy initialization
        # For simplicity, use the first symbol's data
        if not data:
            raise BacktestError("No data provided for backtest")

        first_symbol = next(iter(data))
        first_data = data[first_symbol]

        # Create MarketData object
        market_data = MarketData(
            symbol=first_symbol,
            data_type=MarketDataType.OHLCV,
            data=first_data,
            interval="1d",
            adjustment=AdjustmentType.ADJ
        )

        # Initialize strategy
        strategy.initialize(market_data)

        # Reset state
        self.current_capital = self.initial_capital
        self.current_prices = {}
        self.open_positions = {}
        self.pending_orders = []
        self.trades = []
        self.positions = []
        self.daily_returns = pd.Series(dtype=float)
        self.daily_equity = pd.Series(dtype=float)

        # Initialize equity curve
        self.equity_curve = pd.Series([self.initial_capital], index=[start_date])
        self.returns = pd.Series(dtype=float)

        logger.info(f"Backtest initialized with {len(data)} symbols")

    def _get_trading_days(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> List[datetime]:
        """Get list of trading days for backtest."""
        # Get common trading days across all symbols
        all_dates = set()
        for symbol, df in data.items():
            mask = (df.index >= start_date) & (df.index <= end_date)
            symbol_dates = set(df[mask].index)
            all_dates.update(symbol_dates)

        # Sort dates
        trading_days = sorted(list(all_dates))

        if not trading_days:
            raise BacktestError("No trading days found in the specified date range")

        logger.info(f"Found {len(trading_days)} trading days")
        return trading_days

    def _update_current_prices(
        self,
        current_date: datetime,
        data: Dict[str, pd.DataFrame]
    ) -> None:
        """Update current prices for all symbols."""
        self.current_prices = {}
        for symbol, df in data.items():
            if current_date in df.index:
                # Use close price for valuation
                self.current_prices[symbol] = df.loc[current_date, 'close']
            else:
                # If no data for this date, use last available price
                mask = df.index <= current_date
                if mask.any():
                    last_date = df[mask].index[-1]
                    self.current_prices[symbol] = df.loc[last_date, 'close']
                else:
                    # No data available yet
                    self.current_prices[symbol] = np.nan

    def _get_strategy_signals(
        self,
        strategy: Strategy,
        current_date: datetime,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Get trading signals from strategy."""
        signals = {}

        for symbol in data.keys():
            try:
                # Get historical data up to current date
                mask = data[symbol].index <= current_date
                historical_data = data[symbol][mask]

                if len(historical_data) < strategy.min_data_points:
                    continue

                # Calculate signal
                signal = strategy.calculate_signal(
                    symbol=symbol,
                    data=historical_data,
                    current_date=current_date
                )

                if signal is not None:
                    signals[symbol] = signal

            except Exception as e:
                logger.warning(f"Error calculating signal for {symbol}: {e}")
                continue

        return signals

    def _execute_trades(self, signals: Dict[str, float]) -> List[Trade]:
        """Execute trades based on signals."""
        trades = []

        for symbol, signal in signals.items():
            if symbol not in self.current_prices or np.isnan(self.current_prices[symbol]):
                continue

            current_price = self.current_prices[symbol]

            # Get current position
            current_position = self.open_positions.get(symbol)
            current_quantity = current_position.quantity if current_position else 0

            # Calculate target quantity
            target_quantity = self._calculate_target_quantity(
                symbol=symbol,
                signal=signal,
                current_price=current_price,
                current_quantity=current_quantity
            )

            # Check if trade is needed
            quantity_change = target_quantity - current_quantity
            if abs(quantity_change) < 1e-6:  # No change needed
                continue

            # Create trade
            side = 'buy' if quantity_change > 0 else 'sell'
            quantity = abs(quantity_change)

            # Calculate commission and slippage
            commission = self.calculate_commission(quantity, current_price, side)
            slippage = self.calculate_slippage(quantity, current_price, side)

            # Adjust price for slippage
            execution_price = current_price
            if side == 'buy':
                execution_price += slippage / quantity
            else:
                execution_price -= slippage / quantity

            # Check if we have enough capital for buy
            if side == 'buy':
                trade_cost = quantity * execution_price + commission
                if trade_cost > self.current_capital:
                    logger.warning(
                        f"Insufficient capital for {symbol} buy: "
                        f"needed {trade_cost:.2f}, have {self.current_capital:.2f}"
                    )
                    continue

            # Create trade object
            trade = Trade(
                timestamp=self.current_date,
                symbol=symbol,
                quantity=quantity_change,
                price=execution_price,
                side=side,
                commission=commission,
                slippage=slippage,
                trade_id=f"{self.current_date.strftime('%Y%m%d')}_{symbol}_{len(trades)}"
            )

            trades.append(trade)

            # Update capital
            if side == 'buy':
                self.current_capital -= (quantity * execution_price + commission)
            else:
                self.current_capital += (quantity * execution_price - commission)

        return trades

    def _calculate_target_quantity(
        self,
        symbol: str,
        signal: float,
        current_price: float,
        current_quantity: float
    ) -> float:
        """Calculate target position quantity based on signal."""
        if abs(signal) < 0.01:  # Weak signal, close position
            return 0.0

        # Calculate position size based on signal strength and max position size
        position_fraction = min(abs(signal), self.max_position_size)
        target_value = self.current_capital * position_fraction

        # Calculate target quantity
        target_quantity = target_value / current_price

        # Apply signal direction
        if signal > 0:
            target_quantity = abs(target_quantity)
        else:
            target_quantity = -abs(target_quantity)

        # Round to nearest whole share
        target_quantity = round(target_quantity)

        return target_quantity

    def _update_positions(self, trades: List[Trade]) -> None:
        """Update positions based on executed trades."""
        for trade in trades:
            symbol = trade.symbol
            current_position = self.open_positions.get(symbol)

            if current_position is None:
                # Create new position
                if trade.quantity != 0:
                    new_position = Position(
                        timestamp=trade.timestamp,
                        symbol=symbol,
                        quantity=trade.quantity,
                        avg_price=trade.price,
                        market_value=abs(trade.quantity) * trade.price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0
                    )
                    self.open_positions[symbol] = new_position
                    self.positions.append(new_position)
            else:
                # Update existing position
                if current_position.quantity * trade.quantity >= 0:
                    # Adding to position
                    total_quantity = current_position.quantity + trade.quantity
                    total_cost = (current_position.quantity * current_position.avg_price +
                                 trade.quantity * trade.price)
                    new_avg_price = total_cost / total_quantity if total_quantity != 0 else 0

                    current_position.quantity = total_quantity
                    current_position.avg_price = new_avg_price
                    current_position.timestamp = trade.timestamp
                else:
                    # Reducing or reversing position
                    # Calculate realized P&L
                    realized_pnl = (trade.price - current_position.avg_price) * abs(trade.quantity)
                    if trade.quantity < 0 and current_position.quantity > 0:
                        realized_pnl = -realized_pnl  # Selling long position

                    current_position.realized_pnl += realized_pnl
                    current_position.quantity += trade.quantity
                    current_position.timestamp = trade.timestamp

                    # If position is closed, remove it
                    if abs(current_position.quantity) < 1e-6:
                        del self.open_positions[symbol]

            # Add trade to history
            self.trades.append(trade)

    def _check_position_triggers(self) -> None:
        """Check stop loss and take profit triggers."""
        if not self.stop_loss and not self.take_profit:
            return

        positions_to_close = []

        for symbol, position in self.open_positions.items():
            if symbol not in self.current_prices:
                continue

            current_price = self.current_prices[symbol]
            if np.isnan(current_price):
                continue

            # Calculate unrealized P&L percentage
            cost_basis = position.avg_price
            if cost_basis == 0:
                continue

            pnl_pct = (current_price - cost_basis) / cost_basis
            if position.quantity < 0:  # Short position
                pnl_pct = -pnl_pct

            # Check stop loss
            if self.stop_loss and pnl_pct < -self.stop_loss:
                logger.info(
                    f"Stop loss triggered for {symbol}: "
                    f"PNL={pnl_pct:.2%}, threshold={-self.stop_loss:.2%}"
                )
                positions_to_close.append(symbol)

            # Check take profit
            elif self.take_profit and pnl_pct > self.take_profit:
                logger.info(
                    f"Take profit triggered for {symbol}: "
                    f"PNL={pnl_pct:.2%}, threshold={self.take_profit:.2%}"
                )
                positions_to_close.append(symbol)

        # Close positions
        for symbol in positions_to_close:
            position = self.open_positions[symbol]
            signal = 0.0  # Signal to close position
            trades = self._execute_trades({symbol: signal})
            self._update_positions(trades)

    def _update_portfolio_value(self, current_date: datetime) -> None:
        """Update total portfolio value."""
        portfolio_value = self.current_capital

        for symbol, position in self.open_positions.items():
            if symbol in self.current_prices and not np.isnan(self.current_prices[symbol]):
                position.market_value = abs(position.quantity) * self.current_prices[symbol]

                # Calculate unrealized P&L
                if position.quantity != 0:
                    price_change = self.current_prices[symbol] - position.avg_price
                    position.unrealized_pnl = position.quantity * price_change
                    portfolio_value += position.market_value

        # Update equity curve
        self.equity_curve[current_date] = portfolio_value

    def _update_performance_metrics(self, current_date: datetime) -> None:
        """Update daily performance metrics."""
        if len(self.equity_curve) < 2:
            return

        # Calculate daily return
        prev_equity = self.equity_curve.iloc[-2] if len(self.equity_curve) > 1 else self.initial_capital
        current_equity = self.equity_curve[current_date]

        if prev_equity > 0:
            daily_return = (current_equity - prev_equity) / prev_equity
            self.daily_returns[current_date] = daily_return

        # Update peak and trough
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.trough_equity = current_equity
        elif current_equity < self.trough_equity:
            self.trough_equity = current_equity

        # Calculate drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)

    def _calculate_metrics(
        self,
        strategy: Strategy,
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> BacktestResult:
        """Calculate performance metrics from backtest results."""
        # Ensure equity curve is sorted
        self.equity_curve = self.equity_curve.sort_index()

        # Calculate returns
        if len(self.equity_curve) > 1:
            self.returns = self.equity_curve.pct_change().dropna()
        else:
            self.returns = pd.Series(dtype=float)

        # Calculate metrics
        metrics = calculate_all_metrics(
            returns=self.returns,
            equity_curve=self.equity_curve,
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
            final_capital=float(self.equity_curve.iloc[-1]) if not self.equity_curve.empty else self.initial_capital,
            peak_capital=metrics['peak_capital'],
            trough_capital=metrics['trough_capital'],
            total_commission=sum(t.commission for t in self.trades),
            total_slippage=sum(t.slippage for t in self.trades),

            # Time series data
            equity_curve=self.equity_curve,
            returns=self.returns,
            drawdown=metrics['drawdown'],
            positions=self.positions,
            trades=self.trades,

            # Metadata
            strategy_name=strategy.name,
            symbols=list(self.current_prices.keys()),
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

    def _process_bar(self, current_date: datetime) -> None:
        """Process a single bar/period of data."""
        # This engine already processes bars in the run method
        # This method is required by the abstract base class but not used here
        pass
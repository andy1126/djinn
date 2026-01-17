"""
模块目的：提供回测结果性能指标的计算功能，支持全面的策略评估和风险分析。

实现方案：
1. 综合指标计算：calculate_all_metrics 函数整合了收益、风险、交易和回撤四大类指标的计算。
2. 模块化设计：将指标计算分解为多个内部函数，便于维护和扩展。
3. 基准对比功能：calculate_benchmark_metrics 函数支持策略与基准的对比分析。
4. 便捷工具类：BacktestMetrics 类提供面向对象的指标计算和汇总功能。

主要指标类别：
1. 收益指标：总收益率、年化收益率、CAGR（复合年增长率）
2. 风险指标：波动率、夏普比率、索提诺比率、卡玛比率
3. 交易指标：胜率、盈亏比、平均盈亏、最大连续盈亏
4. 回撤指标：最大回撤、平均回撤、回撤持续时间
5. 基准对比指标：Alpha、Beta、信息比率、跟踪误差

使用方法：
1. 直接使用 calculate_all_metrics 函数计算所有指标。
2. 使用 BacktestMetrics 类进行面向对象的指标计算和管理。
3. 通过 calculate_benchmark_metrics 函数进行策略与基准的对比分析。

示例：
    # 计算所有指标
    metrics = calculate_all_metrics(returns, equity_curve, trades, initial_capital)

    # 使用工具类
    metrics_calculator = BacktestMetrics(returns, equity_curve, trades, initial_capital)
    metrics_calculator.calculate()
    summary = metrics_calculator.get_summary()
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
import warnings

from ...utils.exceptions import PerformanceError
from ...utils.logger import get_logger
from .base import Trade

logger = get_logger(__name__)


def calculate_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    trades: List[Trade],
    initial_capital: float,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252
) -> Dict[str, Any]:
    """
    计算所有性能指标。

    目的：一站式计算回测结果的所有关键性能指标，包括收益、风险、交易和回撤四大类指标。

    实现方案：
    1. 输入验证：检查数据是否为空，确保计算可行性
    2. 模块化计算：调用四个专用函数分别计算不同类型的指标
    3. 结果整合：将所有指标合并到统一的字典中返回
    4. 异常处理：处理数据异常和计算错误情况

    参数说明：
        returns: 收益率序列，每日或每期收益率
        equity_curve: 净值曲线序列，组合净值随时间变化
        trades: 交易记录列表，包含所有Trade对象
        initial_capital: 初始资金，用于计算收益率和收益指标
        risk_free_rate: 无风险利率，年化，用于计算风险调整收益
        trading_days_per_year: 年化交易日数，默认252天

    返回值：
        Dict[str, Any]: 包含所有计算指标的字典，键为指标名称，值为指标数值

    指标分类：
        1. 收益指标：总收益率、年化收益率、CAGR等
        2. 风险指标：波动率、夏普比率、索提诺比率等
        3. 交易指标：胜率、盈亏比、平均交易收益率等
        4. 回撤指标：最大回撤、平均回撤、回撤持续时间等

    使用方法：
        from djinn.core.backtest.metrics import calculate_all_metrics
        import pandas as pd

        # 准备数据
        returns = pd.Series([0.01, -0.02, 0.03, ...])
        equity_curve = pd.Series([100000, 101000, 98980, ...])
        trades = [...]  # Trade对象列表

        # 计算所有指标
        metrics = calculate_all_metrics(
            returns=returns,
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=100000,
            risk_free_rate=0.02,
            trading_days_per_year=252
        )

        # 访问具体指标
        total_return = metrics['total_return']
        sharpe_ratio = metrics['sharpe_ratio']
        max_drawdown = metrics['max_drawdown']
    """
    if returns.empty or equity_curve.empty:
        raise PerformanceError("Cannot calculate metrics with empty data")

    metrics = {}

    # Basic return metrics
    metrics.update(_calculate_return_metrics(
        returns, equity_curve, initial_capital, risk_free_rate, trading_days_per_year
    ))

    # Risk metrics
    metrics.update(_calculate_risk_metrics(returns, risk_free_rate, trading_days_per_year))

    # Trade metrics
    metrics.update(_calculate_trade_metrics(trades, equity_curve))

    # Drawdown metrics
    metrics.update(_calculate_drawdown_metrics(equity_curve))

    return metrics


def _calculate_return_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    initial_capital: float,
    risk_free_rate: float,
    trading_days_per_year: int
) -> Dict[str, float]:
    """
    计算收益相关指标。

    目的：计算与投资回报相关的核心指标，评估策略的盈利能力。

    实现方案：
    1. 总收益率计算：基于净值曲线和初始资金的简单收益率
    2. 年化收益率计算：考虑时间因素的年度化收益率
    3. 复合年增长率（CAGR）计算：考虑复利效应的年化收益率

    参数说明：
        returns: 收益率序列，用于计算时间长度和收益分布
        equity_curve: 净值曲线序列，用于计算总收益率和CAGR
        initial_capital: 初始资金，作为收益率计算的基准
        risk_free_rate: 无风险利率（此函数中未直接使用，为接口一致性保留）
        trading_days_per_year: 年化交易日数（此函数中未直接使用，为接口一致性保留）

    返回值：
        Dict[str, float]: 包含收益指标的字典，包括total_return、annual_return、cagr

    计算公式：
        1. 总收益率 = (最终净值 - 初始资金) / 初始资金
        2. 年化收益率 = (1 + 总收益率)^(1/年数) - 1
        3. CAGR = (最终净值/初始资金)^(1/年数) - 1

    注意事项：
        1. 处理空数据或单点数据情况，返回零值
        2. 使用实际日历天数计算年化因子（365.25天/年）
        3. 年化收益率和CAGR在数学上等价，但计算方式略有不同
    """
    metrics = {}

    # Total return
    if len(equity_curve) > 1:
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    else:
        total_return = 0.0
    metrics['total_return'] = total_return

    # Annualized return
    if len(returns) > 0:
        days = (returns.index[-1] - returns.index[0]).days
        if days > 0:
            years = days / 365.25
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = total_return
    else:
        annual_return = 0.0
    metrics['annual_return'] = annual_return

    # CAGR (Compound Annual Growth Rate)
    if len(equity_curve) > 1:
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            years = days / 365.25
            cagr = (equity_curve.iloc[-1] / initial_capital) ** (1 / years) - 1
        else:
            cagr = total_return
    else:
        cagr = 0.0
    metrics['cagr'] = cagr

    return metrics


def _calculate_risk_metrics(
    returns: pd.Series,
    risk_free_rate: float,
    trading_days_per_year: int
) -> Dict[str, float]:
    """
    计算风险相关指标。

    目的：评估策略的风险特征和风险调整后收益，提供全面的风险分析。

    实现方案：
    1. 波动率计算：收益率序列的年化标准差，衡量总体风险
    2. 夏普比率计算：超额收益与波动率的比率，衡量风险调整后收益
    3. 索提诺比率计算：使用下行偏差替代总体波动率，关注下行风险
    4. 其他风险指标：为基准相关指标预留位置（需要基准数据）

    参数说明：
        returns: 收益率序列，用于计算波动率和各种比率
        risk_free_rate: 无风险利率，年化，用于计算超额收益
        trading_days_per_year: 年化交易日数，用于年化计算

    返回值：
        Dict[str, float]: 包含风险指标的字典，包括volatility、sharpe_ratio、sortino_ratio等

    计算公式：
        1. 年化波动率 = 收益率标准差 × sqrt(年化交易日数)
        2. 夏普比率 = (平均超额收益 / 超额收益标准差) × sqrt(年化交易日数)
        3. 索提诺比率 = (年化收益 - 无风险利率) / 下行偏差
        4. 下行偏差 = 负收益的标准差 × sqrt(年化交易日数)

    注意事项：
        1. 处理空收益率序列，返回零值指标
        2. 夏普比率使用日度无风险利率调整
        3. 索提诺比率仅考虑负收益（下行风险）
        4. 卡玛比率、信息比率等需要额外数据，此处返回占位值
    """
    metrics = {}

    if returns.empty:
        metrics.update({
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'information_ratio': 0.0,
            'beta': 0.0,
            'alpha': 0.0,
            'treynor_ratio': 0.0
        })
        return metrics

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(trading_days_per_year)
    metrics['volatility'] = volatility

    # Sharpe Ratio
    excess_returns = returns - (risk_free_rate / trading_days_per_year)
    if excess_returns.std() > 0:
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days_per_year)
    else:
        sharpe_ratio = 0.0
    metrics['sharpe_ratio'] = sharpe_ratio

    # Sortino Ratio (uses downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = downside_returns.std() * np.sqrt(trading_days_per_year)
        if downside_deviation > 0:
            sortino_ratio = (returns.mean() * trading_days_per_year - risk_free_rate) / downside_deviation
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0
    metrics['sortino_ratio'] = sortino_ratio

    # Calmar Ratio (CAGR / Max Drawdown)
    # Note: Max drawdown is calculated separately
    metrics['calmar_ratio'] = 0.0  # Will be updated later

    # Information Ratio (requires benchmark)
    metrics['information_ratio'] = 0.0

    # Beta and Alpha (require benchmark)
    metrics['beta'] = 0.0
    metrics['alpha'] = 0.0

    # Treynor Ratio (requires beta)
    metrics['treynor_ratio'] = 0.0

    return metrics


def _calculate_trade_metrics(
    trades: List[Trade],
    equity_curve: pd.Series
) -> Dict[str, Any]:
    """
    计算交易相关指标。

    目的：分析交易执行的质量和特征，评估交易策略的执行效果。

    实现方案：
    1. 交易盈亏分析：按品种分组计算每笔交易的盈亏
    2. 交易统计计算：胜率、盈亏比、平均盈亏等基础统计
    3. 交易模式分析：连续盈亏、最大单笔盈亏等模式识别
    4. 持仓时间分析：交易持有期统计

    参数说明：
        trades: 交易记录列表，包含所有Trade对象
        equity_curve: 净值曲线序列（此函数中未直接使用，为接口一致性保留）

    返回值：
        Dict[str, Any]: 包含交易指标的字典，包括胜率、盈亏比、平均交易收益率等

    计算流程：
        1. 按交易品种分组，跟踪每个品种的持仓状态
        2. 计算每笔平仓交易的盈亏（考虑佣金和滑点）
        3. 统计盈利交易和亏损交易的数量和金额
        4. 计算各项交易指标：胜率、盈亏比、平均盈亏等
        5. 分析交易模式：最大连续盈亏、最大单笔盈亏等

    主要指标：
        1. 胜率：盈利交易次数占总交易次数的比例
        2. 盈亏比：总盈利金额与总亏损金额的比率
        3. 平均盈利：盈利交易的平均盈利金额
        4. 平均亏损：亏损交易的平均亏损金额
        5. 平均交易收益率：所有交易的平均收益率
        6. 平均持仓时间：交易的平均持有期
        7. 最大连续盈利/亏损：连续盈利或亏损的最大次数

    注意事项：
        1. 处理空交易列表，返回零值指标
        2. 正确计算做多和做空交易的盈亏
        3. 考虑交易成本（佣金和滑点）的影响
        4. 处理部分平仓和加仓的复杂情况
    """
    metrics = {}

    if not trades:
        metrics.update({
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_trade_return': 0.0,
            'avg_trade_duration': pd.Timedelta(0),
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'largest_winning_trade': 0.0,
            'largest_losing_trade': 0.0
        })
        return metrics

    # Calculate trade returns
    trade_returns = []
    trade_durations = []
    winning_trades = 0
    losing_trades = 0
    total_win_amount = 0.0
    total_loss_amount = 0.0
    largest_win = 0.0
    largest_loss = 0.0

    # Group trades by symbol to calculate P&L per trade
    trades_by_symbol = {}
    for trade in trades:
        if trade.symbol not in trades_by_symbol:
            trades_by_symbol[trade.symbol] = []
        trades_by_symbol[trade.symbol].append(trade)

    # Calculate P&L for each symbol's trades
    for symbol, symbol_trades in trades_by_symbol.items():
        position = 0
        entry_price = 0.0
        entry_time = None
        entry_trade_ids = []

        for trade in sorted(symbol_trades, key=lambda x: x.timestamp):
            if position == 0:
                # Opening a position
                position = trade.quantity
                entry_price = trade.price
                entry_time = trade.timestamp
                entry_trade_ids = [trade.trade_id]
            else:
                # Adding to or reducing position
                if position * trade.quantity >= 0:
                    # Adding to position
                    total_quantity = position + trade.quantity
                    total_cost = (position * entry_price + trade.quantity * trade.price)
                    entry_price = total_cost / total_quantity if total_quantity != 0 else 0
                    position = total_quantity
                    entry_trade_ids.append(trade.trade_id)
                else:
                    # Reducing or closing position
                    close_quantity = min(abs(trade.quantity), abs(position))
                    if position > 0:
                        # Closing long position
                        trade_return = (trade.price - entry_price) * close_quantity
                    else:
                        # Closing short position
                        trade_return = (entry_price - trade.price) * close_quantity

                    # Subtract commissions and slippage
                    trade_return -= trade.commission + trade.slippage

                    trade_returns.append(trade_return)
                    trade_duration = trade.timestamp - entry_time
                    trade_durations.append(trade_duration)

                    # Update statistics
                    if trade_return > 0:
                        winning_trades += 1
                        total_win_amount += trade_return
                        largest_win = max(largest_win, trade_return)
                    else:
                        losing_trades += 1
                        total_loss_amount += abs(trade_return)
                        largest_loss = max(largest_loss, abs(trade_return))

                    # Update position
                    position += trade.quantity
                    if abs(position) < 1e-6:
                        # Position fully closed
                        position = 0
                        entry_price = 0.0
                        entry_time = None
                        entry_trade_ids = []
                    else:
                        # Position partially closed
                        entry_trade_ids.append(trade.trade_id)

    # Calculate metrics
    total_trades = len(trade_returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    avg_win = total_win_amount / winning_trades if winning_trades > 0 else 0.0
    avg_loss = total_loss_amount / losing_trades if losing_trades > 0 else 0.0

    profit_factor = total_win_amount / total_loss_amount if total_loss_amount > 0 else 0.0

    avg_trade_return = sum(trade_returns) / total_trades if total_trades > 0 else 0.0

    if trade_durations:
        avg_trade_duration = pd.Timedelta(sum(trade_durations, pd.Timedelta(0)) / len(trade_durations))
    else:
        avg_trade_duration = pd.Timedelta(0)

    # Calculate consecutive wins/losses
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0

    for ret in trade_returns:
        if ret > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

    metrics.update({
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade_return': avg_trade_return,
        'avg_trade_duration': avg_trade_duration,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'largest_winning_trade': largest_win,
        'largest_losing_trade': largest_loss
    })

    return metrics


def _calculate_drawdown_metrics(equity_curve: pd.Series) -> Dict[str, Any]:
    """
    计算回撤相关指标。

    目的：评估策略的下行风险和资金回撤情况，衡量策略的稳健性。

    实现方案：
    1. 计算运行最大值：净值曲线的历史最高点序列
    2. 计算回撤序列：当前净值相对于历史最高点的回撤比例
    3. 统计回撤指标：最大回撤、平均回撤、回撤持续时间等
    4. 识别峰值和谷值：记录净值曲线的最高点和最低点

    参数说明：
        equity_curve: 净值曲线序列，组合净值随时间变化

    返回值：
        Dict[str, Any]: 包含回撤指标的字典，包括最大回撤、平均回撤、回撤持续时间等

    计算公式：
        1. 运行最大值 = 净值曲线的扩展最大值（历史最高点）
        2. 回撤 = (运行最大值 - 当前净值) / 运行最大值
        3. 最大回撤 = 回撤序列的最大值
        4. 平均回撤 = 正回撤值的平均值
        5. 峰值资金 = 净值曲线的最大值
        6. 谷值资金 = 净值曲线的最小值

    主要指标：
        1. 最大回撤：净值从峰值到谷值的最大跌幅
        2. 平均回撤：所有正回撤值的平均值
        3. 最大回撤持续时间：回撤持续的最大时间长度
        4. 回撤序列：每个时间点的回撤值序列
        5. 峰值资金：净值曲线的历史最高点
        6. 谷值资金：净值曲线的历史最低点

    注意事项：
        1. 处理空净值曲线，返回零值指标
        2. 使用扩展最大值计算运行最高点
        3. 处理除零错误（当运行最大值为零时）
        4. 正确计算回撤持续时间
    """
    metrics = {}

    if equity_curve.empty:
        metrics.update({
            'max_drawdown': 0.0,
            'avg_drawdown': 0.0,
            'max_drawdown_duration': pd.Timedelta(0),
            'drawdown': pd.Series(dtype=float),
            'peak_capital': 0.0,
            'trough_capital': 0.0
        })
        return metrics

    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = (running_max - equity_curve) / running_max
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Max drawdown
    max_drawdown = drawdown.max() if not drawdown.empty else 0.0

    # Average drawdown (excluding zeros)
    non_zero_drawdown = drawdown[drawdown > 0]
    avg_drawdown = non_zero_drawdown.mean() if len(non_zero_drawdown) > 0 else 0.0

    # Max drawdown duration
    max_drawdown_duration = _calculate_max_drawdown_duration(drawdown, equity_curve.index)

    # Peak and trough capital
    peak_capital = equity_curve.max() if not equity_curve.empty else 0.0
    trough_capital = equity_curve.min() if not equity_curve.empty else 0.0

    metrics.update({
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'drawdown': drawdown,
        'peak_capital': peak_capital,
        'trough_capital': trough_capital
    })

    return metrics


def _calculate_max_drawdown_duration(
    drawdown: pd.Series,
    index: pd.DatetimeIndex
) -> pd.Timedelta:
    """Calculate maximum drawdown duration."""
    if drawdown.empty:
        return pd.Timedelta(0)

    max_duration = pd.Timedelta(0)
    current_start = None

    for i, (date, dd) in enumerate(zip(index, drawdown)):
        if dd > 0:
            if current_start is None:
                current_start = date
        else:
            if current_start is not None:
                duration = date - current_start
                if duration > max_duration:
                    max_duration = duration
                current_start = None

    # Check if drawdown is still ongoing at the end
    if current_start is not None and len(index) > 0:
        duration = index[-1] - current_start
        if duration > max_duration:
            max_duration = duration

    return max_duration


def calculate_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate benchmark-relative metrics.

    Args:
        strategy_returns: Strategy returns series
        benchmark_returns: Benchmark returns series
        risk_free_rate: Annual risk-free rate
        trading_days_per_year: Number of trading days per year

    Returns:
        Dictionary with benchmark-relative metrics
    """
    if strategy_returns.empty or benchmark_returns.empty:
        raise PerformanceError("Cannot calculate benchmark metrics with empty data")

    # Align indices
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) == 0:
        raise PerformanceError("No common dates between strategy and benchmark returns")

    strategy_aligned = strategy_returns.loc[common_idx]
    benchmark_aligned = benchmark_returns.loc[common_idx]

    metrics = {}

    # Excess returns
    excess_returns = strategy_aligned - benchmark_aligned

    # Information Ratio
    if excess_returns.std() > 0:
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days_per_year)
    else:
        information_ratio = 0.0
    metrics['information_ratio'] = information_ratio

    # Beta (covariance / variance)
    covariance = np.cov(strategy_aligned, benchmark_aligned)[0, 1]
    benchmark_variance = benchmark_aligned.var()
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
    metrics['beta'] = beta

    # Alpha
    strategy_annual_return = strategy_aligned.mean() * trading_days_per_year
    benchmark_annual_return = benchmark_aligned.mean() * trading_days_per_year
    alpha = strategy_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
    metrics['alpha'] = alpha

    # Treynor Ratio
    if beta > 0:
        treynor_ratio = (strategy_annual_return - risk_free_rate) / beta
    else:
        treynor_ratio = 0.0
    metrics['treynor_ratio'] = treynor_ratio

    # R-squared
    correlation = strategy_aligned.corr(benchmark_aligned)
    r_squared = correlation ** 2 if not pd.isna(correlation) else 0.0
    metrics['r_squared'] = r_squared

    # Tracking Error
    tracking_error = excess_returns.std() * np.sqrt(trading_days_per_year)
    metrics['tracking_error'] = tracking_error

    return metrics


class BacktestMetrics:
    """
    Convenience class for calculating and storing backtest metrics.
    """

    def __init__(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: List[Trade],
        initial_capital: float,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252
    ):
        """
        Initialize metrics calculator.

        Args:
            returns: Strategy returns series
            equity_curve: Portfolio equity curve
            trades: List of trades
            initial_capital: Initial capital
            benchmark_returns: Benchmark returns (optional)
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Trading days per year
        """
        self.returns = returns
        self.equity_curve = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

        self.metrics = {}
        self.benchmark_metrics = {}

    def calculate(self) -> None:
        """Calculate all metrics."""
        # Calculate basic metrics
        self.metrics = calculate_all_metrics(
            returns=self.returns,
            equity_curve=self.equity_curve,
            trades=self.trades,
            initial_capital=self.initial_capital,
            risk_free_rate=self.risk_free_rate,
            trading_days_per_year=self.trading_days_per_year
        )

        # Calculate benchmark metrics if available
        if self.benchmark_returns is not None:
            try:
                self.benchmark_metrics = calculate_benchmark_metrics(
                    strategy_returns=self.returns,
                    benchmark_returns=self.benchmark_returns,
                    risk_free_rate=self.risk_free_rate,
                    trading_days_per_year=self.trading_days_per_year
                )
            except Exception as e:
                logger.warning(f"Failed to calculate benchmark metrics: {e}")
                self.benchmark_metrics = {}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            'performance': {
                'total_return': self.metrics.get('total_return', 0.0),
                'annual_return': self.metrics.get('annual_return', 0.0),
                'cagr': self.metrics.get('cagr', 0.0),
                'sharpe_ratio': self.metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': self.metrics.get('sortino_ratio', 0.0),
                'max_drawdown': self.metrics.get('max_drawdown', 0.0),
                'volatility': self.metrics.get('volatility', 0.0)
            },
            'trades': {
                'total_trades': self.metrics.get('total_trades', 0),
                'win_rate': self.metrics.get('win_rate', 0.0),
                'profit_factor': self.metrics.get('profit_factor', 0.0),
                'avg_trade_return': self.metrics.get('avg_trade_return', 0.0)
            }
        }

        if self.benchmark_metrics:
            summary['benchmark'] = {
                'alpha': self.benchmark_metrics.get('alpha', 0.0),
                'beta': self.benchmark_metrics.get('beta', 0.0),
                'information_ratio': self.benchmark_metrics.get('information_ratio', 0.0)
            }

        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame."""
        data = []

        # Add performance metrics
        perf_metrics = [
            ('Total Return', f"{self.metrics.get('total_return', 0):.2%}"),
            ('Annual Return', f"{self.metrics.get('annual_return', 0):.2%}"),
            ('CAGR', f"{self.metrics.get('cagr', 0):.2%}"),
            ('Sharpe Ratio', f"{self.metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{self.metrics.get('sortino_ratio', 0):.2f}"),
            ('Max Drawdown', f"{self.metrics.get('max_drawdown', 0):.2%}"),
            ('Volatility', f"{self.metrics.get('volatility', 0):.2%}")
        ]

        for name, value in perf_metrics:
            data.append({'Metric': name, 'Value': value})

        # Add trade metrics
        trade_metrics = [
            ('Total Trades', str(self.metrics.get('total_trades', 0))),
            ('Win Rate', f"{self.metrics.get('win_rate', 0):.2%}"),
            ('Profit Factor', f"{self.metrics.get('profit_factor', 0):.2f}"),
            ('Avg Win', f"{self.metrics.get('avg_win', 0):.2f}"),
            ('Avg Loss', f"{self.metrics.get('avg_loss', 0):.2f}")
        ]

        for name, value in trade_metrics:
            data.append({'Metric': name, 'Value': value})

        # Add benchmark metrics if available
        if self.benchmark_metrics:
            bench_metrics = [
                ('Alpha', f"{self.benchmark_metrics.get('alpha', 0):.2%}"),
                ('Beta', f"{self.benchmark_metrics.get('beta', 0):.2f}"),
                ('Information Ratio', f"{self.benchmark_metrics.get('information_ratio', 0):.2f}")
            ]

            for name, value in bench_metrics:
                data.append({'Metric': name, 'Value': value})

        return pd.DataFrame(data)
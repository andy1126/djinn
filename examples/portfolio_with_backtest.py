"""
投资组合与回测模块集成示例。

本示例演示了如何将投资组合管理模块与回测引擎集成，实现完整的量化投资策略回测流程。
展示了投资组合管理在回测中的关键作用，包括持仓管理、风险控制和自动再平衡。

集成方案概述：
1. 创建自定义回测引擎，使用 EquityPortfolio 管理持仓
2. 在回测循环中，每个时间点更新投资组合价格
3. 根据策略信号执行交易，通过投资组合管理交易执行
4. 定期检查并执行投资组合再平衡
5. 使用投资组合的快照功能记录绩效
6. 最终生成包含投资组合管理细节的回测报告

集成优势：
1. 统一的持仓管理：所有交易通过投资组合接口执行，确保一致性
2. 自动风险管理：利用投资组合内置的风险管理器控制头寸风险
3. 灵活的再平衡：支持多种再平衡策略，可根据配置自动执行
4. 详细的绩效跟踪：投资组合快照提供详细的持仓和绩效数据
5. 模块化设计：投资组合模块可独立测试和复用

演示场景：
我们创建一个简单的移动平均线交叉策略，在回测中使用投资组合管理持仓。
策略在快速均线上穿慢速均线时买入，下穿时卖出。
投资组合负责管理交易执行、持仓跟踪、风险控制和定期再平衡。

运行步骤：
1. 加载历史价格数据
2. 创建投资组合实例
3. 创建回测引擎并设置投资组合
4. 运行回测，模拟策略执行
5. 分析回测结果，包括投资组合表现
6. 生成可视化报告

预期结果：
- 成功将投资组合集成到回测流程中
- 投资组合正确管理所有交易和持仓
- 风险控制和再平衡功能正常工作
- 生成包含投资组合详细信息的回测报告

注意：本示例使用简化实现，实际项目中可能需要更复杂的集成。
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.djinn.core.portfolio import (
    EquityPortfolio,
    create_portfolio,
    RebalancingFrequency,
    PortfolioAllocation
)


class PortfolioBacktestEngine:
    """
    使用投资组合管理的简化回测引擎。

    这个类演示了如何将投资组合模块集成到回测引擎中。
    在实际项目中，你可能需要扩展现有的回测引擎类。
    """

    def __init__(self, portfolio, initial_capital=100000.0):
        """
        初始化回测引擎。

        Args:
            portfolio: 投资组合实例
            initial_capital: 初始资金
        """
        self.portfolio = portfolio
        self.initial_capital = initial_capital
        self.results = {
            'dates': [],
            'portfolio_values': [],
            'cash_balances': [],
            'signals': [],
            'trades': [],
            'snapshots': []
        }

    def run_backtest(self, data, signals, commission=0.001):
        """
        运行回测。

        Args:
            data: 价格数据DataFrame，索引为日期，列为资产代码
            signals: 信号DataFrame，与data相同形状，1表示买入，-1表示卖出，0表示持有
            commission: 佣金率
        """
        print(f"开始回测，时间范围: {data.index[0]} 到 {data.index[-1]}")
        print(f"资产数量: {len(data.columns)}")

        # 初始投资组合配置（等权重）
        symbols = list(data.columns)
        equal_weight = 1.0 / len(symbols)

        for symbol in symbols:
            self.portfolio.add_allocation(
                symbol=symbol,
                target_weight=equal_weight,
                min_weight=0.05,
                max_weight=0.40
            )

        # 回测主循环
        for i, date in enumerate(data.index):
            # 获取当前价格
            current_prices = data.loc[date].to_dict()

            # 更新投资组合价格
            self.portfolio.update_prices(current_prices)

            # 获取当前信号
            current_signals = signals.loc[date].to_dict() if i > 0 else {s: 0 for s in symbols}

            # 执行交易（基于信号）
            self._execute_trades(current_signals, current_prices, commission)

            # 检查并执行再平衡（每月一次）
            if self._should_rebalance(date):
                self._rebalance_portfolio(current_prices, commission)

            # 记录快照
            snapshot = self.portfolio.take_snapshot()

            # 记录结果
            self.results['dates'].append(date)
            self.results['portfolio_values'].append(self.portfolio.get_total_value())
            self.results['cash_balances'].append(self.portfolio.current_capital)
            self.results['signals'].append(current_signals.copy())
            self.results['snapshots'].append(snapshot)

            # 进度显示
            if (i + 1) % 50 == 0 or i == len(data) - 1:
                progress = (i + 1) / len(data) * 100
                print(f"进度: {progress:.1f}% ({i + 1}/{len(data)})")

        print(f"回测完成，最终组合价值: ${self.portfolio.get_total_value():,.2f}")

    def _execute_trades(self, signals, prices, commission):
        """
        根据信号执行交易。

        Args:
            signals: 信号字典
            prices: 价格字典
            commission: 佣金率
        """
        for symbol, signal in signals.items():
            if signal == 0:
                continue

            # 计算交易数量（简化：每次交易组合价值的5%）
            portfolio_value = self.portfolio.get_total_value()
            trade_amount = portfolio_value * 0.05
            price = prices.get(symbol)

            if price and price > 0:
                quantity = signal * (trade_amount / price)

                try:
                    self.portfolio.execute_trade(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        commission=commission
                    )

                    # 记录交易
                    self.results['trades'].append({
                        'date': self.results['dates'][-1] if self.results['dates'] else datetime.now(),
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'signal': signal
                    })

                except Exception as e:
                    # 交易失败（如资金不足）
                    pass

    def _should_rebalance(self, date):
        """
        检查是否需要再平衡。

        Args:
            date: 当前日期

        Returns:
            bool: 是否需要再平衡
        """
        # 简化：每月第一天再平衡
        return date.day == 1

    def _rebalance_portfolio(self, prices, commission):
        """
        执行再平衡。

        Args:
            prices: 当前价格
            commission: 佣金率
        """
        try:
            trades = self.portfolio.rebalance(self.portfolio.target_allocations)

            # 记录再平衡交易
            for trade in trades:
                self.results['trades'].append({
                    'date': self.results['dates'][-1] if self.results['dates'] else datetime.now(),
                    'symbol': trade['symbol'],
                    'quantity': trade['quantity'],
                    'price': prices.get(trade['symbol'], 0),
                    'side': trade['side'],
                    'type': 'rebalance'
                })

        except Exception as e:
            print(f"再平衡失败: {e}")

    def get_results(self):
        """获取回测结果。"""
        return self.results

    def calculate_metrics(self):
        """计算回测绩效指标。"""
        portfolio_values = np.array(self.results['portfolio_values'])

        if len(portfolio_values) < 2:
            return {}

        # 计算收益率
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # 基本指标
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)

        # 最大回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        # 夏普比率（无风险利率2%）
        risk_free_rate = 0.02
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # 交易统计
        num_trades = len(self.results['trades'])
        winning_trades = sum(1 for t in self.results['trades']
                            if t.get('type') != 'rebalance')  # 简化

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'final_value': portfolio_values[-1]
        }


def generate_sample_data(symbols=None, start_date='2023-01-01', end_date='2023-12-31'):
    """
    生成样本数据。

    Args:
        symbols: 资产代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        tuple: (价格数据, 信号数据)
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    # 生成日期范围
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)

    # 生成价格数据（几何布朗运动）
    price_data = pd.DataFrame(index=dates, columns=symbols)

    # 初始价格
    initial_prices = {'AAPL': 150, 'MSFT': 250, 'GOOGL': 100, 'AMZN': 120, 'TSLA': 200}

    for symbol in symbols:
        # 生成随机收益率
        mu = 0.0003  # 日期望收益率
        sigma = 0.02  # 日波动率
        returns = np.random.normal(mu, sigma, len(dates))

        # 生成价格序列
        prices = initial_prices.get(symbol, 100) * np.exp(np.cumsum(returns))
        price_data[symbol] = prices

    # 生成简单的移动平均线交叉信号
    signal_data = pd.DataFrame(index=dates, columns=symbols)

    for symbol in symbols:
        prices = price_data[symbol]

        # 计算快速和慢速移动平均线
        fast_ma = prices.rolling(window=10).mean()
        slow_ma = prices.rolling(window=30).mean()

        # 生成信号：快线上穿慢线买入，下穿卖出
        signals = pd.Series(0, index=dates)
        signals[fast_ma > slow_ma] = 1  # 买入信号
        signals[fast_ma < slow_ma] = -1  # 卖出信号

        signal_data[symbol] = signals

    return price_data, signal_data


def plot_backtest_results(results, metrics):
    """
    绘制回测结果图表。

    Args:
        results: 回测结果
        metrics: 绩效指标
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('投资组合回测结果', fontsize=16)

    # 1. 投资组合价值曲线
    ax1 = axes[0, 0]
    ax1.plot(results['dates'], results['portfolio_values'], linewidth=2, color='blue')
    ax1.set_title('投资组合价值变化')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('组合价值 ($)')
    ax1.grid(True, alpha=0.3)

    # 添加初始和最终价值标注
    ax1.axhline(y=results['portfolio_values'][0], color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=results['portfolio_values'][-1], color='red', linestyle='--', alpha=0.5)

    # 2. 现金余额变化
    ax2 = axes[0, 1]
    ax2.plot(results['dates'], results['cash_balances'], linewidth=2, color='green')
    ax2.set_title('现金余额变化')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('现金余额 ($)')
    ax2.grid(True, alpha=0.3)

    # 3. 资产权重变化（热图）
    ax3 = axes[1, 0]
    # 提取权重数据
    snapshots = results['snapshots']
    if snapshots and hasattr(snapshots[0], 'allocations'):
        # 获取所有资产代码
        all_symbols = set()
        for snapshot in snapshots:
            all_symbols.update(snapshot.allocations.keys())

        # 创建权重矩阵
        symbols = sorted(all_symbols)
        weight_matrix = np.zeros((len(snapshots), len(symbols)))

        for i, snapshot in enumerate(snapshots):
            for j, symbol in enumerate(symbols):
                weight_matrix[i, j] = snapshot.allocations.get(symbol, 0)

        # 绘制热图
        im = ax3.imshow(weight_matrix.T, aspect='auto', cmap='YlOrRd',
                       extent=[0, len(snapshots), 0, len(symbols)])
        ax3.set_title('资产权重变化热图')
        ax3.set_xlabel('时间点')
        ax3.set_ylabel('资产')
        ax3.set_yticks(range(len(symbols)))
        ax3.set_yticklabels(symbols)
        plt.colorbar(im, ax=ax3, label='权重')

    # 4. 绩效指标条形图
    ax4 = axes[1, 1]
    metric_names = ['总收益率', '年化收益率', '波动率', '最大回撤', '夏普比率']
    metric_values = [
        metrics['total_return'],
        metrics['annual_return'],
        metrics['volatility'],
        metrics['max_drawdown'],
        metrics['sharpe_ratio']
    ]

    colors = ['green', 'green', 'red', 'red', 'blue']
    bars = ax4.bar(metric_names, metric_values, color=colors)
    ax4.set_title('绩效指标')
    ax4.set_ylabel('数值')
    ax4.grid(True, alpha=0.3, axis='y')

    # 在条形上添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2%}' if value != metrics['sharpe_ratio'] else f'{value:.2f}',
                ha='center', va='bottom')

    # 5. 交易次数统计
    ax5 = axes[2, 0]
    trades = results['trades']
    if trades:
        # 按资产统计交易次数
        trade_counts = {}
        for trade in trades:
            symbol = trade['symbol']
            trade_counts[symbol] = trade_counts.get(symbol, 0) + 1

        symbols = list(trade_counts.keys())
        counts = list(trade_counts.values())

        ax5.bar(symbols, counts, color='steelblue')
        ax5.set_title('各资产交易次数')
        ax5.set_xlabel('资产')
        ax5.set_ylabel('交易次数')
        ax5.grid(True, alpha=0.3, axis='y')

    # 6. 回撤曲线
    ax6 = axes[2, 1]
    portfolio_values = np.array(results['portfolio_values'])
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak

    ax6.fill_between(results['dates'], 0, drawdown, color='red', alpha=0.3)
    ax6.plot(results['dates'], drawdown, color='darkred', linewidth=1)
    ax6.set_title('回撤曲线')
    ax6.set_xlabel('日期')
    ax6.set_ylabel('回撤')
    ax6.grid(True, alpha=0.3)
    ax6.invert_yaxis()  # 回撤向下显示

    plt.tight_layout()
    plt.savefig('portfolio_backtest_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_portfolio_backtest_example():
    """
    运行投资组合回测示例。
    """
    print("=" * 80)
    print("投资组合与回测模块集成示例")
    print("=" * 80)
    print()

    # 步骤1: 创建投资组合
    print("步骤1: 创建投资组合")
    print("-" * 40)

    portfolio = create_portfolio(
        initial_capital=100000.0,
        name="回测投资组合",
        currency="USD",
        rebalancing_frequency="monthly",
        rebalancing_strategy="equal_weight"
    )

    print(f"✓ 创建投资组合: {portfolio.name}")
    print(f"  初始资金: ${portfolio.initial_capital:,.2f}")
    print(f"  再平衡策略: {portfolio.rebalancing_strategy_type}")
    print()

    # 步骤2: 生成样本数据
    print("步骤2: 生成样本数据")
    print("-" * 40)

    price_data, signal_data = generate_sample_data()
    symbols = list(price_data.columns)

    print(f"生成 {len(price_data)} 个交易日的数据")
    print(f"资产: {', '.join(symbols)}")
    print(f"时间范围: {price_data.index[0].date()} 到 {price_data.index[-1].date()}")
    print()

    # 步骤3: 创建并运行回测引擎
    print("步骤3: 运行回测")
    print("-" * 40)

    backtest_engine = PortfolioBacktestEngine(portfolio)
    backtest_engine.run_backtest(price_data, signal_data, commission=0.001)

    print()

    # 步骤4: 分析结果
    print("步骤4: 分析回测结果")
    print("-" * 40)

    results = backtest_engine.get_results()
    metrics = backtest_engine.calculate_metrics()

    print("绩效指标:")
    print(f"  总收益率: {metrics['total_return']:.2%}")
    print(f"  年化收益率: {metrics['annual_return']:.2%}")
    print(f"  波动率: {metrics['volatility']:.2%}")
    print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  交易次数: {metrics['num_trades']}")
    print(f"  最终价值: ${metrics['final_value']:,.2f}")
    print()

    # 步骤5: 投资组合状态
    print("步骤5: 最终投资组合状态")
    print("-" * 40)

    print(f"现金余额: ${portfolio.current_capital:,.2f}")
    print(f"持仓数量: {len(portfolio.holdings)}")
    print(f"总价值: ${portfolio.get_total_value():,.2f}")

    if portfolio.holdings:
        print("\n当前持仓:")
        print(f"{'代码':<8} {'数量':<10} {'均价':<10} {'现价':<10} {'市值':<12} {'盈亏':<12}")
        print("-" * 70)

        for symbol, holding in portfolio.holdings.items():
            print(
                f"{symbol:<8} "
                f"{holding.quantity:<10.0f} "
                f"${holding.avg_price:<9.2f} "
                f"${holding.current_price:<9.2f} "
                f"${holding.market_value:<11,.2f} "
                f"${holding.unrealized_pnl:<11,.2f}"
            )
    print()

    # 步骤6: 可视化
    print("步骤6: 生成可视化图表")
    print("-" * 40)

    try:
        plot_backtest_results(results, metrics)
        print("✓ 图表已保存为 'portfolio_backtest_results.png'")
    except Exception as e:
        print(f"✗ 图表生成失败: {e}")
        print("  确保已安装 matplotlib: pip install matplotlib")

    print()
    print("示例运行完成！")

    return portfolio, backtest_engine, results, metrics


if __name__ == "__main__":
    try:
        portfolio, backtest_engine, results, metrics = run_portfolio_backtest_example()
    except Exception as e:
        print(f"\n示例运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
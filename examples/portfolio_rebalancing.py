"""
投资组合再平衡策略对比示例。

本示例演示了 Djinn 投资组合模块中不同再平衡策略的使用方法和效果对比，包括：
1. 等权重再平衡策略 (Equal Weight)
2. 风险平价再平衡策略 (Risk Parity)
3. 均值-方差优化再平衡策略 (Mean-Variance Optimization)
4. Black-Litterman 模型再平衡策略

演示场景：
我们创建一个包含5只美股（不同行业）的投资组合，初始资金为100,000美元。
分别使用四种不同的再平衡策略管理该组合，模拟一年的价格变动（12个月），
比较不同策略在收益率、波动率、最大回撤和夏普比率等方面的表现。

使用的关键类和函数：
- EquityPortfolio: 股票投资组合具体实现类
- create_rebalancer: 再平衡策略工厂函数
- EqualWeightRebalancer, RiskParityRebalancer, MeanVarianceRebalancer, BlackLittermanRebalancer
- PortfolioAllocation: 目标配置数据结构
- RiskManager: 风险管理器

运行步骤：
1. 准备测试数据（模拟资产价格和波动率）
2. 创建四个使用不同再平衡策略的投资组合
3. 模拟12个月的价格变动和每月再平衡
4. 记录每个月的组合表现
5. 比较四种策略的最终表现
6. 可视化结果

预期结果：
- 等权重策略：简单稳定，但可能不是最优风险调整回报
- 风险平价策略：更好的风险分散，波动率较低
- 均值-方差优化：理论上最优的风险回报平衡，但对输入参数敏感
- Black-Litterman模型：结合市场均衡和投资者观点，更稳定

注意：本示例使用模拟数据，实际表现可能因市场条件而异。
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
    create_rebalancer,
    RebalancingFrequency,
    PortfolioAllocation
)


def generate_simulated_data(num_assets=5, num_months=12):
    """
    生成模拟的资产数据。

    Args:
        num_assets: 资产数量
        num_months: 月份数量

    Returns:
        dict: 包含价格数据、波动率和相关性的字典
    """
    np.random.seed(42)  # 固定随机种子以确保可重复性

    # 资产名称和初始特征
    assets = [
        {"symbol": "AAPL", "sector": "科技", "base_volatility": 0.25},
        {"symbol": "JPM", "sector": "金融", "base_volatility": 0.20},
        {"symbol": "XOM", "sector": "能源", "base_volatility": 0.30},
        {"symbol": "JNJ", "sector": "医疗", "base_volatility": 0.18},
        {"symbol": "WMT", "sector": "消费", "base_volatility": 0.15}
    ]

    # 生成相关性矩阵（基于行业）
    sectors = [a["sector"] for a in assets]
    correlation_matrix = np.eye(num_assets)

    # 设置行业内部相关性较高
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            if sectors[i] == sectors[j]:
                corr = 0.7  # 同行业高相关
            else:
                corr = 0.3  # 跨行业低相关
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr

    # 生成协方差矩阵
    volatilities = np.array([a["base_volatility"] for a in assets]) / np.sqrt(12)  # 月波动率
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

    # 生成月度收益率
    monthly_returns = np.random.multivariate_normal(
        mean=np.array([0.01, 0.008, 0.012, 0.007, 0.006]),  # 月期望收益
        cov=covariance_matrix,
        size=num_months
    )

    # 生成价格序列（从100开始）
    initial_prices = np.array([175.0, 150.0, 110.0, 160.0, 170.0])
    price_series = [initial_prices]

    for t in range(num_months):
        new_prices = price_series[-1] * (1 + monthly_returns[t])
        price_series.append(new_prices)

    price_series = np.array(price_series)

    # 生成日期索引
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=30 * i) for i in range(num_months + 1)]

    return {
        "assets": assets,
        "price_series": price_series,
        "monthly_returns": monthly_returns,
        "volatilities": volatilities * np.sqrt(12),  # 年化波动率
        "covariance_matrix": covariance_matrix * 12,  # 年化协方差矩阵
        "dates": dates,
        "sectors": sectors
    }


def create_portfolio_with_strategy(strategy_name, initial_capital, assets):
    """
    创建使用特定再平衡策略的投资组合。

    Args:
        strategy_name: 策略名称
        initial_capital: 初始资金
        assets: 资产列表

    Returns:
        EquityPortfolio: 配置好的投资组合实例
    """
    # 创建再平衡器
    rebalancer = create_rebalancer(strategy_name)

    # 创建投资组合
    portfolio = EquityPortfolio(
        initial_capital=initial_capital,
        name=f"{strategy_name.replace('_', ' ').title()} 组合",
        currency="USD",
        rebalancing_frequency=RebalancingFrequency.MONTHLY,
        rebalancing_strategy=strategy_name
    )

    # 添加资产配置（初始等权重）
    num_assets = len(assets)
    equal_weight = 1.0 / num_assets

    for asset in assets:
        portfolio.add_allocation(
            symbol=asset["symbol"],
            target_weight=equal_weight,
            min_weight=0.05,
            max_weight=0.40
        )

    return portfolio


def run_rebalancing_backtest(portfolio, price_series, dates):
    """
    运行再平衡回测。

    Args:
        portfolio: 投资组合实例
        price_series: 价格序列
        dates: 日期序列

    Returns:
        dict: 回测结果
    """
    results = {
        "dates": [],
        "total_values": [],
        "cash_balances": [],
        "holdings_values": [],
        "weights": [],
        "trades": []
    }

    num_months = len(price_series) - 1

    # 初始快照
    portfolio.take_snapshot()
    results["dates"].append(dates[0])
    results["total_values"].append(portfolio.get_total_value())
    results["cash_balances"].append(portfolio.current_capital)
    results["holdings_values"].append(portfolio.get_holdings_value())
    results["weights"].append(portfolio.calculate_weights())

    # 每月迭代
    for month in range(num_months):
        current_date = dates[month + 1]

        # 更新价格
        prices = {asset["symbol"]: price_series[month + 1, i]
                 for i, asset in enumerate(data["assets"])}
        portfolio.update_prices(prices)

        # 检查并执行再平衡
        if portfolio.needs_rebalancing(threshold=0.05):
            try:
                trades = portfolio.rebalance(portfolio.target_allocations)
                results["trades"].append({
                    "date": current_date,
                    "trades": trades
                })
            except Exception as e:
                print(f"再平衡失败: {e}")

        # 记录快照
        portfolio.take_snapshot()

        # 记录结果
        results["dates"].append(current_date)
        results["total_values"].append(portfolio.get_total_value())
        results["cash_balances"].append(portfolio.current_capital)
        results["holdings_values"].append(portfolio.get_holdings_value())
        results["weights"].append(portfolio.calculate_weights())

    return results


def calculate_performance_metrics(results):
    """
    计算绩效指标。

    Args:
        results: 回测结果

    Returns:
        dict: 绩效指标
    """
    total_values = np.array(results["total_values"])
    returns = np.diff(total_values) / total_values[:-1]

    if len(returns) == 0:
        return {}

    # 计算基本指标
    total_return = (total_values[-1] - total_values[0]) / total_values[0]
    annual_return = (1 + total_return) ** (12 / len(returns)) - 1  # 年化
    volatility = np.std(returns) * np.sqrt(12)  # 年化波动率

    # 计算最大回撤
    peak = np.maximum.accumulate(total_values)
    drawdown = (peak - total_values) / peak
    max_drawdown = np.max(drawdown)

    # 计算夏普比率（假设无风险利率为2%）
    risk_free_rate = 0.02
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "num_trades": sum(len(r["trades"]) for r in results["trades"]) if results["trades"] else 0
    }


def plot_comparison(strategy_results, data):
    """
    绘制策略比较图表。

    Args:
        strategy_results: 各策略的结果字典
        data: 模拟数据
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('再平衡策略表现比较', fontsize=16)

    # 1. 组合价值曲线
    ax1 = axes[0, 0]
    for strategy_name, results in strategy_results.items():
        ax1.plot(results["dates"], results["total_values"],
                label=strategy_name.replace('_', ' ').title(), linewidth=2)

    ax1.set_title('投资组合价值变化')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('组合价值 ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 月度收益率分布
    ax2 = axes[0, 1]
    returns_data = []
    labels = []

    for strategy_name, results in strategy_results.items():
        total_values = np.array(results["total_values"])
        returns = np.diff(total_values) / total_values[:-1]
        returns_data.append(returns)
        labels.append(strategy_name.replace('_', ' ').title())

    ax2.boxplot(returns_data, labels=labels)
    ax2.set_title('月度收益率分布')
    ax2.set_ylabel('月度收益率')
    ax2.grid(True, alpha=0.3)

    # 3. 绩效指标雷达图
    ax3 = axes[1, 0]
    metrics = ["annual_return", "sharpe_ratio", "volatility", "max_drawdown"]
    metric_labels = ["年化收益", "夏普比率", "波动率", "最大回撤"]

    # 标准化指标用于雷达图
    normalized_data = []
    for metric in metrics:
        values = [results["metrics"].get(metric, 0) for results in strategy_results.values()]

        if metric in ["volatility", "max_drawdown"]:
            # 对于风险指标，越小越好，取倒数
            max_val = max(values)
            if max_val > 0:
                normalized = [1 - (v / max_val) for v in values]
            else:
                normalized = [1.0] * len(values)
        else:
            # 对于收益指标，越大越好
            max_val = max(values)
            if max_val > 0:
                normalized = [v / max_val for v in values]
            else:
                normalized = [0.0] * len(values)

        normalized_data.append(normalized)

    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax3 = plt.subplot(2, 2, 3, projection='polar')
    for i, (strategy_name, results) in enumerate(strategy_results.items()):
        values = [data[i] for data in normalized_data]
        values += values[:1]  # 闭合图形
        ax3.plot(angles, values, 'o-', linewidth=2, label=strategy_name.replace('_', ' ').title())
        ax3.fill(angles, values, alpha=0.1)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metric_labels)
    ax3.set_title('绩效指标雷达图（标准化）')
    ax3.grid(True)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 4. 资产权重变化（以等权重策略为例）
    ax4 = axes[1, 1]
    if "equal_weight" in strategy_results:
        weights_history = strategy_results["equal_weight"]["weights"]
        dates = strategy_results["equal_weight"]["dates"]

        # 提取每种资产的权重时间序列
        symbols = [asset["symbol"] for asset in data["assets"]]
        weight_series = {symbol: [] for symbol in symbols}

        for weight_dict in weights_history:
            for symbol in symbols:
                weight_series[symbol].append(weight_dict.get(symbol, 0))

        # 绘制堆叠面积图
        dates_str = [d.strftime('%Y-%m') for d in dates]
        bottom = np.zeros(len(dates_str))

        for symbol in symbols:
            weights = weight_series[symbol]
            ax4.fill_between(dates_str, bottom, bottom + weights, alpha=0.7, label=symbol)
            bottom += weights

        ax4.set_title('等权重策略资产配置变化')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('权重')
        ax4.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('rebalancing_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_strategy_comparison(strategy_results):
    """
    打印策略比较表格。

    Args:
        strategy_results: 各策略的结果字典
    """
    print("=" * 100)
    print("再平衡策略表现比较")
    print("=" * 100)
    print()

    # 准备表格数据
    headers = ["策略", "总收益率", "年化收益率", "波动率", "最大回撤", "夏普比率", "交易次数"]
    rows = []

    for strategy_name, results in strategy_results.items():
        metrics = results["metrics"]
        row = [
            strategy_name.replace('_', ' ').title(),
            f"{metrics.get('total_return', 0):.2%}",
            f"{metrics.get('annual_return', 0):.2%}",
            f"{metrics.get('volatility', 0):.2%}",
            f"{metrics.get('max_drawdown', 0):.2%}",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            f"{metrics.get('num_trades', 0)}"
        ]
        rows.append(row)

    # 打印表格
    col_widths = [20, 12, 12, 12, 12, 12, 12]

    # 打印表头
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # 打印数据行
    for row in rows:
        row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(row_line)

    print()

    # 打印总结
    print("策略特点总结:")
    print("1. 等权重策略: 简单易行，分散化好，但未考虑资产风险特征")
    print("2. 风险平价策略: 平衡各资产风险贡献，通常有更好的风险调整回报")
    print("3. 均值-方差优化: 理论上最优，但对输入参数（期望收益、协方差）非常敏感")
    print("4. Black-Litterman模型: 结合市场均衡和投资者观点，减少估计误差")
    print()
    print("注意: 以上结果基于模拟数据，实际市场表现可能有所不同。")


def run_rebalancing_comparison_example():
    """
    运行再平衡策略对比示例。
    """
    print("=" * 80)
    print("投资组合再平衡策略对比示例")
    print("=" * 80)
    print()

    # 步骤1: 生成模拟数据
    print("步骤1: 生成模拟数据")
    print("-" * 40)
    global data
    data = generate_simulated_data(num_assets=5, num_months=12)

    print(f"模拟 {len(data['assets'])} 只资产，{len(data['dates'])-1} 个月的数据")
    print("资产列表:")
    for i, asset in enumerate(data["assets"]):
        print(f"  {asset['symbol']} ({asset['sector']}): 初始价格 ${data['price_series'][0, i]:.2f}, "
              f"年化波动率 {data['volatilities'][i]:.1%}")
    print()

    # 步骤2: 创建使用不同策略的投资组合
    print("步骤2: 创建使用不同再平衡策略的投资组合")
    print("-" * 40)

    strategies = [
        "equal_weight",
        "risk_parity",
        "mean_variance",
        "black_litterman"
    ]

    portfolios = {}
    for strategy in strategies:
        portfolios[strategy] = create_portfolio_with_strategy(
            strategy, 100000.0, data["assets"]
        )
        print(f"✓ 创建 {strategy.replace('_', ' ').title()} 策略组合")
    print()

    # 步骤3: 运行回测
    print("步骤3: 运行回测（模拟12个月）")
    print("-" * 40)

    strategy_results = {}
    for strategy_name, portfolio in portfolios.items():
        print(f"正在运行 {strategy_name.replace('_', ' ').title()} 策略回测...")

        # 运行回测
        results = run_rebalancing_backtest(
            portfolio,
            data["price_series"],
            data["dates"]
        )

        # 计算绩效指标
        metrics = calculate_performance_metrics(results)

        strategy_results[strategy_name] = {
            "portfolio": portfolio,
            "results": results,
            "metrics": metrics
        }

        print(f"  → 总收益率: {metrics.get('total_return', 0):.2%}")
        print(f"  → 交易次数: {metrics.get('num_trades', 0)}")
    print()

    # 步骤4: 比较结果
    print("步骤4: 策略表现比较")
    print("-" * 40)

    print_strategy_comparison(strategy_results)

    # 步骤5: 可视化
    print("步骤5: 生成可视化图表")
    print("-" * 40)

    try:
        plot_comparison(
            {k: v["results"] for k, v in strategy_results.items()},
            data
        )
        print("✓ 图表已保存为 'rebalancing_strategy_comparison.png'")
    except Exception as e:
        print(f"✗ 图表生成失败: {e}")
        print("  确保已安装 matplotlib: pip install matplotlib")

    print()
    print("示例运行完成！")

    return strategy_results


if __name__ == "__main__":
    try:
        strategy_results = run_rebalancing_comparison_example()
    except Exception as e:
        print(f"\n示例运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
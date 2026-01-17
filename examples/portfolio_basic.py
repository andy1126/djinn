"""
投资组合基础使用示例。

本示例演示了 Djinn 投资组合模块的基本使用方法，包括：
1. 创建投资组合实例
2. 添加目标资产配置
3. 更新资产价格
4. 执行买入和卖出交易
5. 查看投资组合状态和绩效
6. 执行再平衡操作

演示场景：
假设我们有一个初始资金为100,000美元的投资组合，想要投资三只股票：
AAPL（苹果）、MSFT（微软）、GOOGL（谷歌），目标配置为等权重（各33.3%）。
我们将演示从零开始构建投资组合，执行交易，并管理投资组合的完整流程。

使用的关键类和函数：
- EquityPortfolio: 股票投资组合具体实现类
- create_portfolio: 工厂函数，用于创建投资组合
- PortfolioAllocation: 目标配置数据结构
- PortfolioHolding: 持仓数据结构
- PortfolioSnapshot: 投资组合快照

运行步骤：
1. 创建投资组合实例
2. 添加目标配置
3. 模拟价格更新
4. 执行买入交易
5. 查看投资组合状态
6. 执行卖出交易
7. 执行再平衡操作
8. 查看绩效报告

预期结果：
- 成功创建投资组合并添加目标配置
- 能够正常执行买入和卖出交易
- 投资组合状态正确更新（现金、持仓、市值等）
- 再平衡操作能够调整持仓至目标权重
- 绩效指标能够正确计算

注意：本示例使用模拟数据，实际应用中需要从数据源获取真实价格。
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.djinn.core.portfolio import (
    create_portfolio,
    EquityPortfolio,
    PortfolioAllocation,
    RebalancingFrequency
)


def run_basic_portfolio_example():
    """
    运行基础投资组合示例。
    """
    print("=" * 80)
    print("投资组合基础使用示例")
    print("=" * 80)
    print()

    # 步骤1: 创建投资组合实例
    print("步骤1: 创建投资组合实例")
    print("-" * 40)

    # 方法1: 使用工厂函数创建
    portfolio = create_portfolio(
        initial_capital=100000.0,
        name="我的第一个投资组合",
        currency="USD",
        rebalancing_frequency="monthly",
        rebalancing_strategy="equal_weight"
    )

    # 方法2: 直接创建（两种方式任选其一）
    # portfolio = EquityPortfolio(
    #     initial_capital=100000.0,
    #     name="我的第一个投资组合",
    #     currency="USD",
    #     rebalancing_frequency=RebalancingFrequency.MONTHLY,
    #     rebalancing_strategy="equal_weight"
    # )

    print(f"✓ 创建投资组合: {portfolio.name}")
    print(f"  初始资金: ${portfolio.initial_capital:,.2f}")
    print(f"  货币: {portfolio.currency}")
    print(f"  再平衡频率: {portfolio.rebalancing_frequency.value}")
    print()

    # 步骤2: 添加目标配置
    print("步骤2: 添加目标资产配置")
    print("-" * 40)

    # 定义三只股票的目标配置
    target_allocations = [
        {"symbol": "AAPL", "target_weight": 0.30, "min_weight": 0.20, "max_weight": 0.40},
        {"symbol": "MSFT", "target_weight": 0.30, "min_weight": 0.20, "max_weight": 0.40},
        {"symbol": "GOOGL", "target_weight": 0.40, "min_weight": 0.30, "max_weight": 0.50}
    ]

    for alloc in target_allocations:
        portfolio.add_allocation(**alloc)
        print(f"✓ 添加目标配置: {alloc['symbol']}，目标权重: {alloc['target_weight']:.1%}")

    print(f"总目标权重: {sum(a['target_weight'] for a in target_allocations):.1%}")
    print()

    # 步骤3: 模拟价格更新
    print("步骤3: 更新资产价格")
    print("-" * 40)

    # 模拟当前市场价格
    current_prices = {
        "AAPL": 175.50,
        "MSFT": 330.25,
        "GOOGL": 145.80
    }

    portfolio.update_prices(current_prices)

    for symbol, price in current_prices.items():
        print(f"✓ 更新价格: {symbol} = ${price:.2f}")
    print()

    # 步骤4: 执行买入交易
    print("步骤4: 执行买入交易")
    print("-" * 40)

    # 计算每只股票应买入的金额（按目标权重分配可用资金）
    total_value = portfolio.get_total_value()
    cash_available = portfolio.current_capital

    # 使用80%的现金进行初始投资
    investment_amount = cash_available * 0.8

    trades_executed = 0
    for alloc in target_allocations:
        symbol = alloc["symbol"]
        target_weight = alloc["target_weight"]
        price = current_prices[symbol]

        # 计算目标买入金额和数量
        target_amount = investment_amount * target_weight
        quantity = int(target_amount / price)  # 取整数股

        if quantity > 0:
            try:
                portfolio.execute_trade(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    commission=0.001  # 0.1%佣金
                )
                print(f"✓ 买入 {quantity} 股 {symbol} @ ${price:.2f}，金额: ${quantity * price:,.2f}")
                trades_executed += 1
            except Exception as e:
                print(f"✗ 买入 {symbol} 失败: {e}")

    print(f"共执行 {trades_executed} 笔买入交易")
    print(f"剩余现金: ${portfolio.current_capital:,.2f}")
    print()

    # 步骤5: 查看投资组合状态
    print("步骤5: 查看投资组合状态")
    print("-" * 40)

    # 获取当前持仓
    holdings = portfolio.holdings
    if holdings:
        print("当前持仓:")
        print(f"{'代码':<8} {'数量':<10} {'均价':<10} {'现价':<10} {'市值':<12} {'盈亏':<12} {'权重':<8}")
        print("-" * 80)

        total_market_value = portfolio.get_holdings_value()
        for symbol, holding in holdings.items():
            weight = holding.market_value / total_market_value if total_market_value > 0 else 0
            print(
                f"{symbol:<8} "
                f"{holding.quantity:<10.0f} "
                f"${holding.avg_price:<9.2f} "
                f"${holding.current_price:<9.2f} "
                f"${holding.market_value:<11,.2f} "
                f"${holding.unrealized_pnl:<11,.2f} "
                f"{weight:<7.1%}"
            )
    else:
        print("暂无持仓")

    print()
    print("投资组合概览:")
    print(f"  总价值: ${portfolio.get_total_value():,.2f}")
    print(f"  现金余额: ${portfolio.current_capital:,.2f}")
    print(f"  持仓总值: ${portfolio.get_holdings_value():,.2f}")
    print(f"  持仓数量: {len(holdings)}")
    print(f"  杠杆率: {portfolio.get_leverage():.2f}")
    print()

    # 步骤6: 查看目标配置偏差
    print("步骤6: 查看目标配置偏差")
    print("-" * 40)

    deviations = portfolio.calculate_deviation()
    if deviations:
        print(f"{'代码':<8} {'当前权重':<12} {'目标权重':<12} {'偏差':<12} {'状态':<10}")
        print("-" * 60)

        for symbol, deviation in deviations.items():
            current_weight = portfolio.calculate_weights().get(symbol, 0)
            target_allocation = portfolio.get_target_allocation(symbol)
            target_weight = target_allocation.target_weight if target_allocation else 0

            status = "超配" if deviation > 0.01 else "低配" if deviation < -0.01 else "平衡"
            print(
                f"{symbol:<8} "
                f"{current_weight:<11.1%} "
                f"{target_weight:<11.1%} "
                f"{deviation:<11.1%} "
                f"{status:<10}"
            )
    else:
        print("暂无偏差数据")

    print()

    # 步骤7: 检查是否需要再平衡
    print("步骤7: 检查再平衡需求")
    print("-" * 40)

    needs_rebalance = portfolio.needs_rebalancing(threshold=0.05)
    print(f"是否需要再平衡: {'是' if needs_rebalance else '否'}")

    if needs_rebalance:
        print("执行再平衡操作...")
        try:
            # 执行再平衡
            trades = portfolio.rebalance(portfolio.target_allocations)
            print(f"再平衡完成，执行了 {len(trades)} 笔交易")
        except Exception as e:
            print(f"再平衡失败: {e}")
    else:
        print("当前权重在阈值范围内，无需再平衡")

    print()

    # 步骤8: 查看绩效报告
    print("步骤8: 查看绩效报告")
    print("-" * 40)

    # 创建快照以记录当前状态
    snapshot = portfolio.take_snapshot()

    # 获取绩效摘要
    performance = portfolio.get_performance_summary()
    if performance:
        print("绩效摘要:")
        print(f"  总收益率: {performance.get('total_return', 0):.2%}")
        print(f"  年化收益率: {performance.get('annual_return', 0):.2%}")
        print(f"  波动率: {performance.get('volatility', 0):.2%}")
        print(f"  当前价值: ${performance.get('current_value', 0):,.2f}")
        print(f"  现金: ${performance.get('cash', 0):,.2f}")
        print(f"  持仓数量: {performance.get('num_holdings', 0)}")
        print(f"  持有天数: {performance.get('days_held', 0)}")
    else:
        print("暂无绩效数据（需要至少两个快照）")

    print()

    # 步骤9: 模拟价格变动后的效果
    print("步骤9: 模拟价格变动后的投资组合")
    print("-" * 40)

    # 模拟价格变动
    new_prices = {
        "AAPL": 180.00,  # 上涨
        "MSFT": 320.00,  # 下跌
        "GOOGL": 150.00  # 上涨
    }

    print("价格变动:")
    for symbol, old_price in current_prices.items():
        new_price = new_prices.get(symbol, old_price)
        change = (new_price - old_price) / old_price
        print(f"  {symbol}: ${old_price:.2f} → ${new_price:.2f} ({change:+.2%})")

    # 更新价格
    portfolio.update_prices(new_prices)

    # 查看更新后的状态
    new_total_value = portfolio.get_total_value()
    value_change = new_total_value - total_value
    print(f"\n投资组合价值变化: ${total_value:,.2f} → ${new_total_value:,.2f} ({value_change:+,.2f})")

    # 查看更新后的持仓盈亏
    print("\n更新后持仓盈亏:")
    for symbol, holding in portfolio.holdings.items():
        if symbol in new_prices:
            unrealized_pnl = holding.unrealized_pnl
            print(f"  {symbol}: ${unrealized_pnl:+,.2f}")

    print()

    # 步骤10: 执行卖出交易（部分平仓）
    print("步骤10: 执行卖出交易（部分平仓）")
    print("-" * 40)

    # 卖出部分AAPL持仓
    if "AAPL" in portfolio.holdings:
        aapl_holding = portfolio.holdings["AAPL"]
        sell_quantity = int(aapl_holding.quantity * 0.3)  # 卖出30%

        if sell_quantity > 0:
            try:
                portfolio.execute_trade(
                    symbol="AAPL",
                    quantity=-sell_quantity,  # 负数表示卖出
                    price=new_prices["AAPL"],
                    commission=0.001
                )
                print(f"✓ 卖出 {sell_quantity} 股 AAPL @ ${new_prices['AAPL']:.2f}")
                print(f"  实现盈亏: ${aapl_holding.realized_pnl:+,.2f}")
            except Exception as e:
                print(f"✗ 卖出 AAPL 失败: {e}")

    print()

    # 最终状态总结
    print("最终状态总结")
    print("=" * 80)
    print(f"投资组合名称: {portfolio.name}")
    print(f"最终总价值: ${portfolio.get_total_value():,.2f}")
    print(f"现金余额: ${portfolio.current_capital:,.2f}")
    print(f"持仓数量: {len(portfolio.holdings)}")
    print(f"交易总数: {len(portfolio.get_trade_history())}")
    print(f"快照数量: {len(portfolio.snapshots)}")
    print("示例运行完成！")

    return portfolio


if __name__ == "__main__":
    try:
        portfolio = run_basic_portfolio_example()
    except Exception as e:
        print(f"\n示例运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
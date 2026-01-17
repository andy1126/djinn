"""
Djinn 投资组合命令行接口模块。

本模块提供了投资组合管理的命令行接口，支持以下操作：
1. create: 创建新的投资组合
2. rebalance: 执行投资组合再平衡
3. analyze: 分析投资组合状态和绩效

实现方案概述：
1. 使用 argparse 处理命令行参数
2. 支持从配置文件或命令行参数创建投资组合
3. 集成投资组合工厂函数，支持多种创建方式
4. 提供详细的输出格式（文本、JSON、CSV）
5. 支持交互式操作和批量操作

主要函数：
- handle_portfolio_command: 主处理函数，根据action参数调用相应子函数
- create_portfolio: 创建投资组合
- rebalance_portfolio: 执行再平衡
- analyze_portfolio: 分析投资组合

参数解析逻辑：
1. 支持位置参数（action）和可选参数
2. 配置文件支持 YAML 和 JSON 格式
3. 参数优先级：命令行参数 > 配置文件 > 默认值
4. 支持环境变量覆盖配置

使用示例：
    djinn portfolio create --config portfolio_config.yaml
    djinn portfolio rebalance --config portfolio_config.yaml --threshold 0.1
    djinn portfolio analyze --config portfolio_config.yaml --output-format json
"""

import sys
import os
import json
import yaml
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import config_manager
from ..core.portfolio import (
    create_portfolio,
    create_portfolio_from_config_file,
    EquityPortfolio,
    RebalancingFrequency
)

logger = get_logger(__name__)


def handle_portfolio_command(args: argparse.Namespace) -> int:
    """
    处理投资组合命令。

    Args:
        args: 命令行参数

    Returns:
        int: 退出代码（0表示成功，非0表示失败）
    """
    action = args.action.lower()

    try:
        if action == 'create':
            return create_portfolio_cli(args)
        elif action == 'rebalance':
            return rebalance_portfolio_cli(args)
        elif action == 'analyze':
            return analyze_portfolio_cli(args)
        else:
            print(f"未知操作: {action}")
            print("可用操作: create, rebalance, analyze")
            return 1

    except Exception as e:
        logger.error(f"投资组合操作失败: {e}")
        print(f"错误: {e}")
        return 1


def create_portfolio_cli(args: argparse.Namespace) -> int:
    """
    创建投资组合。

    Args:
        args: 命令行参数

    Returns:
        int: 退出代码
    """
    try:
        print("创建投资组合...")

        # 从配置文件创建
        if args.config:
            print(f"使用配置文件: {args.config}")

            # 检查文件是否存在
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"配置文件不存在: {args.config}")
                return 1

            # 从配置文件创建投资组合
            portfolio = create_portfolio_from_config_file(args.config)

            if portfolio is None:
                print("投资组合功能未启用或配置错误")
                return 1

        # 从命令行参数创建
        elif args.initial_capital:
            print(f"使用命令行参数，初始资金: {args.initial_capital}")

            portfolio = create_portfolio(
                initial_capital=args.initial_capital,
                name=args.name or "CLI投资组合",
                currency=args.currency or "USD",
                rebalancing_frequency=args.rebalancing_frequency or "monthly",
                rebalancing_strategy=args.rebalancing_strategy or "equal_weight"
            )

        else:
            print("请提供配置文件(--config)或初始资金(--initial-capital)")
            return 1

        # 显示创建结果
        print(f"✓ 成功创建投资组合: {portfolio.name}")
        print(f"  组合ID: {id(portfolio)}")
        print(f"  初始资金: ${portfolio.initial_capital:,.2f} {portfolio.currency}")
        print(f"  货币: {portfolio.currency}")
        print(f"  再平衡频率: {portfolio.rebalancing_frequency.value}")
        print(f"  再平衡策略: {portfolio.rebalancing_strategy_type}")

        # 如果有初始配置，显示配置详情
        if portfolio.target_allocations:
            print(f"  目标配置数量: {len(portfolio.target_allocations)}")
            for alloc in portfolio.target_allocations:
                print(f"    - {alloc.symbol}: {alloc.target_weight:.1%}")

        # 保存组合信息到文件（如果指定了输出文件）
        if args.output:
            save_portfolio_info(portfolio, args.output, args.output_format)
            print(f"  组合信息已保存到: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"创建投资组合失败: {e}")
        print(f"错误: {e}")
        return 1


def rebalance_portfolio_cli(args: argparse.Namespace) -> int:
    """
    执行投资组合再平衡。

    Args:
        args: 命令行参数

    Returns:
        int: 退出代码
    """
    try:
        print("执行投资组合再平衡...")

        # 加载投资组合（简化：这里应该从文件或数据库加载）
        # 实际应用中，需要实现投资组合的持久化和加载
        if not args.config:
            print("请提供配置文件(--config)")
            return 1

        # 从配置文件重新创建投资组合（简化）
        portfolio = create_portfolio_from_config_file(args.config)
        if portfolio is None:
            print("无法加载投资组合")
            return 1

        print(f"投资组合: {portfolio.name}")
        print(f"当前总价值: ${portfolio.get_total_value():,.2f}")

        # 检查是否需要再平衡
        threshold = args.threshold or 0.05
        needs_rebalance = portfolio.needs_rebalancing(threshold)

        if not needs_rebalance:
            print(f"当前权重偏差未超过阈值({threshold:.1%})，无需再平衡")
            return 0

        print(f"检测到权重偏差超过阈值({threshold:.1%})，执行再平衡...")

        # 获取当前价格（需要实际数据，这里使用最后记录的价格）
        # 实际应用中应从数据源获取最新价格
        prices = {}
        for symbol, holding in portfolio.holdings.items():
            if hasattr(holding, 'current_price') and holding.current_price > 0:
                prices[symbol] = holding.current_price
            else:
                # 如果没有价格，使用默认值（实际应用应从数据源获取）
                prices[symbol] = 100.0

        # 执行再平衡
        trades = portfolio.rebalance(portfolio.target_allocations)

        if trades:
            print(f"✓ 再平衡完成，执行了 {len(trades)} 笔交易:")
            print(f"{'代码':<8} {'方向':<6} {'数量':<10} {'价值':<12} {'前权重':<8} {'后权重':<8}")
            print("-" * 70)

            total_trade_value = 0
            for trade in trades:
                symbol = trade['symbol']
                side = trade['side']
                quantity = trade['quantity']
                trade_value = trade['trade_value']
                current_weight = trade['current_weight']
                target_weight = trade['target_weight']

                total_trade_value += trade_value

                print(
                    f"{symbol:<8} "
                    f"{side:<6} "
                    f"{quantity:<10.0f} "
                    f"${trade_value:<11,.2f} "
                    f"{current_weight:<7.1%} "
                    f"{target_weight:<7.1%}"
                )

            print(f"\n总交易价值: ${total_trade_value:,.2f}")
            print(f"再平衡后总价值: ${portfolio.get_total_value():,.2f}")
            print(f"现金余额: ${portfolio.current_capital:,.2f}")

            # 保存再平衡结果
            if args.output:
                save_rebalance_result(portfolio, trades, args.output, args.output_format)
                print(f"再平衡结果已保存到: {args.output}")

        else:
            print("再平衡未产生交易（可能由于价格缺失或其他限制）")

        return 0

    except Exception as e:
        logger.error(f"再平衡失败: {e}")
        print(f"错误: {e}")
        return 1


def analyze_portfolio_cli(args: argparse.Namespace) -> int:
    """
    分析投资组合。

    Args:
        args: 命令行参数

    Returns:
        int: 退出代码
    """
    try:
        print("分析投资组合...")

        # 加载投资组合（简化）
        if not args.config:
            print("请提供配置文件(--config)")
            return 1

        portfolio = create_portfolio_from_config_file(args.config)
        if portfolio is None:
            print("无法加载投资组合")
            return 1

        print(f"投资组合: {portfolio.name}")
        print("=" * 80)

        # 基本概况
        print("基本概况:")
        print(f"  创建时间: {portfolio.created_date}")
        print(f"  状态: {portfolio.status.value}")
        print(f"  初始资金: ${portfolio.initial_capital:,.2f} {portfolio.currency}")
        print(f"  当前总价值: ${portfolio.get_total_value():,.2f}")
        print(f"  现金余额: ${portfolio.current_capital:,.2f}")
        print(f"  持仓价值: ${portfolio.get_holdings_value():,.2f}")
        print(f"  杠杆率: {portfolio.get_leverage():.2f}")
        print()

        # 持仓分析
        holdings = portfolio.holdings
        if holdings:
            print("持仓分析:")
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

            # 计算集中度
            weights = [holding.market_value / total_market_value for holding in holdings.values()]
            if weights:
                herfindahl = sum(w ** 2 for w in weights)
                print(f"\n赫芬达尔指数（集中度）: {herfindahl:.4f}")
                if herfindahl > 0.25:
                    print("⚠️  警告: 组合集中度较高")
        else:
            print("当前无持仓")
        print()

        # 配置偏差分析
        print("配置偏差分析:")
        deviations = portfolio.calculate_deviation()
        if deviations:
            print(f"{'代码':<8} {'当前权重':<12} {'目标权重':<12} {'偏差':<12} {'状态':<10}")
            print("-" * 60)

            current_weights = portfolio.calculate_weights()
            for symbol, deviation in deviations.items():
                current_weight = current_weights.get(symbol, 0)
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

            # 计算平均绝对偏差
            avg_abs_deviation = sum(abs(d) for d in deviations.values()) / len(deviations)
            print(f"\n平均绝对偏差: {avg_abs_deviation:.2%}")
        else:
            print("无目标配置或无法计算偏差")
        print()

        # 绩效分析
        print("绩效分析:")
        performance = portfolio.get_performance_summary()
        if performance:
            print(f"  总收益率: {performance.get('total_return', 0):.2%}")
            print(f"  年化收益率: {performance.get('annual_return', 0):.2%}")
            print(f"  波动率: {performance.get('volatility', 0):.2%}")
            print(f"  当前价值: ${performance.get('current_value', 0):,.2f}")
            print(f"  现金: ${performance.get('cash', 0):,.2f}")
            print(f"  持仓数量: {performance.get('num_holdings', 0)}")
            print(f"  持有天数: {performance.get('days_held', 0)}")
        else:
            print("  暂无绩效数据（需要至少两个快照）")
        print()

        # 风险管理
        print("风险管理检查:")
        print(f"  杠杆限制: {portfolio.check_leverage_limit()}")
        print(f"  头寸限制: {portfolio.check_position_limits()}")

        # 保存分析报告
        if args.output:
            save_analysis_report(portfolio, args.output, args.output_format)
            print(f"分析报告已保存到: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"分析失败: {e}")
        print(f"错误: {e}")
        return 1


def save_portfolio_info(portfolio: EquityPortfolio, output_path: str, format: str = 'text'):
    """
    保存投资组合信息到文件。

    Args:
        portfolio: 投资组合实例
        output_path: 输出文件路径
        format: 输出格式（text, json, csv）
    """
    try:
        data = portfolio.to_dict()

        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:  # text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"投资组合信息: {portfolio.name}\n")
                f.write("=" * 50 + "\n\n")
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")

    except Exception as e:
        logger.error(f"保存投资组合信息失败: {e}")


def save_rebalance_result(portfolio: EquityPortfolio, trades: list, output_path: str, format: str = 'text'):
    """
    保存再平衡结果到文件。

    Args:
        portfolio: 投资组合实例
        trades: 交易列表
        output_path: 输出文件路径
        format: 输出格式
    """
    try:
        data = {
            'portfolio_name': portfolio.name,
            'timestamp': portfolio.last_rebalanced.isoformat() if portfolio.last_rebalanced else None,
            'total_value': portfolio.get_total_value(),
            'cash_balance': portfolio.current_capital,
            'num_trades': len(trades),
            'trades': trades
        }

        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:  # text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"再平衡结果: {portfolio.name}\n")
                f.write(f"时间: {data['timestamp']}\n")
                f.write(f"总价值: ${data['total_value']:,.2f}\n")
                f.write(f"现金余额: ${data['cash_balance']:,.2f}\n")
                f.write(f"交易数量: {data['num_trades']}\n\n")
                f.write("交易明细:\n")
                f.write("-" * 60 + "\n")
                for trade in trades:
                    f.write(f"{trade['symbol']}: {trade['side']} {trade['quantity']} 股，"
                           f"价值: ${trade['trade_value']:,.2f}\n")

    except Exception as e:
        logger.error(f"保存再平衡结果失败: {e}")


def save_analysis_report(portfolio: EquityPortfolio, output_path: str, format: str = 'text'):
    """
    保存分析报告到文件。

    Args:
        portfolio: 投资组合实例
        output_path: 输出文件路径
        format: 输出格式
    """
    try:
        # 收集分析数据
        analysis_data = {
            'portfolio_info': portfolio.to_dict(),
            'holdings': [
                {
                    'symbol': h.symbol,
                    'quantity': h.quantity,
                    'avg_price': h.avg_price,
                    'current_price': h.current_price,
                    'market_value': h.market_value,
                    'unrealized_pnl': h.unrealized_pnl,
                    'realized_pnl': h.realized_pnl
                }
                for h in portfolio.holdings.values()
            ],
            'performance': portfolio.get_performance_summary(),
            'deviations': portfolio.calculate_deviation()
        }

        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(analysis_data, f, default_flow_style=False, allow_unicode=True)
        else:  # text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"投资组合分析报告: {portfolio.name}\n")
                f.write("=" * 60 + "\n\n")

                # 基本信息
                f.write("基本信息:\n")
                f.write("-" * 40 + "\n")
                for key, value in analysis_data['portfolio_info'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

                # 持仓信息
                if analysis_data['holdings']:
                    f.write("持仓信息:\n")
                    f.write("-" * 40 + "\n")
                    for holding in analysis_data['holdings']:
                        f.write(f"  {holding['symbol']}: {holding['quantity']} 股，"
                               f"市值: ${holding['market_value']:,.2f}，"
                               f"盈亏: ${holding['unrealized_pnl']:+,.2f}\n")
                    f.write("\n")

                # 绩效信息
                if analysis_data['performance']:
                    f.write("绩效信息:\n")
                    f.write("-" * 40 + "\n")
                    for key, value in analysis_data['performance'].items():
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.2%}\n")
                        else:
                            f.write(f"  {key}: {value}\n")

    except Exception as e:
        logger.error(f"保存分析报告失败: {e}")


def add_portfolio_arguments(parser: argparse.ArgumentParser):
    """
    添加投资组合命令的参数。

    Args:
        parser: ArgumentParser 实例
    """
    # 通用参数
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径 (YAML 或 JSON)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径"
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "yaml"],
        default="text",
        help="输出格式 (默认: text)"
    )

    # create 命令特有参数
    parser.add_argument(
        "--initial-capital",
        type=float,
        help="初始资金"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="组合名称"
    )
    parser.add_argument(
        "--currency",
        type=str,
        default="USD",
        help="货币代码 (默认: USD)"
    )
    parser.add_argument(
        "--rebalancing-frequency",
        choices=["daily", "weekly", "monthly", "quarterly", "yearly", "never"],
        help="再平衡频率"
    )
    parser.add_argument(
        "--rebalancing-strategy",
        choices=["equal_weight", "risk_parity", "mean_variance", "black_litterman"],
        help="再平衡策略"
    )

    # rebalance 命令特有参数
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="再平衡阈值 (默认: 0.05)"
    )
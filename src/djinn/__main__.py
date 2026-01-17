"""
Djinn 命令行入口点。

这个模块提供了从命令行运行 Djinn 的功能。
"""

import sys
import argparse
from typing import Optional, List

from . import __version__, get_version_info, set_log_level
from .cli.commands import main as cli_main

def main(args: Optional[List[str]] = None) -> int:
    """
    主函数，处理命令行参数并执行相应命令。

    Args:
        args: 命令行参数列表，如果为 None 则使用 sys.argv[1:]

    Returns:
        int: 退出代码
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Djinn - 多市场量化回测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --version
  %(prog)s --info
  %(prog)s run-backtest --config config.yaml
  %(prog)s optimize --strategy MovingAverageCrossover
        """
    )

    # 全局参数
    parser.add_argument(
        "--version",
        action="store_true",
        help="显示版本信息"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="显示详细系统信息"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="设置日志级别 (默认: INFO)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径"
    )

    # 子命令
    subparsers = parser.add_subparsers(
        dest="command",
        help="可用命令"
    )

    # backtest 命令
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="运行回测"
    )
    backtest_parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="策略名称"
    )
    backtest_parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="股票代码"
    )
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="开始日期 (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="结束日期 (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="初始资金 (默认: 100000)"
    )

    # optimize 命令
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="优化策略参数"
    )
    optimize_parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="策略名称"
    )
    optimize_parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="股票代码"
    )
    optimize_parser.add_argument(
        "--param-space",
        type=str,
        required=True,
        help="参数空间配置文件路径"
    )

    # data 命令
    data_parser = subparsers.add_parser(
        "data",
        help="数据管理"
    )
    data_parser.add_argument(
        "action",
        choices=["download", "update", "clean", "list"],
        help="操作类型"
    )
    data_parser.add_argument(
        "--symbol",
        type=str,
        help="股票代码"
    )
    data_parser.add_argument(
        "--market",
        choices=["US", "HK", "CN"],
        help="市场类型"
    )

    # portfolio 命令
    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="投资组合管理"
    )
    portfolio_parser.add_argument(
        "action",
        choices=["create", "rebalance", "analyze"],
        help="操作类型"
    )
    portfolio_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="组合配置文件"
    )

    # 如果没有提供参数，显示帮助
    if not args:
        parser.print_help()
        return 0

    parsed_args = parser.parse_args(args)

    # 设置日志级别
    set_log_level(parsed_args.log_level)

    # 处理全局参数
    if parsed_args.version:
        print(f"Djinn v{__version__}")
        return 0

    if parsed_args.info:
        info = get_version_info()
        print("Djinn 系统信息:")
        print(f"  版本: {info['djinn_version']}")
        print(f"  Python: {info['python_version']}")
        print(f"  平台: {info['platform']}")
        print("  依赖:")
        for dep, version in info['dependencies'].items():
            print(f"    {dep}: {version}")
        return 0

    # 如果没有指定命令，显示帮助
    if not parsed_args.command:
        parser.print_help()
        return 0

    # 调用 CLI 主函数
    try:
        return cli_main(parsed_args)
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        return 130
    except Exception as e:
        print(f"错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
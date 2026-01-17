"""
Djinn 命令行接口主模块。

这个模块提供了所有 CLI 命令的入口点和处理函数。
它负责将命令行参数分发给相应的命令处理器。

架构设计：
1. cli_main 函数是 CLI 的主入口点
2. 每个命令有独立的处理函数
3. 命令处理器负责具体的业务逻辑
4. 统一的错误处理和日志记录

支持的命令：
1. backtest: 运行回测
2. optimize: 优化策略参数
3. data: 数据管理
4. portfolio: 投资组合管理

模块职责：
- 参数验证和预处理
- 命令路由
- 错误处理和用户反馈
- 结果输出格式化
"""

import sys
import argparse
from typing import Dict, Any, Callable, Optional

from ..utils.logger import get_logger

# 导入命令处理器（避免循环导入）
try:
    from .portfolio import handle_portfolio_command
    HAS_PORTFOLIO = True
except ImportError:
    HAS_PORTFOLIO = False
    handle_portfolio_command = None

logger = get_logger(__name__)


def cli_main(args: argparse.Namespace) -> int:
    """
    CLI 主函数，处理所有命令。

    Args:
        args: 命令行参数

    Returns:
        int: 退出代码（0表示成功，非0表示失败）
    """
    command = args.command.lower()

    try:
        logger.debug(f"执行命令: {command}，参数: {vars(args)}")

        # 路由到相应的命令处理器
        if command == "backtest":
            return handle_backtest_command(args)
        elif command == "optimize":
            return handle_optimize_command(args)
        elif command == "data":
            return handle_data_command(args)
        elif command == "portfolio":
            return handle_portfolio_command(args)
        else:
            print(f"未知命令: {command}")
            print("可用命令: backtest, optimize, data, portfolio")
            return 1

    except KeyboardInterrupt:
        print("\n操作被用户中断")
        return 130
    except Exception as e:
        logger.error(f"命令执行失败: {e}", exc_info=True)
        print(f"错误: {e}")
        return 1


def handle_backtest_command(args: argparse.Namespace) -> int:
    """
    处理回测命令。

    Args:
        args: 命令行参数

    Returns:
        int: 退出代码
    """
    print(f"运行回测命令: {args}")
    print("回测功能正在开发中...")
    return 0


def handle_optimize_command(args: argparse.Namespace) -> int:
    """
    处理优化命令。

    Args:
        args: 命令行参数

    Returns:
        int: 退出代码
    """
    print(f"运行优化命令: {args}")
    print("优化功能正在开发中...")
    return 0


def handle_data_command(args: argparse.Namespace) -> int:
    """
    处理数据命令。

    Args:
        args: 命令行参数

    Returns:
        int: 退出代码
    """
    print(f"运行数据命令: {args}")
    print("数据功能正在开发中...")
    return 0


def get_command_help(command: Optional[str] = None) -> str:
    """
    获取命令帮助信息。

    Args:
        command: 命令名称，如果为 None 则显示所有命令

    Returns:
        str: 帮助信息
    """
    if command:
        # 返回特定命令的帮助
        commands = {
            "backtest": "运行策略回测",
            "optimize": "优化策略参数",
            "data": "数据管理（下载、更新、清理）",
            "portfolio": "投资组合管理（创建、再平衡、分析）"
        }

        if command in commands:
            return f"{command}: {commands[command]}"
        else:
            return f"未知命令: {command}"
    else:
        # 返回所有命令的帮助
        help_text = "可用命令:\n"
        help_text += "  backtest    运行策略回测\n"
        help_text += "  optimize    优化策略参数\n"
        help_text += "  data        数据管理（下载、更新、清理）\n"
        help_text += "  portfolio   投资组合管理（创建、再平衡、分析）\n"
        help_text += "\n使用 'djinn <命令> --help' 查看具体命令的用法"
        return help_text


# 导出
__all__ = [
    'cli_main',
    'get_command_help'
]
"""
Djinn 命令行接口模块。

这个模块提供了 Djinn 框架的所有命令行接口功能。
它导出 CLI 主函数和相关的工具函数。

主要导出：
- cli_main: CLI 主入口函数，处理所有命令
- get_command_help: 获取命令帮助信息的函数

使用方式：
    from djinn.cli import cli_main

    # 在脚本中调用
    args = parse_arguments()  # 解析参数
    exit_code = cli_main(args)
"""

from .commands import cli_main, get_command_help

__all__ = [
    'cli_main',
    'get_command_help'
]
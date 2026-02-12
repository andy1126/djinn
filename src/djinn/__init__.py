"""
Djinn - A multi-market quantitative backtesting framework for US, HK, and Chinese stocks.

Djinn provides a comprehensive backtesting framework with support for multiple markets,
portfolio management, and advanced analytics.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

import logging
from typing import Optional

from .utils.logger import setup_logger

# 设置默认日志器
logger = setup_logger(__name__)

# 简化策略框架的导入
from djinn.core.strategy import (
    SimpleStrategy,
    param,
    Parameter,
)

# 导出主要模块
__all__ = [
    # 数据模块
    "data",
    # 核心模块
    "core",
    # 可视化模块
    "visualization",
    # 命令行接口
    "cli",
    # 工具模块
    "utils",
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    "logger",
    # 简化策略框架
    "SimpleStrategy",
    "param",
    "Parameter",
]

# 配置日志级别
def set_log_level(level: str = "INFO") -> None:
    """
    设置全局日志级别。

    Args:
        level: 日志级别，可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.getLogger("djinn").setLevel(numeric_level)
    logger.info(f"Log level set to: {level}")

# 初始化检查
def check_dependencies() -> bool:
    """
    检查必要的依赖是否已安装。

    Returns:
        bool: 所有依赖是否可用
    """
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import akshare as ak
        import pydantic
        import loguru

        logger.debug("All core dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

# 版本信息
def get_version_info() -> dict:
    """
    获取版本和依赖信息。

    Returns:
        dict: 包含版本和依赖信息的字典
    """
    import sys
    import platform

    info = {
        "djinn_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependencies": {}
    }

    # 尝试获取主要依赖的版本
    try:
        import pandas as pd
        info["dependencies"]["pandas"] = pd.__version__
    except ImportError:
        info["dependencies"]["pandas"] = "Not installed"

    try:
        import numpy as np
        info["dependencies"]["numpy"] = np.__version__
    except ImportError:
        info["dependencies"]["numpy"] = "Not installed"

    try:
        import yfinance as yf
        info["dependencies"]["yfinance"] = yf.__version__
    except ImportError:
        info["dependencies"]["yfinance"] = "Not installed"

    try:
        import akshare as ak
        info["dependencies"]["akshare"] = ak.__version__
    except ImportError:
        info["dependencies"]["akshare"] = "Not installed"

    return info

# 在导入时检查依赖
if not check_dependencies():
    logger.warning(
        "Some dependencies may be missing. "
        "Run 'pip install djinn[dev]' to install all dependencies."
    )
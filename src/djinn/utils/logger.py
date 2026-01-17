"""
Djinn 日志系统。

这个模块提供了统一的日志配置和管理功能。
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from loguru import logger as loguru_logger

# 移除默认的 loguru 处理器
loguru_logger.remove()

# 自定义日志级别
LOG_LEVELS = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

# 添加自定义日志级别
for level_name, level_value in LOG_LEVELS.items():
    logging.addLevelName(level_value, level_name)


class InterceptHandler(logging.Handler):
    """拦截标准 logging 日志并转发到 loguru。"""

    def emit(self, record: logging.LogRecord) -> None:
        """发射日志记录到 loguru。"""
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 获取调用者信息
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(
    name: str = "djinn",
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "1 day",
    retention: str = "30 days",
    compression: str = "zip",
    format_str: Optional[str] = None,
    intercept_standard_logging: bool = True,
) -> loguru_logger:
    """
    设置和配置日志器。

    Args:
        name: 日志器名称
        level: 日志级别 (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为 None 则不写入文件
        rotation: 日志轮转规则
        retention: 日志保留时间
        compression: 日志压缩格式
        format_str: 自定义日志格式字符串
        intercept_standard_logging: 是否拦截标准 logging 日志

    Returns:
        配置好的 loguru 日志器
    """
    # 移除所有现有的处理器，避免重复日志
    loguru_logger.remove()

    # 设置日志级别
    log_level = level.upper()
    if log_level not in LOG_LEVELS:
        raise ValueError(f"无效的日志级别: {level}")

    # 默认日志格式
    if format_str is None:
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # 控制台处理器配置
    console_config = {
        "sink": sys.stderr,
        "level": log_level,
        "format": format_str,
        "colorize": True,
        "backtrace": True,
        "diagnose": True,
    }

    # 添加控制台处理器
    loguru_logger.add(**console_config)

    # 添加文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_config = {
            "sink": str(log_path),
            "level": log_level,
            "format": format_str,
            "rotation": rotation,
            "retention": retention,
            "compression": compression,
            "backtrace": True,
            "diagnose": True,
        }

        loguru_logger.add(**file_config)

    # 拦截标准 logging 日志
    if intercept_standard_logging:
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        # 为常用库设置日志级别
        for library in ["urllib3", "requests", "yfinance", "akshare"]:
            logging.getLogger(library).setLevel(logging.WARNING)

    return loguru_logger.bind(name=name)


def get_logger(name: str) -> loguru_logger:
    """
    获取指定名称的日志器。

    Args:
        name: 日志器名称

    Returns:
        loguru 日志器实例
    """
    return loguru_logger.bind(name=name)


class PerformanceLogger:
    """性能日志记录器。

    用于记录和跟踪性能指标。
    """

    def __init__(self, name: str = "performance"):
        """
        初始化性能日志记录器。

        Args:
            name: 日志器名称
        """
        self.logger = get_logger(name)
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def start_timer(self, operation: str) -> None:
        """
        开始计时一个操作。

        Args:
            operation: 操作名称
        """
        if operation in self.metrics:
            self.logger.warning(f"操作 '{operation}' 的计时器已经存在，将重新开始")

        self.metrics[operation] = {
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None,
        }
        self.logger.debug(f"开始操作: {operation}")

    def stop_timer(self, operation: str) -> float:
        """
        停止计时一个操作并返回持续时间。

        Args:
            operation: 操作名称

        Returns:
            float: 操作持续时间（秒）

        Raises:
            KeyError: 如果操作不存在
        """
        if operation not in self.metrics:
            raise KeyError(f"操作 '{operation}' 的计时器不存在")

        end_time = datetime.now()
        start_time = self.metrics[operation]["start_time"]
        duration = (end_time - start_time).total_seconds()

        self.metrics[operation].update(
            {"end_time": end_time, "duration": duration}
        )

        self.logger.debug(
            f"完成操作: {operation}, 耗时: {duration:.4f} 秒"
        )

        return duration

    def log_metric(self, name: str, value: Any, unit: str = "") -> None:
        """
        记录一个性能指标。

        Args:
            name: 指标名称
            value: 指标值
            unit: 指标单位
        """
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"性能指标 - {name}: {value}{unit_str}")

    def log_memory_usage(self) -> None:
        """记录当前内存使用情况。"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            self.logger.info(
                f"内存使用 - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, "
                f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB"
            )
        except ImportError:
            self.logger.warning("无法记录内存使用情况，请安装 psutil 库")

    def get_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要。

        Returns:
            Dict[str, Any]: 性能摘要字典
        """
        summary = {}
        for operation, metrics in self.metrics.items():
            if metrics["duration"] is not None:
                summary[operation] = {
                    "duration_seconds": metrics["duration"],
                    "start_time": metrics["start_time"].isoformat(),
                    "end_time": metrics["end_time"].isoformat(),
                }

        return summary


# 默认日志器实例
logger = setup_logger()

# 性能日志器实例
performance_logger = PerformanceLogger()


def log_exception(
    exception: Exception,
    message: Optional[str] = None,
    level: str = "ERROR",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    记录异常信息。

    Args:
        exception: 异常实例
        message: 自定义消息
        level: 日志级别
        context: 额外的上下文信息
    """
    log_func = getattr(logger, level.lower())

    if message:
        log_msg = f"{message}: {str(exception)}"
    else:
        log_msg = f"异常: {str(exception)}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        log_msg = f"{log_msg} [{context_str}]"

    log_func(log_msg, exc_info=exception)


# 导出
__all__ = [
    "setup_logger",
    "get_logger",
    "logger",
    "PerformanceLogger",
    "performance_logger",
    "log_exception",
    "LOG_LEVELS",
]
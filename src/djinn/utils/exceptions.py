"""
Djinn 异常处理模块。

这个模块定义了 Djinn 框架中使用的所有自定义异常类。
"""

from typing import Optional, Any


class DjinnError(Exception):
    """Djinn 基础异常类。

    所有 Djinn 自定义异常的基类。
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        初始化异常。

        Args:
            message: 错误消息
            details: 额外的错误详情
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """返回异常的字符串表示。"""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{details_str}]"
        return self.message


class DataError(DjinnError):
    """数据相关异常。

    当数据获取、处理或验证失败时抛出。
    """

    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化数据异常。

        Args:
            message: 错误消息
            data_source: 数据源名称
            symbol: 股票代码
            details: 额外的错误详情
        """
        details = details or {}
        if data_source:
            details["data_source"] = data_source
        if symbol:
            details["symbol"] = symbol

        super().__init__(f"数据错误: {message}", details)


class StrategyError(DjinnError):
    """策略相关异常。

    当策略初始化、执行或验证失败时抛出。
    """

    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        parameters: Optional[dict] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化策略异常。

        Args:
            message: 错误消息
            strategy_name: 策略名称
            parameters: 策略参数
            details: 额外的错误详情
        """
        details = details or {}
        if strategy_name:
            details["strategy"] = strategy_name
        if parameters:
            details["parameters"] = parameters

        super().__init__(f"策略错误: {message}", details)


class BacktestError(DjinnError):
    """回测相关异常。

    当回测引擎执行失败时抛出。
    """

    def __init__(
        self,
        message: str,
        backtest_id: Optional[str] = None,
        step: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化回测异常。

        Args:
            message: 错误消息
            backtest_id: 回测ID
            step: 失败步骤
            details: 额外的错误详情
        """
        details = details or {}
        if backtest_id:
            details["backtest_id"] = backtest_id
        if step:
            details["step"] = step

        super().__init__(f"回测错误: {message}", details)


class PortfolioError(DjinnError):
    """投资组合相关异常。

    当投资组合管理失败时抛出。
    """

    def __init__(
        self,
        message: str,
        portfolio_id: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化投资组合异常。

        Args:
            message: 错误消息
            portfolio_id: 组合ID
            operation: 失败操作
            details: 额外的错误详情
        """
        details = details or {}
        if portfolio_id:
            details["portfolio_id"] = portfolio_id
        if operation:
            details["operation"] = operation

        super().__init__(f"投资组合错误: {message}", details)


class ValidationError(DjinnError):
    """数据验证异常。

    当数据验证失败时抛出。
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[Any] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化验证异常。

        Args:
            message: 错误消息
            field: 验证失败的字段
            value: 实际值
            expected: 期望值
            details: 额外的错误详情
        """
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["actual_value"] = value
        if expected is not None:
            details["expected_value"] = expected

        super().__init__(f"验证错误: {message}", details)


class ConfigurationError(DjinnError):
    """配置相关异常。

    当配置加载、验证或解析失败时抛出。
    """

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        section: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化配置异常。

        Args:
            message: 错误消息
            config_file: 配置文件路径
            section: 配置节
            details: 额外的错误详情
        """
        details = details or {}
        if config_file:
            details["config_file"] = config_file
        if section:
            details["section"] = section

        super().__init__(f"配置错误: {message}", details)


class NetworkError(DjinnError):
    """网络相关异常。

    当网络请求失败时抛出。
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化网络异常。

        Args:
            message: 错误消息
            url: 请求的URL
            status_code: HTTP状态码
            details: 额外的错误详情
        """
        details = details or {}
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code

        super().__init__(f"网络错误: {message}", details)


class CacheError(DjinnError):
    """缓存相关异常。

    当缓存操作失败时抛出。
    """

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        cache_type: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化缓存异常。

        Args:
            message: 错误消息
            cache_key: 缓存键
            cache_type: 缓存类型
            details: 额外的错误详情
        """
        details = details or {}
        if cache_key:
            details["cache_key"] = cache_key
        if cache_type:
            details["cache_type"] = cache_type

        super().__init__(f"缓存错误: {message}", details)


class PerformanceError(DjinnError):
    """性能相关异常。

    当性能指标计算失败时抛出。
    """

    def __init__(
        self,
        message: str,
        metric: Optional[str] = None,
        data_length: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化性能异常。

        Args:
            message: 错误消息
            metric: 性能指标名称
            data_length: 数据长度
            details: 额外的错误详情
        """
        details = details or {}
        if metric:
            details["metric"] = metric
        if data_length is not None:
            details["data_length"] = data_length

        super().__init__(f"性能计算错误: {message}", details)


# 异常处理工具函数
def handle_exception(
    exception: Exception,
    context: Optional[dict] = None,
    raise_original: bool = False,
) -> DjinnError:
    """
    处理异常，将其转换为 DjinnError 或重新抛出。

    Args:
        exception: 原始异常
        context: 额外的上下文信息
        raise_original: 是否重新抛出原始异常

    Returns:
        DjinnError: 转换后的异常

    Raises:
        Exception: 如果 raise_original 为 True
    """
    if raise_original:
        raise exception

    # 如果已经是 DjinnError，直接返回
    if isinstance(exception, DjinnError):
        if context:
            exception.details.update(context)
        return exception

    # 根据异常类型转换为相应的 DjinnError
    context = context or {}
    error_message = str(exception)

    if isinstance(exception, (ValueError, TypeError)):
        return ValidationError(
            f"数据验证失败: {error_message}",
            details={"original_error": error_message, **context},
        )
    elif isinstance(exception, (KeyError, AttributeError)):
        return ConfigurationError(
            f"配置错误: {error_message}",
            details={"original_error": error_message, **context},
        )
    elif isinstance(exception, (ConnectionError, TimeoutError)):
        return NetworkError(
            f"网络连接失败: {error_message}",
            details={"original_error": error_message, **context},
        )
    elif isinstance(exception, MemoryError):
        return DjinnError(
            f"内存不足: {error_message}",
            details={"original_error": error_message, **context},
        )
    else:
        # 其他异常转换为通用的 DjinnError
        return DjinnError(
            f"未处理的错误: {error_message}",
            details={
                "original_error": error_message,
                "exception_type": type(exception).__name__,
                **context,
            },
        )


class IndicatorError(DjinnError):
    """指标计算相关异常。

    当技术指标计算失败时抛出。
    """

    def __init__(
        self,
        message: str,
        indicator_name: Optional[str] = None,
        parameters: Optional[dict] = None,
        data_length: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化指标异常。

        Args:
            message: 错误消息
            indicator_name: 指标名称
            parameters: 指标计算参数
            data_length: 数据长度
            details: 额外的错误详情
        """
        details = details or {}
        if indicator_name:
            details["indicator"] = indicator_name
        if parameters:
            details["parameters"] = parameters
        if data_length is not None:
            details["data_length"] = data_length

        super().__init__(f"指标计算错误: {message}", details)


class RiskError(DjinnError):
    """风险相关异常。

    当风险管理、风险指标计算或风险控制失败时抛出。
    """

    def __init__(
        self,
        message: str,
        risk_metric: Optional[str] = None,
        threshold: Optional[float] = None,
        actual_value: Optional[float] = None,
        portfolio_id: Optional[str] = None,
        position_size: Optional[float] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化风险异常。

        Args:
            message: 错误消息
            risk_metric: 风险指标名称（如 VaR、波动率、最大回撤等）
            threshold: 风险阈值
            actual_value: 实际风险值
            portfolio_id: 投资组合ID
            position_size: 头寸规模
            details: 额外的错误详情
        """
        details = details or {}
        if risk_metric:
            details["risk_metric"] = risk_metric
        if threshold is not None:
            details["threshold"] = threshold
        if actual_value is not None:
            details["actual_value"] = actual_value
        if portfolio_id:
            details["portfolio_id"] = portfolio_id
        if position_size is not None:
            details["position_size"] = position_size

        super().__init__(f"风险错误: {message}", details)
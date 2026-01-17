"""
Djinn 配置管理模块。

这个模块提供了统一的配置加载、验证和管理功能。
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum

import pydantic
from pydantic import BaseModel, Field, validator, root_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logger import logger
from .exceptions import ConfigurationError


class MarketType(str, Enum):
    """市场类型枚举。"""

    US = "US"  # 美股
    HK = "HK"  # 港股
    CN = "CN"  # A股


class DataSourceType(str, Enum):
    """数据源类型枚举。"""

    YAHOO_FINANCE = "yahoo_finance"
    AKSHARE = "akshare"
    TUSHARE = "tushare"
    LOCAL = "local"


class IntervalType(str, Enum):
    """数据间隔类型枚举。"""

    DAILY = "1d"
    HOURLY = "1h"
    MINUTE_30 = "30m"
    MINUTE_15 = "15m"
    MINUTE_5 = "5m"
    MINUTE_1 = "1m"


class PositionSizingMethod(str, Enum):
    """仓位管理方法枚举。"""

    FIXED_FRACTIONAL = "fixed_fractional"
    FIXED_UNITS = "fixed_units"
    PERCENT_RISK = "percent_risk"
    KELLY = "kelly"


class RebalanceFrequency(str, Enum):
    """再平衡频率枚举。"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class AllocationMethod(str, Enum):
    """资产分配方法枚举。"""

    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP = "market_cap"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"


class OptimizationMethod(str, Enum):
    """优化方法枚举。"""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"


# Pydantic 数据模型
class RiskConfig(BaseModel):
    """风险配置模型。"""

    max_position_size: float = Field(0.1, ge=0.0, le=1.0, description="最大单仓位比例")
    stop_loss: float = Field(0.1, ge=0.0, le=1.0, description="止损比例")
    take_profit: float = Field(0.2, ge=0.0, le=1.0, description="止盈比例")
    max_drawdown: float = Field(0.2, ge=0.0, le=1.0, description="最大回撤限制")
    volatility_limit: float = Field(0.3, ge=0.0, le=1.0, description="波动率限制")

    @validator("take_profit")
    def validate_take_profit(cls, v, values):
        """验证止盈比例大于止损比例。"""
        if "stop_loss" in values and v <= values["stop_loss"]:
            raise ValueError("止盈比例必须大于止损比例")
        return v


class OrderTypesConfig(BaseModel):
    """订单类型配置模型。"""

    market_order: bool = Field(True, description="是否允许市价单")
    limit_order: bool = Field(True, description="是否允许限价单")
    stop_order: bool = Field(True, description="是否允许止损单")
    stop_limit_order: bool = Field(False, description="是否允许止损限价单")


class PerformanceMetricsConfig(BaseModel):
    """绩效指标配置模型。"""

    metrics: List[str] = Field(
        default_factory=lambda: [
            "total_return",
            "annual_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "omega_ratio",
            "win_rate",
            "profit_factor",
            "recovery_factor",
        ],
        description="要计算的绩效指标列表",
    )


class ReportConfig(BaseModel):
    """报告配置模型。"""

    format: str = Field("html", description="报告格式")
    save_path: str = Field("./reports", description="报告保存路径")
    include_charts: bool = Field(True, description="是否包含图表")
    include_trades: bool = Field(True, description="是否包含交易记录")
    include_metrics: bool = Field(True, description="是否包含绩效指标")


class PositionSizingConfig(BaseModel):
    """仓位管理配置模型。"""

    method: PositionSizingMethod = Field(
        PositionSizingMethod.FIXED_FRACTIONAL, description="仓位管理方法"
    )
    risk_per_trade: float = Field(0.02, ge=0.0, le=1.0, description="每笔交易风险")
    max_risk: float = Field(0.1, ge=0.0, le=1.0, description="最大总风险")


class StrategyConfig(BaseModel):
    """策略配置模型。"""

    name: str = Field(..., description="策略名称")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="策略参数")
    position_sizing: PositionSizingConfig = Field(
        default_factory=PositionSizingConfig, description="仓位管理配置"
    )


class PortfolioConfig(BaseModel):
    """投资组合配置模型。"""

    enabled: bool = Field(False, description="是否启用投资组合")
    rebalance_frequency: RebalanceFrequency = Field(
        RebalanceFrequency.MONTHLY, description="再平衡频率"
    )
    allocation_method: AllocationMethod = Field(
        AllocationMethod.EQUAL_WEIGHT, description="分配方法"
    )


class OptimizationConfig(BaseModel):
    """优化配置模型。"""

    enabled: bool = Field(False, description="是否启用优化")
    method: OptimizationMethod = Field(
        OptimizationMethod.GRID_SEARCH, description="优化方法"
    )
    param_space: Dict[str, List[Any]] = Field(
        default_factory=dict, description="参数空间"
    )
    metric: str = Field("sharpe_ratio", description="优化目标指标")
    n_trials: int = Field(100, ge=1, description="试验次数")


class BacktestConfig(BaseModel):
    """回测配置模型。"""

    # 基本参数
    initial_capital: float = Field(100000.0, gt=0.0, description="初始资金")
    commission: float = Field(0.001, ge=0.0, le=1.0, description="佣金率")
    slippage: float = Field(0.0005, ge=0.0, le=1.0, description="滑点率")
    tax_rate: float = Field(0.001, ge=0.0, le=1.0, description="印花税率")

    # 数据参数
    data_source: DataSourceType = Field(
        DataSourceType.YAHOO_FINANCE, description="数据源"
    )
    interval: IntervalType = Field(IntervalType.DAILY, description="数据间隔")
    adjust: str = Field("adj", description="价格调整方式")

    # 时间范围
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")

    # 缓存配置
    cache_enabled: bool = Field(True, description="是否启用缓存")
    cache_ttl: int = Field(3600, ge=0, description="缓存过期时间(秒)")

    # 子配置
    risk: RiskConfig = Field(default_factory=RiskConfig, description="风险配置")
    order_types: OrderTypesConfig = Field(
        default_factory=OrderTypesConfig, description="订单类型配置"
    )
    performance: PerformanceMetricsConfig = Field(
        default_factory=PerformanceMetricsConfig, description="绩效指标配置"
    )
    report: ReportConfig = Field(default_factory=ReportConfig, description="报告配置")

    # 策略配置
    strategy: StrategyConfig = Field(..., description="策略配置")

    # 可选配置
    portfolio: Optional[PortfolioConfig] = Field(None, description="投资组合配置")
    optimization: Optional[OptimizationConfig] = Field(None, description="优化配置")

    @validator("end_date")
    def validate_dates(cls, v, values):
        """验证日期范围。"""
        if "start_date" in values:
            from datetime import datetime

            try:
                start = datetime.strptime(values["start_date"], "%Y-%m-%d")
                end = datetime.strptime(v, "%Y-%m-%d")

                if end <= start:
                    raise ValueError("结束日期必须晚于开始日期")

                # 检查日期范围是否合理（不超过20年）
                if (end - start).days > 365 * 20:
                    logger.warning("回测时间范围超过20年，可能会影响性能")

            except ValueError as e:
                raise ValueError(f"日期格式错误或无效: {e}")

        return v


class Settings(BaseSettings):
    """应用设置模型。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 数据源配置
    tushare_token: Optional[str] = Field(None, description="Tushare token")

    # 缓存配置
    cache_dir: str = Field("./.cache/djinn", description="缓存目录")
    cache_ttl: int = Field(3600, description="缓存过期时间(秒)")

    # Redis配置
    redis_host: str = Field("localhost", description="Redis主机")
    redis_port: int = Field(6379, description="Redis端口")
    redis_password: Optional[str] = Field(None, description="Redis密码")
    redis_db: int = Field(0, description="Redis数据库")

    # 日志配置
    log_level: str = Field("INFO", description="日志级别")
    log_file: Optional[str] = Field(None, description="日志文件路径")

    # 性能配置
    max_workers: int = Field(4, description="最大并行工作线程数")
    chunk_size: int = Field(1000, description="数据分块大小")

    # 代理配置
    http_proxy: Optional[str] = Field(None, description="HTTP代理")
    https_proxy: Optional[str] = Field(None, description="HTTPS代理")

    # 开发配置
    debug: bool = Field(False, description="调试模式")
    test_mode: bool = Field(False, description="测试模式")


class ConfigManager:
    """配置管理器。"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置管理器。

        Args:
            config_dir: 配置文件目录，如果为 None 则使用默认目录
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)

        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 加载环境设置
        self.settings = Settings()

        # 缓存已加载的配置
        self._config_cache: Dict[str, Any] = {}

        logger.info(f"配置管理器初始化完成，配置目录: {self.config_dir}")

    def load_config(
        self,
        config_file: str,
        config_type: str = "yaml",
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        加载配置文件。

        Args:
            config_file: 配置文件路径或名称
            config_type: 配置类型 (yaml, json)
            validate: 是否验证配置

        Returns:
            Dict[str, Any]: 配置字典

        Raises:
            ConfigurationError: 如果配置加载或验证失败
        """
        # 检查缓存
        cache_key = f"{config_file}:{config_type}"
        if cache_key in self._config_cache:
            logger.debug(f"从缓存加载配置: {cache_key}")
            return self._config_cache[cache_key]

        # 构建完整路径
        config_path = self._get_config_path(config_file, config_type)

        try:
            # 加载配置文件
            if config_type.lower() == "yaml":
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
            elif config_type.lower() == "json":
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            else:
                raise ConfigurationError(f"不支持的配置类型: {config_type}")

            # 验证配置
            if validate:
                config_data = self._validate_config(config_data, config_file)

            # 缓存配置
            self._config_cache[cache_key] = config_data

            logger.info(f"成功加载配置: {config_path}")
            return config_data

        except Exception as e:
            raise ConfigurationError(
                f"加载配置文件失败: {config_path}",
                config_file=str(config_path),
                details={"error": str(e)},
            )

    def load_backtest_config(self, config_file: str) -> BacktestConfig:
        """
        加载回测配置。

        Args:
            config_file: 回测配置文件路径或名称

        Returns:
            BacktestConfig: 回测配置对象
        """
        config_data = self.load_config(config_file, "yaml")

        try:
            # 提取回测配置部分
            backtest_data = config_data.get("backtest", config_data)

            # 创建配置对象
            config = BacktestConfig(**backtest_data)

            logger.info(f"成功加载回测配置: {config_file}")
            return config

        except pydantic.ValidationError as e:
            raise ConfigurationError(
                f"回测配置验证失败: {config_file}",
                config_file=config_file,
                details={"validation_errors": str(e)},
            )

    def save_config(
        self,
        config_data: Dict[str, Any],
        config_file: str,
        config_type: str = "yaml",
    ) -> None:
        """
        保存配置到文件。

        Args:
            config_data: 配置数据
            config_file: 配置文件路径或名称
            config_type: 配置类型 (yaml, json)

        Raises:
            ConfigurationError: 如果配置保存失败
        """
        config_path = self._get_config_path(config_file, config_type)

        try:
            # 确保目录存在
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存配置文件
            if config_type.lower() == "yaml":
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            elif config_type.lower() == "json":
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            else:
                raise ConfigurationError(f"不支持的配置类型: {config_type}")

            # 更新缓存
            cache_key = f"{config_file}:{config_type}"
            self._config_cache[cache_key] = config_data

            logger.info(f"成功保存配置: {config_path}")

        except Exception as e:
            raise ConfigurationError(
                f"保存配置文件失败: {config_path}",
                config_file=str(config_path),
                details={"error": str(e)},
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        获取环境设置。

        Args:
            key: 设置键名
            default: 默认值

        Returns:
            Any: 设置值
        """
        return getattr(self.settings, key, default)

    def update_setting(self, key: str, value: Any) -> None:
        """
        更新环境设置。

        Args:
            key: 设置键名
            value: 设置值
        """
        setattr(self.settings, key, value)
        logger.debug(f"更新设置: {key} = {value}")

    def clear_cache(self) -> None:
        """清除配置缓存。"""
        self._config_cache.clear()
        logger.debug("配置缓存已清除")

    def _get_config_path(
        self, config_file: str, config_type: str
    ) -> Path:
        """
        获取配置文件的完整路径。

        Args:
            config_file: 配置文件路径或名称
            config_type: 配置类型

        Returns:
            Path: 完整配置文件路径
        """
        config_path = Path(config_file)

        # 如果提供了相对路径或文件名，添加扩展名并查找在配置目录中
        if not config_path.is_absolute():
            # 添加扩展名
            if not config_path.suffix:
                config_path = config_path.with_suffix(f".{config_type}")

            # 在配置目录中查找
            config_path = self.config_dir / config_path

        return config_path

    def _validate_config(
        self, config_data: Dict[str, Any], config_file: str
    ) -> Dict[str, Any]:
        """
        验证配置数据。

        Args:
            config_data: 配置数据
            config_file: 配置文件路径

        Returns:
            Dict[str, Any]: 验证后的配置数据

        Raises:
            ConfigurationError: 如果配置验证失败
        """
        # 这里可以添加自定义的配置验证逻辑
        # 目前主要依赖 Pydantic 的验证

        # 检查必需字段
        required_fields = []
        if "backtest" in config_data:
            backtest_data = config_data["backtest"]
            if "start_date" not in backtest_data:
                required_fields.append("backtest.start_date")
            if "end_date" not in backtest_data:
                required_fields.append("backtest.end_date")

        if required_fields:
            raise ConfigurationError(
                f"配置缺少必需字段: {', '.join(required_fields)}",
                config_file=config_file,
                details={"missing_fields": required_fields},
            )

        return config_data


# 全局配置管理器实例
config_manager = ConfigManager()


# 导出
__all__ = [
    # 枚举类型
    "MarketType",
    "DataSourceType",
    "IntervalType",
    "PositionSizingMethod",
    "RebalanceFrequency",
    "AllocationMethod",
    "OptimizationMethod",
    # 配置模型
    "RiskConfig",
    "OrderTypesConfig",
    "PerformanceMetricsConfig",
    "ReportConfig",
    "PositionSizingConfig",
    "StrategyConfig",
    "PortfolioConfig",
    "OptimizationConfig",
    "BacktestConfig",
    "Settings",
    # 配置管理器
    "ConfigManager",
    "config_manager",
]
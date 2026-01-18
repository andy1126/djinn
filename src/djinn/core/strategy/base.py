"""
Djinn量化回测框架的策略基类模块。

本模块定义了策略的抽象基类、信号生成、头寸规模管理等核心组件，为所有交易策略提供统一接口和通用功能。

目的：
1. 提供策略开发的标准化框架，确保策略接口一致性
2. 封装信号生成、头寸计算、指标计算等通用逻辑
3. 支持多种仓位管理方法和风险管理规则

实现方案：
1. 抽象基类（Strategy）：定义策略生命周期方法（初始化、信号生成、指标计算）
2. 信号系统（Signal, SignalType）：封装交易信号及其元数据
3. 仓位管理（PositionSizing）：支持固定分数、固定单位、百分比风险、凯利公式等多种仓位计算方法
4. 参数验证：自动验证策略参数和风险参数的有效性

使用方法：
1. 继承Strategy基类实现自定义策略，重写initialize、generate_signals、calculate_indicators方法
2. 使用Signal类创建交易信号，包含时间戳、代码、信号类型、价格、强度等信息
3. 配置PositionSizing对象定义仓位管理规则
4. 调用update方法更新策略状态并生成信号
5. 使用get_summary、get_signals_df、get_indicators_df方法获取策略状态和结果
"""

import abc
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
import warnings

from ...utils.logger import logger
from ...utils.exceptions import StrategyError, ValidationError
from ...utils.validation import Validator
from ...data.market_data import MarketData, OHLCV


class SignalType(str, Enum):
    """
    信号类型枚举类。

    定义交易策略生成的信号类型，包括买入、卖出、持有和退出信号。

    目的：
    1. 标准化信号类型，确保策略间信号一致性
    2. 提供明确的交易指令类型，便于回测引擎处理
    3. 支持信号统计和分析

    实现方案：
    1. 继承str和Enum，确保信号类型既是字符串又是枚举值
    2. 定义四种基本信号类型：BUY（买入）、SELL（卖出）、HOLD（持有）、EXIT（退出）

    使用方法：
    1. 创建信号时指定signal_type参数：Signal(signal_type=SignalType.BUY, ...)
    2. 比较信号类型：if signal.signal_type == SignalType.BUY:
    3. 获取枚举值字符串：SignalType.BUY.value 返回 "BUY"
    """

    BUY = "BUY"  # 买入信号
    SELL = "SELL"  # 卖出信号
    HOLD = "HOLD"  # 持有信号
    EXIT = "EXIT"  # 退出信号


class PositionType(str, Enum):
    """
    仓位类型枚举类。

    定义交易仓位的方向类型，包括多头和空头仓位。

    目的：
    1. 标准化仓位方向类型，支持多空策略
    2. 明确仓位方向，便于风险管理和损益计算
    3. 为后续扩展其他仓位类型提供基础

    实现方案：
    1. 继承str和Enum，确保仓位类型既是字符串又是枚举值
    2. 定义两种基本仓位类型：LONG（多头）、SHORT（空头）

    使用方法：
    1. 在策略中标识仓位方向：position_type=PositionType.LONG
    2. 根据仓位类型调整风险计算和止损策略
    3. 获取枚举值字符串：PositionType.LONG.value 返回 "LONG"
    """

    LONG = "LONG"  # 多头仓位
    SHORT = "SHORT"  # 空头仓位


@dataclass
class Signal:
    """
    交易信号数据类。

    封装交易策略生成的交易信号，包含信号的所有必要信息和元数据。

    目的：
    1. 标准化信号数据结构，确保信号信息完整性和一致性
    2. 提供信号验证机制，确保信号数据的有效性
    3. 支持信号序列化和反序列化，便于存储和传输

    实现方案：
    1. 使用@dataclass装饰器自动生成初始化方法、比较方法等
    2. 包含时间戳、股票代码、信号类型、价格、信号强度、原因和元数据等字段
    3. 通过__post_init__方法实现自动数据验证
    4. 提供to_dict和from_dict方法支持字典转换

    使用方法：
    1. 直接实例化：signal = Signal(timestamp=..., symbol=..., signal_type=..., price=...)
    2. 从字典创建：signal = Signal.from_dict(data_dict)
    3. 转换为字典：data_dict = signal.to_dict()
    4. 信号强度范围：0.0（弱）到1.0（强），默认1.0
    """

    timestamp: datetime
    symbol: str
    signal_type: SignalType
    price: float
    strength: float = 1.0  # 信号强度 (0.0 到 1.0)
    reason: Optional[str] = None  # 信号原因
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def __post_init__(self):
        """
        数据验证方法（dataclass自动调用）。

        在Signal对象初始化后自动执行，验证信号数据的有效性。

        目的：
        1. 确保信号数据的完整性和有效性
        2. 防止无效数据进入交易系统
        3. 提供详细的错误信息便于调试

        实现方案：
        1. 验证信号强度是否在0.0到1.0之间
        2. 验证价格是否大于0
        3. 使用ValidationError异常提供详细的错误信息

        使用方法：
        1. 自动调用，无需手动调用
        2. 如果验证失败，抛出ValidationError异常
        """
        # 验证信号强度
        if not 0.0 <= self.strength <= 1.0:
            raise ValidationError(
                "信号强度必须在 0.0 到 1.0 之间",
                field="strength",
                value=self.strength,
                expected="0.0 <= strength <= 1.0",
            )

        # 验证价格
        if self.price <= 0:
            raise ValidationError(
                "价格必须大于 0",
                field="price",
                value=self.price,
                expected="> 0",
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        将Signal对象转换为字典格式。

        目的：
        1. 支持信号数据的序列化，便于存储和传输
        2. 提供与外部系统的数据交换格式
        3. 便于将信号数据转换为DataFrame进行分析

        实现方案：
        1. 将所有字段转换为字典键值对
        2. 将枚举类型的signal_type转换为字符串值
        3. 保持字段名称与类属性一致

        使用方法：
        1. 获取字典表示：signal_dict = signal.to_dict()
        2. 可用于JSON序列化：json.dumps(signal.to_dict())
        3. 可用于创建DataFrame：pd.DataFrame([signal.to_dict()])
        """
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "price": self.price,
            "strength": self.strength,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """
        从字典创建Signal对象。

        目的：
        1. 支持信号数据的反序列化，从存储格式恢复对象
        2. 提供从外部系统接收数据并创建信号对象的能力
        3. 便于从JSON或数据库记录创建信号实例

        实现方案：
        1. 从字典中提取必需字段：timestamp、symbol、signal_type、price
        2. 处理可选字段：strength、reason、metadata
        3. 将字符串类型的signal_type转换回SignalType枚举
        4. 提供默认值处理缺失的可选字段

        使用方法：
        1. 从字典创建信号：signal = Signal.from_dict(data_dict)
        2. 与to_dict方法配合实现序列化/反序列化循环
        3. 处理外部数据源：signal = Signal.from_dict(json.loads(json_str))
        """
        return cls(
            timestamp=data["timestamp"],
            symbol=data["symbol"],
            signal_type=SignalType(data["signal_type"]),
            price=data["price"],
            strength=data.get("strength", 1.0),
            reason=data.get("reason"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PositionSizing:
    """
    仓位大小计算配置类。

    定义仓位管理的参数配置，支持多种仓位计算方法。

    目的：
    1. 标准化仓位管理配置，提供统一的参数接口
    2. 支持多种仓位计算方法，适应不同风险偏好
    3. 提供风险管理参数验证，防止无效配置

    实现方案：
    1. 使用@dataclass装饰器简化配置类定义
    2. 支持固定分数法、固定单位法、百分比风险法、凯利公式等多种方法
    3. 包含风险参数：每笔交易风险、最大总风险、最大单仓位比例
    4. 通过__post_init__方法实现参数自动验证

    使用方法：
    1. 创建配置对象：ps = PositionSizing(method="fixed_fractional", risk_per_trade=0.02)
    2. 作为Strategy构造函数的position_sizing参数传入
    3. 策略自动使用配置计算仓位大小
    """

    method: str = "fixed_fractional"  # 仓位管理方法
    risk_per_trade: float = 0.02  # 每笔交易风险 (2%)
    max_risk: float = 0.1  # 最大总风险 (10%)
    fixed_size: Optional[float] = None  # 固定仓位大小
    max_position_size: float = 0.1  # 最大单仓位比例 (10%)

    def __post_init__(self):
        """
        数据验证方法（dataclass自动调用）。

        在PositionSizing对象初始化后自动执行，验证仓位管理参数的有效性。

        目的：
        1. 确保风险参数在合理范围内，防止过度风险暴露
        2. 验证参数之间的逻辑一致性（如每笔交易风险不大于最大总风险）
        3. 提供详细的错误信息便于调试和配置修复

        实现方案：
        1. 验证每笔交易风险是否在0.0到1.0之间
        2. 验证最大总风险是否在0.0到1.0之间
        3. 验证每笔交易风险不大于最大总风险
        4. 验证最大单仓位比例是否在0.0到1.0之间
        5. 验证固定仓位大小是否大于0（如果提供）

        使用方法：
        1. 自动调用，无需手动调用
        2. 如果验证失败，抛出ValidationError异常
        3. 配置参数时确保符合验证规则
        """
        # 验证风险参数
        if not 0.0 < self.risk_per_trade <= 1.0:
            raise ValidationError(
                "每笔交易风险必须在 0.0 到 1.0 之间",
                field="risk_per_trade",
                value=self.risk_per_trade,
                expected="0.0 < risk_per_trade <= 1.0",
            )

        if not 0.0 < self.max_risk <= 1.0:
            raise ValidationError(
                "最大总风险必须在 0.0 到 1.0 之间",
                field="max_risk",
                value=self.max_risk,
                expected="0.0 < max_risk <= 1.0",
            )

        if self.risk_per_trade > self.max_risk:
            raise ValidationError(
                "每笔交易风险不能大于最大总风险",
                field="risk_per_trade",
                value=self.risk_per_trade,
                expected=f"<= {self.max_risk}",
            )

        # 验证最大仓位比例
        if not 0.0 < self.max_position_size <= 1.0:
            raise ValidationError(
                "最大单仓位比例必须在 0.0 到 1.0 之间",
                field="max_position_size",
                value=self.max_position_size,
                expected="0.0 < max_position_size <= 1.0",
            )

        # 验证固定仓位大小
        if self.fixed_size is not None and self.fixed_size <= 0:
            raise ValidationError(
                "固定仓位大小必须大于 0",
                field="fixed_size",
                value=self.fixed_size,
                expected="> 0",
            )


class Strategy(abc.ABC):
    """
    策略抽象基类。

    所有交易策略的基类，定义策略的标准接口和通用功能。

    目的：
    1. 提供策略开发的标准化框架，确保接口一致性
    2. 封装策略生命周期管理：初始化、信号生成、指标计算
    3. 提供通用的仓位计算、信号管理、状态跟踪功能
    4. 支持策略参数验证和风险管理

    实现方案：
    1. 使用abc.ABC定义抽象基类，强制子类实现关键方法
    2. 定义三个抽象方法：initialize、generate_signals、calculate_indicators
    3. 提供具体方法：仓位计算、策略更新、信号管理、状态获取
    4. 集成PositionSizing进行仓位管理
    5. 使用日志记录和异常处理增强健壮性

    使用方法：
    1. 继承Strategy类：class MyStrategy(Strategy):
    2. 实现抽象方法：initialize、generate_signals、calculate_indicators
    3. 在子类构造函数中调用super().__init__(name, parameters)
    4. 使用update方法更新策略状态并生成信号
    5. 通过get_summary等方法获取策略状态和结果
    """

    def __init__(
        self,
        name: str,
        parameters: Dict[str, Any],
        position_sizing: Optional[PositionSizing] = None,
    ):
        """
        初始化策略对象。

        目的：
        1. 设置策略基本信息：名称、参数、仓位管理配置
        2. 初始化策略状态变量：信号列表、指标字典、元数据
        3. 验证策略参数的有效性
        4. 设置默认的最小数据点要求

        参数：
            name: 策略名称，用于标识和日志记录
            parameters: 策略参数字典，包含策略特定参数
            position_sizing: 仓位管理配置对象，如果为None则使用默认配置

        实现方案：
        1. 存储名称和验证后的参数
        2. 使用提供的position_sizing或创建默认PositionSizing对象
        3. 初始化状态变量：initialized=False, 空信号列表, 空指标字典
        4. 创建元数据字典记录策略创建信息
        5. 设置默认最小数据点min_data_points=20（子类可覆盖）

        使用方法：
        1. 在子类构造函数中调用：super().__init__(name, parameters, position_sizing)
        2. 参数自动通过_validate_parameters方法验证
        3. 可通过self.min_data_points属性调整最小数据点要求
        """
        self.name = name
        self.parameters = self._validate_parameters(parameters)
        self.position_sizing = position_sizing or PositionSizing()

        # 策略状态
        self.initialized = False
        self.signals: List[Signal] = []
        self.indicators: Dict[str, pd.Series] = {}
        self.metadata: Dict[str, Any] = {
            "name": name,
            "parameters": parameters,
            "created_at": datetime.now().isoformat(),
        }

        # Default minimum data points required for strategy
        self.min_data_points = 20  # Default value, can be overridden by subclasses

        logger.info(f"初始化策略: {name}")

    @abc.abstractmethod
    def initialize(self, data: MarketData) -> None:
        """
        抽象方法：初始化策略。

        目的：
        1. 执行策略初始化逻辑，如计算初始指标、设置状态变量
        2. 验证市场数据的完整性和充足性
        3. 准备策略运行所需的内部数据结构
        4. 标记策略为已初始化状态

        参数：
            data: 市场数据对象，包含OHLCV等市场数据

        实现方案：
        1. 子类必须实现此方法，提供具体的初始化逻辑
        2. 通常包括：验证数据点数量、计算初始技术指标、设置初始状态
        3. 初始化完成后应设置self.initialized = True
        4. 如果初始化失败，应抛出StrategyError异常

        使用方法：
        1. 在子类中实现具体的初始化逻辑
        2. 通过self.min_data_points检查数据是否充足
        3. 使用data.get_ohlcv()等方法获取市场数据
        4. 初始化成功后设置self.initialized = True
        """
        pass


    def get_signals_df(self) -> pd.DataFrame:
        """
        获取信号数据框（DataFrame格式）。

        目的：
        1. 将策略生成的信号列表转换为pandas DataFrame，便于分析
        2. 提供信号数据的结构化表示，支持时间序列操作
        3. 便于信号数据的可视化、统计和导出

        返回：
            pd.DataFrame: 信号数据框，包含timestamp、symbol、signal_type、price、strength、reason、metadata等列

        实现方案：
        1. 如果信号列表为空，返回空DataFrame
        2. 将每个Signal对象转换为字典（使用to_dict方法）
        3. 将字典列表转换为DataFrame
        4. 将timestamp列转换为datetime类型并设置为索引

        使用方法：
        1. 获取信号DataFrame：signals_df = strategy.get_signals_df()
        2. 进行信号分析：signals_df['signal_type'].value_counts()
        3. 可视化信号时间分布：signals_df.resample('D').count().plot()
        4. 导出到CSV：signals_df.to_csv('signals.csv')
        """
        if not self.signals:
            return pd.DataFrame()

        records = [signal.to_dict() for signal in self.signals]
        df = pd.DataFrame(records)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        return df

    def get_summary(self) -> Dict[str, Any]:
        """
        获取策略摘要信息。

        目的：
        1. 提供策略状态的全面概览，便于监控和调试
        2. 汇总关键统计信息：信号数量、信号类型分布、时间范围等
        3. 记录策略配置和元数据，便于复现和审计
        4. 为策略性能评估提供基础数据

        返回：
            Dict[str, Any]: 策略摘要字典，包含名称、参数、初始化状态、信号统计、仓位配置、指标列表等

        实现方案：
        1. 获取信号DataFrame用于统计计算
        2. 构建基础摘要：名称、参数、初始化状态、总信号数
        3. 添加仓位管理配置信息
        4. 如果存在信号，计算信号类型分布和时间范围
        5. 添加指标列表信息

        使用方法：
        1. 获取策略摘要：summary = strategy.get_summary()
        2. 打印摘要信息：import json; print(json.dumps(summary, indent=2, ensure_ascii=False))
        3. 监控策略状态：if summary['total_signals'] > 100: ...
        4. 记录策略运行情况：log_strategy_summary(summary)
        """
        signals_df = self.get_signals_df()

        summary = {
            "name": self.name,
            "parameters": self.parameters,
            "initialized": self.initialized,
            "total_signals": len(self.signals),
            "position_sizing": {
                "method": self.position_sizing.method,
                "risk_per_trade": self.position_sizing.risk_per_trade,
                "max_risk": self.position_sizing.max_risk,
                "max_position_size": self.position_sizing.max_position_size,
            },
            "metadata": self.metadata,
        }

        # 添加信号统计
        if not signals_df.empty:
            signal_counts = signals_df["signal_type"].value_counts().to_dict()
            summary["signal_counts"] = signal_counts

            # 计算信号时间范围
            if "timestamp" in signals_df.columns or not signals_df.index.empty:
                timestamps = signals_df.index if isinstance(signals_df.index, pd.DatetimeIndex) else pd.to_datetime(signals_df["timestamp"])
                summary["first_signal"] = timestamps.min().isoformat()
                summary["last_signal"] = timestamps.max().isoformat()

        # 添加指标信息
        summary["indicators"] = list(self.indicators.keys())

        return summary

    def reset(self) -> None:
        """
        重置策略状态。

        目的：
        1. 将策略恢复到初始状态，便于重新开始回测或交易
        2. 清空历史信号和指标数据，释放内存
        3. 保留策略配置和参数，仅重置运行时状态
        4. 记录重置时间，便于审计和调试

        实现方案：
        1. 设置initialized = False，需要重新初始化
        2. 清空signals列表和indicators字典
        3. 更新元数据，记录重置时间和清零信号计数
        4. 记录重置日志信息

        使用方法：
        1. 重置策略状态：strategy.reset()
        2. 重新初始化：strategy.initialize(data)
        3. 适用于回测中测试不同参数或时间区间
        """
        self.initialized = False
        self.signals.clear()
        self.indicators.clear()
        self.metadata.update({
            "reset_at": datetime.now().isoformat(),
            "total_signals": 0,
        })

        logger.info(f"重置策略: {self.name}")

    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证策略参数。

        目的：
        1. 确保策略参数的有效性和安全性，防止无效参数导致策略错误
        2. 根据参数名称自动应用合适的验证规则
        3. 提供详细的错误信息，便于调试和参数调整

        参数：
            parameters: 策略参数字典，包含策略特定参数

        返回：
            Dict[str, Any]: 验证后的参数字典

        实现方案：
        1. 根据参数名称后缀应用不同的验证规则：
           - _period或_window后缀：验证为1-1000的整数
           - _threshold或_level后缀：验证为0-100的数值
           - _weight或_ratio后缀：验证为0-1的数值
        2. 使用Validator工具类进行标准化验证
        3. 捕获验证异常并转换为ValidationError，提供详细上下文

        使用方法：
        1. 自动调用，无需手动调用
        2. 在策略构造函数中自动验证传入参数
        3. 验证失败时抛出ValidationError异常
        """
        validated_params = {}

        for key, value in parameters.items():
            try:
                # 根据参数类型进行验证
                if key.endswith("_period") or key.endswith("_window"):
                    validated_value = Validator.validate_numeric(
                        value, field=key, min_value=1, max_value=1000
                    )
                elif key.endswith("_threshold") or key.endswith("_level"):
                    validated_value = Validator.validate_numeric(
                        value, field=key, min_value=0, max_value=100
                    )
                elif key.endswith("_weight") or key.endswith("_ratio"):
                    validated_value = Validator.validate_numeric(
                        value, field=key, min_value=0, max_value=1
                    )
                else:
                    # 通用验证
                    validated_value = value

                validated_params[key] = validated_value

            except Exception as e:
                raise ValidationError(
                    f"参数验证失败: {key}",
                    field=key,
                    value=value,
                    details={"error": str(e)},
                )

        return validated_params


    def calculate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_date: datetime,
    ) -> float:
        """
        计算交易信号值（兼容性方法）。

        目的：
        1. 提供与事件驱动回测引擎兼容的信号计算接口
        2. 返回标量信号值（正数买入、负数卖出、0无信号），而非Signal对象列表
        3. 支持需要简单信号值的传统回测系统

        参数：
            symbol: 股票代码或交易品种标识符
            data: 历史数据DataFrame，包含OHLCV等列
            current_date: 当前日期，用于确定计算信号的时间点

        返回：
            float: 交易信号值，正数表示买入信号强度，负数表示卖出信号强度，0表示无信号

        实现方案：
        1. 默认返回0.0，表示无信号
        2. 子类应重写此方法以提供实际的信号计算逻辑
        3. 可基于data中的历史数据计算技术指标并生成信号值
        4. 信号值大小可反映信号强度（如1.0强买入，-0.5弱卖出）

        使用方法：
        1. 事件驱动回测引擎调用：signal_value = strategy.calculate_signal(symbol, data, current_date)
        2. 子类重写示例：def calculate_signal(self, symbol, data, current_date): return 1.0 if condition else -1.0
        3. 信号值处理：if signal_value > 0: 执行买入；elif signal_value < 0: 执行卖出
        """
        # 默认实现：使用generate_signals方法并转换为信号值
        # 注意：这需要创建MarketData对象，但为了简化，我们返回0.0
        # 子类应该重写此方法以提供实际的信号计算
        return 0.0


# 导出
__all__ = [
    "SignalType",
    "PositionType",
    "Signal",
    "PositionSizing",
    "Strategy",
]
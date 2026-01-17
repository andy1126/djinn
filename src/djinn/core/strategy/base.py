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

    @abc.abstractmethod
    def generate_signals(self, data: MarketData) -> List[Signal]:
        """
        抽象方法：生成交易信号。

        目的：
        1. 基于当前市场数据和技术指标生成交易信号
        2. 应用策略逻辑判断买入、卖出、持有或退出时机
        3. 计算信号强度，反映信号置信度
        4. 提供交易决策依据，供回测引擎执行

        参数：
            data: 市场数据对象，包含最新的OHLCV数据

        返回：
            List[Signal]: 交易信号列表，每个信号包含时间戳、代码、类型、价格等信息

        实现方案：
        1. 子类必须实现此方法，提供具体的信号生成逻辑
        2. 通常基于技术指标（如移动平均线、RSI、MACD）计算信号
        3. 可以使用self.indicators中预计算的指标值
        4. 应使用_create_signal辅助方法创建Signal对象
        5. 如果信号生成失败，应抛出StrategyError异常

        使用方法：
        1. 在子类中实现具体的信号生成逻辑
        2. 使用data.get_latest()获取最新市场数据
        3. 通过self.indicators访问已计算的指标
        4. 返回一个或多个Signal对象组成的列表
        """
        pass

    @abc.abstractmethod
    def calculate_indicators(self, data: MarketData) -> Dict[str, pd.Series]:
        """
        抽象方法：计算技术指标。

        目的：
        1. 计算策略所需的技术指标，如移动平均线、RSI、MACD等
        2. 为信号生成提供技术分析基础
        3. 将计算结果存储在self.indicators中供其他方法使用
        4. 支持指标数据的后续分析和可视化

        参数：
            data: 市场数据对象，包含历史OHLCV数据

        返回：
            Dict[str, pd.Series]: 技术指标字典，键为指标名称，值为pandas Series时间序列

        实现方案：
        1. 子类必须实现此方法，提供具体的指标计算逻辑
        2. 可以使用TechnicalIndicators工具类或自定义计算函数
        3. 计算结果应为pandas Series，索引与数据时间索引对齐
        4. 应将结果存储在self.indicators字典中
        5. 如果指标计算失败，应抛出StrategyError异常

        使用方法：
        1. 在子类中实现具体的指标计算逻辑
        2. 使用data.get_ohlcv()获取历史数据DataFrame
        3. 计算指标并返回字典，如{'sma_20': sma_series, 'rsi_14': rsi_series}
        4. 指标计算结果会自动通过update方法存储到self.indicators
        """
        pass

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        risk: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> int:
        """
        计算仓位大小（股数）。

        目的：
        1. 根据仓位管理配置计算合适的仓位大小
        2. 应用风险管理规则，控制单笔交易风险
        3. 支持多种仓位计算方法，适应不同交易风格
        4. 确保仓位大小在合理范围内（至少1股，不超过最大仓位限制）

        参数：
            capital: 可用资金总额，用于计算风险金额
            price: 当前资产价格，用于计算可购买股数
            risk: 可选风险金额，如果提供则覆盖配置的风险比例
            stop_loss: 可选止损价格，如果提供则基于止损价计算每股风险

        返回：
            int: 仓位大小（股数），至少1股

        实现方案：
        1. 根据self.position_sizing.method选择计算方法
        2. 支持四种方法：fixed_fractional（固定分数）、fixed_units（固定单位）、percent_risk（百分比风险）、kelly（凯利公式）
        3. 应用最大仓位限制：max_position_size * capital / price
        4. 确保至少购买1股
        5. 提供详细的日志记录和异常处理

        使用方法：
        1. 在策略中调用：position_size = self.calculate_position_size(capital, price)
        2. 可提供止损价优化风险计算：position_size = self.calculate_position_size(capital, price, stop_loss=stop_loss)
        3. 结果可直接用于下单操作
        """
        try:
            method = self.position_sizing.method

            if method == "fixed_fractional":
                # 固定分数法
                if risk is None:
                    risk_amount = capital * self.position_sizing.risk_per_trade
                else:
                    risk_amount = risk

                # 如果提供了止损价，计算每股风险
                if stop_loss is not None:
                    risk_per_share = abs(price - stop_loss)
                    if risk_per_share > 0:
                        position_size = int(risk_amount / risk_per_share)
                    else:
                        position_size = 0
                else:
                    # 没有止损价，使用价格的一定比例作为风险
                    risk_per_share = price * 0.01  # 假设1%的风险
                    position_size = int(risk_amount / risk_per_share)

            elif method == "fixed_units":
                # 固定单位法
                if self.position_sizing.fixed_size is not None:
                    position_size = int(self.position_sizing.fixed_size)
                else:
                    position_size = 100  # 默认100股

            elif method == "percent_risk":
                # 百分比风险法
                if risk is None:
                    risk_percent = self.position_sizing.risk_per_trade
                else:
                    risk_percent = risk / capital

                position_value = capital * risk_percent
                position_size = int(position_value / price)

            elif method == "kelly":
                # 凯利公式（简化版）
                # 需要胜率和赔率信息，这里使用默认值
                win_rate = 0.5  # 默认胜率
                win_loss_ratio = 2.0  # 默认赔率

                kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
                position_value = capital * kelly_fraction * 0.5  # 使用半凯利
                position_size = int(position_value / price)

            else:
                raise StrategyError(
                    f"不支持的仓位管理方法: {method}",
                    strategy_name=self.name,
                    parameters=self.parameters,
                )

            # 应用最大仓位限制
            max_position_value = capital * self.position_sizing.max_position_size
            max_position_size = int(max_position_value / price)

            position_size = min(position_size, max_position_size)

            # 确保至少1股
            position_size = max(1, position_size)

            logger.debug(
                f"仓位计算: 方法={method}, 资金={capital:.2f}, "
                f"价格={price:.2f}, 仓位={position_size}"
            )

            return position_size

        except Exception as e:
            raise StrategyError(
                f"仓位计算失败",
                strategy_name=self.name,
                parameters=self.parameters,
                details={"error": str(e)},
            )

    def update(self, data: MarketData) -> List[Signal]:
        """
        更新策略状态并生成交易信号。

        目的：
        1. 执行完整的策略更新流程：初始化（如果需要）、指标计算、信号生成
        2. 管理策略生命周期状态，确保正确的执行顺序
        3. 记录生成的信号和更新元数据
        4. 提供统一的策略更新入口点

        参数：
            data: 市场数据对象，包含最新的市场数据

        返回：
            List[Signal]: 交易信号列表，可能为空列表（无信号）

        实现方案：
        1. 检查策略是否已初始化，如果未初始化则调用initialize方法
        2. 调用calculate_indicators方法计算技术指标
        3. 调用generate_signals方法生成交易信号
        4. 将信号记录到self.signals列表中
        5. 更新策略元数据（最后更新时间、总信号数）
        6. 返回生成的信号列表

        使用方法：
        1. 在回测或实盘交易中定期调用：signals = strategy.update(data)
        2. 处理返回的信号列表，执行相应的交易操作
        3. 可通过self.get_signals_df()获取历史信号DataFrame
        """
        try:
            # 如果未初始化，先初始化
            if not self.initialized:
                self.initialize(data)
                self.initialized = True

            # 计算指标
            self.indicators = self.calculate_indicators(data)

            # 生成信号
            signals = self.generate_signals(data)

            # 记录信号
            self.signals.extend(signals)

            # 更新元数据
            self.metadata.update({
                "last_update": datetime.now().isoformat(),
                "total_signals": len(self.signals),
            })

            return signals

        except Exception as e:
            raise StrategyError(
                f"策略更新失败",
                strategy_name=self.name,
                parameters=self.parameters,
                details={"error": str(e)},
            )

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

    def get_indicators_df(self) -> pd.DataFrame:
        """
        获取指标数据框（DataFrame格式）。

        目的：
        1. 将策略计算的指标字典转换为pandas DataFrame，便于分析
        2. 提供指标数据的结构化表示，支持时间序列操作
        3. 便于指标数据的可视化、统计和与市场数据对齐分析

        返回：
            pd.DataFrame: 指标数据框，每列为一个技术指标的时间序列

        实现方案：
        1. 如果指标字典为空，返回空DataFrame
        2. 将每个指标Series转换为单独的DataFrame
        3. 使用reduce函数通过outer join合并所有指标DataFrame
        4. 保持时间索引对齐，缺失值用NaN填充

        使用方法：
        1. 获取指标DataFrame：indicators_df = strategy.get_indicators_df()
        2. 与市场数据合并分析：combined_df = pd.concat([price_df, indicators_df], axis=1)
        3. 可视化指标：indicators_df.plot(subplots=True, figsize=(12, 8))
        4. 计算指标统计信息：indicators_df.describe()
        """
        if not self.indicators:
            return pd.DataFrame()

        # 将所有指标合并到一个数据框
        indicators_list = []
        for name, series in self.indicators.items():
            if isinstance(series, pd.Series):
                df_temp = pd.DataFrame({name: series})
                indicators_list.append(df_temp)

        if indicators_list:
            # 合并所有指标
            from functools import reduce
            indicators_df = reduce(
                lambda left, right: pd.merge(
                    left, right, left_index=True, right_index=True, how="outer"
                ),
                indicators_list,
            )
            return indicators_df
        else:
            return pd.DataFrame()

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

    def _calculate_signal_strength(
        self,
        indicator_value: float,
        threshold: float,
        max_value: float,
    ) -> float:
        """
        计算信号强度。

        目的：
        1. 将指标值转换为标准化的信号强度（0.0到1.0）
        2. 反映信号相对于阈值的强弱程度，便于量化信号置信度
        3. 为风险管理提供依据，强信号可分配更大仓位

        参数：
            indicator_value: 指标当前值，如RSI值、价格偏离度等
            threshold: 信号触发阈值，指标超过此值可能产生信号
            max_value: 指标可能的最大值，用于归一化计算

        返回：
            float: 信号强度，范围0.0（无信号）到1.0（最强信号）

        实现方案：
        1. 如果max_value <= threshold，返回0.0（无有效信号）
        2. 计算归一化强度：(indicator_value - threshold) / (max_value - threshold)
        3. 使用max(0.0, min(1.0, strength))确保结果在0.0-1.0范围内

        使用方法：
        1. 在信号生成逻辑中调用：strength = self._calculate_signal_strength(rsi_value, 30, 100)
        2. 将强度值传递给Signal对象：signal = Signal(..., strength=strength)
        3. 强度可用于仓位大小计算或信号过滤
        """
        if max_value <= threshold:
            return 0.0

        strength = (indicator_value - threshold) / (max_value - threshold)
        return max(0.0, min(1.0, strength))

    def _create_signal(
        self,
        timestamp: datetime,
        symbol: str,
        signal_type: SignalType,
        price: float,
        strength: float = 1.0,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """
        创建交易信号对象。

        目的：
        1. 提供创建Signal对象的便捷方法，封装Signal类实例化
        2. 统一信号创建接口，确保信号数据一致性
        3. 简化信号创建代码，提高策略代码可读性

        参数：
            timestamp: 信号时间戳，通常为当前市场时间
            symbol: 股票代码或交易品种标识符
            signal_type: 信号类型枚举值（BUY/SELL/HOLD/EXIT）
            price: 当前价格，用于计算仓位和记录交易价格
            strength: 信号强度（0.0-1.0），默认1.0（最强信号）
            reason: 信号原因描述，便于后续分析和调试
            metadata: 额外元数据字典，可存储策略特定信息

        返回：
            Signal: 交易信号对象

        实现方案：
        1. 直接调用Signal类构造函数
        2. 处理metadata参数的默认值（空字典）
        3. 信号对象会自动通过__post_init__方法验证数据

        使用方法：
        1. 在generate_signals方法中调用：signal = self._create_signal(timestamp, symbol, SignalType.BUY, price)
        2. 可提供信号原因：signal = self._create_signal(..., reason="RSI超卖")
        3. 可添加自定义元数据：signal = self._create_signal(..., metadata={"indicator_value": rsi_value})
        """
        return Signal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            strength=strength,
            reason=reason,
            metadata=metadata or {},
        )

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
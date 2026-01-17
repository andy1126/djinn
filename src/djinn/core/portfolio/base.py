"""
投资组合管理基础类模块。

本模块定义了投资组合管理的抽象基类及相关数据结构。

目的：
    提供投资组合管理的核心抽象接口，支持头寸管理、风险控制和再平衡等功能。
    定义标准化的数据结构，确保投资组合管理的一致性和可扩展性。

实现方案：
    1. 使用枚举类定义投资组合状态和再平衡频率
    2. 通过数据类封装资产、配置、持仓和快照等核心数据结构
    3. Portfolio抽象基类定义投资组合管理的基本接口
    4. RebalancingStrategy抽象基类定义再平衡策略接口

使用方法：
    1. 继承Portfolio类实现具体的投资组合管理逻辑
    2. 使用数据类构建和操作投资组合相关数据
    3. 继承RebalancingStrategy类实现自定义再平衡策略
    4. 通过Portfolio抽象方法实现具体的交易、再平衡和性能跟踪功能
"""

import abc
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from ...utils.exceptions import PortfolioError, ValidationError
from ...utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioStatus(Enum):
    """投资组合状态枚举类。

    目的：
        定义投资组合可能的状态，用于状态管理和监控。

    状态说明：
        ACTIVE: 活跃状态，投资组合正常运作和交易
        CLOSED: 关闭状态，投资组合已停止运作
        SUSPENDED: 暂停状态，投资组合暂时停止交易

    使用方法：
        在Portfolio类中跟踪和更新投资组合状态。
    """
    ACTIVE = "active"
    CLOSED = "closed"
    SUSPENDED = "suspended"


class RebalancingFrequency(Enum):
    """投资组合再平衡频率枚举类。

    目的：
        定义投资组合再平衡的时间频率选项，用于定期再平衡策略。

    频率说明：
        DAILY: 每日再平衡
        WEEKLY: 每周再平衡
        MONTHLY: 每月再平衡
        QUARTERLY: 每季度再平衡
        YEARLY: 每年再平衡
        NEVER: 从不自动再平衡

    使用方法：
        在Portfolio初始化时设置rebalancing_frequency参数。
    """
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NEVER = "never"


@dataclass
class Asset:
    """投资组合中的金融资产数据类。

    目的：
        封装金融资产的基本信息，用于资产管理和分类。

    属性说明：
        symbol: 资产代码（如'AAPL'）
        name: 资产名称（如'Apple Inc.'）
        asset_type: 资产类型（如'stock'、'bond'、'etf'、'cash'等）
        currency: 计价货币，默认'USD'
        exchange: 交易所（可选）
        sector: 行业分类（可选）
        country: 国家/地区（可选）

    使用方法：
        用于构建投资组合的资产目录，支持按类型、行业、国家等进行分类和分析。
    """
    symbol: str
    name: str
    asset_type: str  # 资产类型：'stock'、'bond'、'etf'、'cash'等
    currency: str = 'USD'
    exchange: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None


@dataclass
class PortfolioAllocation:
    """投资组合资产目标配置数据类。

    目的：
        定义单个资产在投资组合中的目标配置参数，用于再平衡和权重管理。

    属性说明：
        symbol: 资产代码
        target_weight: 目标权重（占投资组合的比例，0到1之间）
        min_weight: 最小允许权重（比例），默认0.0
        max_weight: 最大允许权重（比例），默认1.0
        is_core: 是否为核心持仓，默认True

    使用方法：
        用于构建目标配置列表，指导投资组合的再平衡和权重调整。
    """
    symbol: str
    target_weight: float  # 目标权重（占投资组合的比例）
    min_weight: float = 0.0  # 最小允许权重
    max_weight: float = 1.0  # 最大允许权重
    is_core: bool = True  # 是否为核心持仓


@dataclass
class PortfolioHolding:
    """投资组合当前持仓数据类。

    目的：
        封装单个资产的持仓详细信息，用于持仓管理和损益计算。

    属性说明：
        symbol: 资产代码
        quantity: 持仓数量（股数/合约数）
        avg_price: 平均买入价格（美元）
        current_price: 当前市场价格（美元）
        market_value: 当前市值（数量 × 当前价格）
        cost_basis: 成本基础（数量 × 平均价格）
        unrealized_pnl: 未实现损益（市值 - 成本基础）
        realized_pnl: 已实现损益，默认0.0
        entry_date: 首次买入日期（可选）

    使用方法：
        用于跟踪投资组合中每个资产的详细持仓信息，支持损益分析和风险管理。
    """
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_date: Optional[datetime] = None


@dataclass
class PortfolioSnapshot:
    """投资组合状态快照数据类。

    目的：
        封装特定时间点的投资组合完整状态，用于性能跟踪和历史分析。

    属性说明：
        timestamp: 快照时间戳
        total_value: 投资组合总价值（现金 + 持仓市值）
        cash: 现金余额
        holdings: 持仓字典，键为资产代码，值为PortfolioHolding对象
        allocations: 当前权重字典，键为资产代码，值为权重（比例）
        performance: 性能指标字典，包含各种性能指标

    使用方法：
        用于定期记录投资组合状态，构建性能历史，支持回测和报告生成。
    """
    timestamp: datetime
    total_value: float
    cash: float
    holdings: Dict[str, PortfolioHolding]
    allocations: Dict[str, float]  # 当前权重
    performance: Dict[str, float]  # 性能指标


class Portfolio(abc.ABC):
    """投资组合管理抽象基类。

    目的：
        定义投资组合管理的标准接口，支持资产头寸管理、风险控制和再平衡功能。
        提供投资组合状态跟踪、性能分析和风险管理的基础框架。

    主要功能：
        1. 资产价格更新和持仓市值计算
        2. 权重计算和配置偏差分析
        3. 再平衡决策和交易执行
        4. 投资组合快照和性能跟踪
        5. 风险限制检查和合规监控

    实现方案：
        使用抽象方法定义核心接口，具体实现由子类完成。
        内置状态管理、性能计算和风险检查等通用功能。

    使用方法：
        继承此类并实现所有抽象方法，创建具体的投资组合管理类。
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        name: str = "Default Portfolio",
        currency: str = 'USD',
        benchmark_symbol: Optional[str] = None,
        rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY,
        allow_short: bool = False,
        max_leverage: float = 1.0
    ):
        """初始化投资组合。

        目的：
            设置投资组合的基本参数和初始状态。

        参数说明：
            initial_capital: 初始资本（美元），默认100,000
            name: 投资组合名称，默认"Default Portfolio"
            currency: 投资组合货币，默认'USD'
            benchmark_symbol: 基准资产代码，用于性能比较（可选）
            rebalancing_frequency: 再平衡频率，默认每月再平衡
            allow_short: 是否允许空头头寸，默认False
            max_leverage: 最大允许杠杆（1.0表示无杠杆），默认1.0

        实现方案：
            初始化投资组合状态变量，包括持仓、配置、资产目录和性能跟踪数据结构。
            设置风险限制参数，如最大头寸规模、行业暴露和国家暴露限制。
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.name = name
        self.currency = currency
        self.benchmark_symbol = benchmark_symbol
        self.rebalancing_frequency = rebalancing_frequency
        self.allow_short = allow_short
        self.max_leverage = max_leverage

        # 投资组合状态
        self.status = PortfolioStatus.ACTIVE
        self.created_date = datetime.now()
        self.last_rebalanced = None

        # 持仓和配置
        self.holdings: Dict[str, PortfolioHolding] = {}
        self.target_allocations: List[PortfolioAllocation] = []
        self.assets: Dict[str, Asset] = {}

        # 性能跟踪
        self.snapshots: List[PortfolioSnapshot] = []
        self.performance_history = pd.DataFrame()

        # 风险限制
        self.max_position_size = 0.1  # 单个头寸最大10%
        self.max_sector_exposure = 0.3  # 单个行业最大暴露30%
        self.max_country_exposure = 0.5  # 单个国家最大暴露50%

        logger.info(f"Initialized portfolio '{name}' with {initial_capital:,.2f} {currency}")

    @abc.abstractmethod
    def update_prices(self, prices: Dict[str, float]) -> None:
        """更新资产价格。

        目的：
            更新投资组合中资产的最新市场价格，重新计算持仓市值和相关指标。

        参数说明：
            prices: 价格字典，键为资产代码，值为当前价格（美元）

        实现方案：
            更新每个持仓的current_price，重新计算market_value和unrealized_pnl。
            具体实现应由子类完成，需要考虑价格验证和异常处理。

        使用方法：
            定期调用此方法，使用最新的市场价格更新投资组合估值。
        """
        pass

    @abc.abstractmethod
    def calculate_weights(self) -> Dict[str, float]:
        """计算当前投资组合权重。

        目的：
            计算每个资产在当前投资组合中的权重（占总投资组合价值的比例）。

        返回：
            权重字典，键为资产代码，值为当前权重（0到1之间的比例）

        实现方案：
            计算每个持仓的市值占总投资组合价值的比例。
            总投资组合价值 = 现金余额 + 所有持仓市值之和。

        使用方法：
            用于权重监控、再平衡决策和性能分析。
        """
        pass

    @abc.abstractmethod
    def calculate_deviation(self) -> Dict[str, float]:
        """计算当前权重与目标配置的偏差。

        目的：
            计算每个资产的当前权重与目标权重之间的差异，用于再平衡决策。

        返回：
            偏差字典，键为资产代码，值为权重偏差（当前权重 - 目标权重）

        实现方案：
            对每个有目标配置的资产，计算当前权重与目标权重的差值。
            正值表示超配，负值表示低配。

        使用方法：
            用于再平衡触发条件和交易量计算。
        """
        pass

    @abc.abstractmethod
    def needs_rebalancing(self, threshold: float = 0.05) -> bool:
        """检查投资组合是否需要再平衡。

        目的：
            根据权重偏差判断投资组合是否需要进行再平衡调整。

        参数说明：
            threshold: 再平衡阈值（比例），例如0.05表示5%的偏差触发再平衡

        返回：
            如果需要再平衡返回True，否则返回False

        实现方案：
            检查是否有任何资产的权重偏差绝对值超过阈值。
            也可考虑其他因素，如距离上次再平衡的时间、交易成本等。

        使用方法：
            定期调用此方法，决定是否执行再平衡操作。
        """
        pass

    @abc.abstractmethod
    def rebalance(self, target_allocations: List[PortfolioAllocation]) -> List[Dict]:
        """执行投资组合再平衡。

        目的：
            将投资组合调整到目标配置，生成必要的交易订单。

        参数说明：
            target_allocations: 目标配置列表，包含PortfolioAllocation对象

        返回：
            交易订单列表，每个订单为字典，包含symbol、quantity、side等信息

        实现方案：
            计算当前权重与目标权重的差异，确定需要买入或卖出的数量。
            考虑交易成本、最小交易单位、流动性约束等实际因素。

        使用方法：
            在needs_rebalancing返回True时调用此方法，执行再平衡调整。
        """
        pass

    @abc.abstractmethod
    def add_allocation(
        self,
        symbol: str,
        target_weight: float,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        is_core: bool = True
    ) -> None:
        """添加或更新资产的目标配置。

        目的：
            为资产定义目标配置参数，用于后续的再平衡和权重管理。

        参数说明：
            symbol: 资产代码
            target_weight: 目标权重（占投资组合的比例，0到1之间）
            min_weight: 最小允许权重（比例），默认0.0
            max_weight: 最大允许权重（比例），默认1.0
            is_core: 是否为核心持仓，默认True

        实现方案：
            如果资产已有配置，则更新现有配置；否则添加新配置。
            应调用validate_allocation方法验证配置参数的有效性。

        使用方法：
            在投资组合构建阶段设置目标配置，或在策略调整时更新配置。
        """
        pass

    @abc.abstractmethod
    def remove_allocation(self, symbol: str) -> None:
        """移除资产的目标配置。

        目的：
            从目标配置列表中移除指定资产的配置。

        参数说明：
            symbol: 要移除配置的资产代码

        实现方案：
            从target_allocations列表中移除匹配symbol的PortfolioAllocation对象。
            如果资产不存在配置，应妥善处理（忽略或抛出异常）。

        使用方法：
            当资产不再包含在投资组合策略中时，移除其目标配置。
        """
        pass

    def get_target_allocation(self, symbol: str) -> Optional[PortfolioAllocation]:
        """获取资产的目标配置。

        目的：
            根据资产代码查找并返回对应的目标配置对象。

        参数说明：
            symbol: 资产代码

        返回：
            如果找到配置返回PortfolioAllocation对象，否则返回None

        实现方案：
            遍历target_allocations列表，查找symbol匹配的配置对象。

        使用方法：
            在需要查看或验证特定资产配置时调用此方法。
        """
        for allocation in self.target_allocations:
            if allocation.symbol == symbol:
                return allocation
        return None

    @abc.abstractmethod
    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float = 0.0
    ) -> None:
        """执行交易并更新投资组合。

        目的：
            模拟交易执行，更新投资组合的现金余额、持仓数量和成本基础。

        参数说明：
            symbol: 资产代码
            quantity: 交易数量（正数表示买入，负数表示卖出）
            price: 执行价格（每股美元）
            commission: 交易佣金（美元），默认0.0

        实现方案：
            更新现金余额：买入时减少，卖出时增加（考虑佣金）。
            更新持仓：计算新的平均价格、持仓数量和成本基础。
            记录已实现损益（卖出时）。

        使用方法：
            在再平衡或主动交易时调用此方法，模拟交易对投资组合的影响。
        """
        pass

    @abc.abstractmethod
    def take_snapshot(self) -> PortfolioSnapshot:
        """拍摄投资组合状态快照。

        目的：
            捕获当前投资组合的完整状态，用于性能跟踪和历史分析。

        返回：
            PortfolioSnapshot对象，包含时间戳、总价值、现金、持仓和权重等信息

        实现方案：
            收集当前投资组合的所有状态信息，构建PortfolioSnapshot对象。
            应包括计算当前权重和性能指标。

        使用方法：
            定期调用此方法（如每日收盘后），记录投资组合历史状态。
        """
        pass

    def get_total_value(self) -> float:
        """获取投资组合总价值。

        目的：
            计算投资组合的当前总价值，包括现金和所有持仓市值。

        返回：
            投资组合总价值（美元）

        实现方案：
            总价值 = 现金余额 + 所有持仓市值之和

        使用方法：
            用于性能计算、权重计算和风险管理。
        """
        total_value = self.current_capital

        for holding in self.holdings.values():
            total_value += holding.market_value

        return total_value

    def get_cash_balance(self) -> float:
        """获取当前现金余额。

        目的：
            返回投资组合的可用现金余额。

        返回：
            现金余额（美元）

        实现方案：
            直接返回self.current_capital属性值。

        使用方法：
            用于现金管理、交易可行性检查和流动性分析。
        """
        return self.current_capital

    def get_holdings_value(self) -> float:
        """获取所有持仓的总价值。

        目的：
            计算投资组合中所有持仓的市值总和。

        返回：
            持仓总价值（美元）

        实现方案：
            遍历所有持仓，累加每个持仓的market_value。

        使用方法：
            用于杠杆计算、风险暴露分析和绩效归因。
        """
        return sum(h.market_value for h in self.holdings.values())

    def get_leverage(self) -> float:
        """计算当前杠杆率。

        目的：
            衡量投资组合使用杠杆的程度。

        返回：
            杠杆率（持仓总价值 / 投资组合总价值）

        实现方案：
            杠杆率 = 持仓总价值 / 投资组合总价值
            值大于1表示使用杠杆（借入资金），等于1表示无杠杆。

        使用方法：
            用于风险控制和合规检查，确保杠杆率不超过预设限制。
        """
        total_value = self.get_total_value()
        if total_value > 0:
            return self.get_holdings_value() / total_value
        return 0.0

    def check_leverage_limit(self) -> bool:
        """检查杠杆率是否在限制范围内。

        目的：
            验证当前杠杆率是否超过预设的最大杠杆限制。

        返回：
            如果杠杆率不超过限制返回True，否则返回False

        实现方案：
            比较当前杠杆率与self.max_leverage阈值。

        使用方法：
            在交易执行前调用，确保交易不会导致杠杆违规。
        """
        current_leverage = self.get_leverage()
        return current_leverage <= self.max_leverage

    def check_position_limits(self) -> Dict[str, bool]:
        """检查是否有头寸超过规模限制。

        目的：
            验证每个持仓的规模是否超过预设的最大头寸规模限制。

        返回：
            字典，键为资产代码，值为布尔值（True表示合规，False表示违规）

        实现方案：
            对每个持仓计算其占投资组合总价值的比例，与self.max_position_size比较。

        使用方法：
            用于风险监控和合规报告，识别规模过大的头寸。
        """
        violations = {}
        total_value = self.get_total_value()

        for symbol, holding in self.holdings.items():
            position_size = holding.market_value / total_value if total_value > 0 else 0
            violations[symbol] = position_size <= self.max_position_size

        return violations

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取投资组合性能摘要。

        目的：
            计算投资组合的关键性能指标，用于绩效评估和报告。

        返回：
            性能指标字典，包含总回报、年化回报、波动率、当前价值等指标

        实现方案：
            基于历史快照计算总回报和年化回报。
            使用快照价值序列计算波动率（年化）。
            提取当前状态信息如现金余额、持仓数量等。

        使用方法：
            用于定期性能报告、风险调整收益分析和投资决策支持。
        """
        if not self.snapshots:
            return {}

        # 计算基本性能指标
        first_snapshot = self.snapshots[0]
        latest_snapshot = self.snapshots[-1]

        total_return = (latest_snapshot.total_value - first_snapshot.total_value) / first_snapshot.total_value

        # 计算年化回报
        days_held = (latest_snapshot.timestamp - first_snapshot.timestamp).days
        if days_held > 0:
            years = days_held / 365.25
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = total_return

        # 从快照计算波动率
        values = [s.total_value for s in self.snapshots]
        returns = np.diff(values) / values[:-1]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'current_value': latest_snapshot.total_value,
            'cash': latest_snapshot.cash,
            'num_holdings': len(latest_snapshot.holdings),
            'leverage': self.get_leverage(),
            'days_held': days_held
        }

    def validate_allocation(self, allocation: PortfolioAllocation) -> None:
        """验证配置参数的有效性。

        目的：
            确保PortfolioAllocation对象的参数在有效范围内且逻辑一致。

        参数说明：
            allocation: 要验证的PortfolioAllocation对象

        实现方案：
            检查权重参数是否在0到1之间。
            验证最小权重不超过最大权重。
            验证目标权重在最小和最大权重之间。

        使用方法：
            在添加或更新配置时调用，防止无效配置进入系统。
        """
        if allocation.target_weight < 0 or allocation.target_weight > 1:
            raise ValidationError(
                f"Target weight must be between 0 and 1. Got: {allocation.target_weight}"
            )

        if allocation.min_weight < 0 or allocation.min_weight > 1:
            raise ValidationError(
                f"Minimum weight must be between 0 and 1. Got: {allocation.min_weight}"
            )

        if allocation.max_weight < 0 or allocation.max_weight > 1:
            raise ValidationError(
                f"Maximum weight must be between 0 and 1. Got: {allocation.max_weight}"
            )

        if allocation.min_weight > allocation.max_weight:
            raise ValidationError(
                f"Minimum weight ({allocation.min_weight}) cannot exceed "
                f"maximum weight ({allocation.max_weight})"
            )

        if allocation.target_weight < allocation.min_weight:
            raise ValidationError(
                f"Target weight ({allocation.target_weight}) cannot be less than "
                f"minimum weight ({allocation.min_weight})"
            )

        if allocation.target_weight > allocation.max_weight:
            raise ValidationError(
                f"Target weight ({allocation.target_weight}) cannot exceed "
                f"maximum weight ({allocation.max_weight})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """将投资组合转换为字典。

        目的：
            序列化投资组合的基本信息，用于存储、传输或显示。

        返回：
            包含投资组合关键属性的字典

        实现方案：
            提取投资组合的名称、货币、资本、状态、持仓数量等属性。
            将日期时间对象转换为ISO格式字符串。

        使用方法：
            用于数据持久化、API响应或监控面板显示。
        """
        return {
            'name': self.name,
            'currency': self.currency,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_value': self.get_total_value(),
            'status': self.status.value,
            'created_date': self.created_date.isoformat(),
            'last_rebalanced': self.last_rebalanced.isoformat() if self.last_rebalanced else None,
            'num_holdings': len(self.holdings),
            'num_allocations': len(self.target_allocations),
            'leverage': self.get_leverage(),
            'rebalancing_frequency': self.rebalancing_frequency.value,
            'allow_short': self.allow_short,
            'max_leverage': self.max_leverage
        }


class RebalancingStrategy(abc.ABC):
    """投资组合再平衡策略抽象基类。

    目的：
        定义再平衡策略的标准接口，支持不同的再平衡算法和优化方法。

    主要功能：
        1. 计算再平衡所需的交易订单
        2. 优化投资组合权重分配
        3. 考虑交易约束和成本

    实现方案：
        使用抽象方法定义核心接口，具体实现由子类完成。
        支持从简单阈值再平衡到复杂优化算法的各种策略。

    使用方法：
        继承此类并实现抽象方法，创建具体的再平衡策略类。
        与Portfolio类配合使用，实现自动再平衡功能。
    """

    @abc.abstractmethod
    def calculate_rebalancing_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """计算再平衡所需的交易订单。

        目的：
            根据当前权重与目标权重的差异，计算实现再平衡所需的交易列表。

        参数说明：
            current_weights: 当前投资组合权重字典
            target_weights: 目标投资组合权重字典
            portfolio_value: 投资组合总价值（美元）
            prices: 当前资产价格字典
            constraints: 交易约束字典（佣金、最小交易规模等）

        返回：
            交易字典列表，每个字典包含symbol、quantity、side等键

        实现方案：
            计算每个资产的目标市值与当前市值的差异。
            考虑交易约束（如最小交易单位、整数股要求）确定实际交易数量。
            优化交易顺序以减少交易成本或市场影响。

        使用方法：
            在Portfolio.rebalance方法中调用，生成具体的交易指令。
        """
        pass

    @abc.abstractmethod
    def optimize_weights(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """优化投资组合权重。

        目的：
            基于现代投资组合理论或其他优化方法，计算最优的资产权重分配。

        参数说明：
            expected_returns: 每个资产的预期收益字典
            covariance_matrix: 收益协方差矩阵DataFrame
            constraints: 优化约束字典（如权重上下限、行业限制等）

        返回：
            优化后的权重字典，键为资产代码，值为优化权重（比例）

        实现方案：
            可使用均值-方差优化、风险平价、最大夏普比率等方法。
            考虑实际交易约束，如最小权重、整数股限制等。

        使用方法：
            用于动态资产配置和策略优化，生成目标配置供再平衡使用。
        """
        pass
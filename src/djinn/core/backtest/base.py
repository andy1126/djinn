"""
模块目的：定义回测引擎的抽象基类和结果容器，提供回测的核心接口和数据结构。

实现方案：
1. BacktestEngine 抽象基类：定义了所有回测引擎必须实现的接口，包括运行回测、数据处理、交易执行和指标计算等方法。
2. BacktestResult 数据类：用于存储回测结果，包含性能指标、交易统计、持仓数据和时间序列等完整信息。
3. Trade 和 Position 数据类：表示单笔交易和持仓状态，支持佣金和滑点计算。
4. 数据验证和费用计算：提供通用的数据验证方法和交易成本计算功能。

使用方法：
1. 继承 BacktestEngine 类实现具体的回测引擎（如事件驱动或向量化引擎）。
2. 使用 BacktestResult 类存储和分析回测结果。
3. 通过 validate_data 方法验证输入数据的完整性。
4. 使用 calculate_commission 和 calculate_slippage 方法计算交易成本。

示例：
    # 创建自定义回测引擎
    class MyBacktestEngine(BacktestEngine):
        def run(self, strategy, data, start_date, end_date):
            # 实现具体的回测逻辑
            pass

    # 使用回测结果
    result = BacktestResult(...)
    summary = result.to_dataframe()
"""

import abc
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from ...utils.exceptions import BacktestError, ValidationError
from ...utils.logger import get_logger
from ..strategy.base import Strategy

logger = get_logger(__name__)


class BacktestMode(Enum):
    """
    回测执行模式枚举类。

    目的：定义不同的回测执行模式，支持多种回测引擎的实现。

    实现方案：
    1. EVENT_DRIVEN: 事件驱动模式，逐笔处理数据，模拟真实交易环境
    2. VECTORIZED: 向量化模式，使用数组运算批量处理数据，性能更高
    3. HYBRID: 混合模式，结合事件驱动和向量化的优点

    使用方法：
        from djinn.core.backtest import BacktestMode

        # 选择回测模式
        mode = BacktestMode.EVENT_DRIVEN
        mode = BacktestMode.VECTORIZED
    """
    EVENT_DRIVEN = "event_driven"
    VECTORIZED = "vectorized"
    HYBRID = "hybrid"


@dataclass
class Trade:
    """
    单笔交易执行记录数据类。

    目的：表示回测中的一笔完整交易记录，包含交易详情和成本信息。

    实现方案：
    1. 使用dataclass自动生成构造函数和常用方法
    2. 包含交易时间、品种、数量、价格、方向等基本信息
    3. 支持佣金和滑点成本记录
    4. 提供交易ID和策略名称用于追踪

    字段说明：
    - timestamp: 交易时间
    - symbol: 交易品种
    - quantity: 交易数量（正数表示买入，负数表示卖出）
    - price: 交易价格
    - side: 交易方向 ('buy' 或 'sell')
    - commission: 佣金费用
    - slippage: 滑点成本
    - trade_id: 交易唯一标识符
    - strategy_name: 生成交易的策略名称

    使用方法：
        from djinn.core.backtest import Trade
        from datetime import datetime

        # 创建交易记录
        trade = Trade(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            quantity=100,
            price=150.0,
            side="buy",
            commission=15.0,
            slippage=7.5
        )
    """
    timestamp: datetime
    symbol: str
    quantity: float
    price: float
    side: str  # 'buy' or 'sell'
    commission: float = 0.0
    slippage: float = 0.0
    trade_id: Optional[str] = None
    strategy_name: Optional[str] = None


@dataclass
class Position:
    """
    持仓状态数据类。

    目的：表示在特定时间点的持仓状态，包含持仓详情和盈亏信息。

    实现方案：
    1. 使用dataclass自动生成构造函数和常用方法
    2. 包含持仓时间、品种、数量、成本价等基本信息
    3. 支持未实现盈亏和已实现盈亏的计算和记录
    4. 提供市值计算用于组合估值

    字段说明：
    - timestamp: 持仓时间点
    - symbol: 持仓品种
    - quantity: 持仓数量（正数表示多头，负数表示空头）
    - avg_price: 平均持仓成本价
    - market_value: 当前市值（持仓数量 × 当前价格）
    - unrealized_pnl: 未实现盈亏（持仓浮盈浮亏）
    - realized_pnl: 已实现盈亏（平仓后实现的盈亏）

    使用方法：
        from djinn.core.backtest import Position
        from datetime import datetime

        # 创建持仓记录
        position = Position(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            realized_pnl=0.0
        )
    """
    timestamp: datetime
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """
    回测结果容器数据类。

    目的：存储和管理回测的完整结果，包含性能指标、交易统计、持仓数据和时序信息。

    实现方案：
    1. 使用dataclass自动生成构造函数和常用方法
    2. 整合四大类数据：性能指标、交易统计、组合统计、时序数据
    3. 提供数据转换方法（DataFrame、字典格式）
    4. 支持结果分析和可视化

    主要字段类别：
    1. 性能指标：总收益率、年化收益率、夏普比率、最大回撤等
    2. 交易统计：交易总数、胜率、盈亏比、平均交易收益率等
    3. 组合统计：初始资金、最终资金、峰值资金、总佣金等
    4. 时序数据：净值曲线、收益率序列、回撤序列、持仓列表、交易列表
    5. 元数据：策略名称、交易品种、日期范围、参数配置

    使用方法：
        from djinn.core.backtest import BacktestResult
        import pandas as pd

        # 创建回测结果
        result = BacktestResult(
            total_return=0.25,
            annual_return=0.12,
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            # ... 其他参数
            equity_curve=pd.Series(...),
            returns=pd.Series(...),
            trades=[...],
            positions=[...]
        )

        # 转换为DataFrame分析
        df = result.to_dataframe()

        # 转换为字典格式
        data = result.to_dict()
    """
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    avg_trade_duration: pd.Timedelta
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Portfolio statistics
    initial_capital: float
    final_capital: float
    peak_capital: float
    trough_capital: float
    total_commission: float
    total_slippage: float

    # Time series data
    equity_curve: pd.Series
    returns: pd.Series
    drawdown: pd.Series
    positions: List[Position]
    trades: List[Trade]

    # Additional metadata
    strategy_name: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    frequency: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """
        将回测结果转换为DataFrame格式。

        目的：将回测结果的结构化数据转换为便于分析和可视化的DataFrame格式。

        实现方案：
        1. 提取回测结果中的关键指标和统计数据
        2. 组织为两列DataFrame：指标名称和指标值
        3. 包含性能指标、交易统计和组合统计的主要信息

        返回值：
            pd.DataFrame: 包含回测摘要的DataFrame，包含'Metric'和'Value'两列

        使用方法：
            result = BacktestResult(...)
            df = result.to_dataframe()
            print(df)

        输出示例：
                           Metric      Value
            0         Total Return     25.00%
            1        Annual Return     12.00%
            2          Sharpe Ratio     1.50
            3         Max Drawdown     15.00%
            ...                ...        ...
        """
        # Create summary DataFrame
        summary_data = {
            'Metric': [
                'Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown',
                'Volatility', 'Sortino Ratio', 'Calmar Ratio', 'Win Rate',
                'Profit Factor', 'Avg Win', 'Avg Loss', 'Total Trades',
                'Winning Trades', 'Losing Trades', 'Avg Trade Return',
                'Initial Capital', 'Final Capital', 'Peak Capital',
                'Trough Capital', 'Total Commission', 'Total Slippage'
            ],
            'Value': [
                self.total_return, self.annual_return, self.sharpe_ratio,
                self.max_drawdown, self.volatility, self.sortino_ratio,
                self.calmar_ratio, self.win_rate, self.profit_factor,
                self.avg_win, self.avg_loss, self.total_trades,
                self.winning_trades, self.losing_trades, self.avg_trade_return,
                self.initial_capital, self.final_capital, self.peak_capital,
                self.trough_capital, self.total_commission, self.total_slippage
            ]
        }

        return pd.DataFrame(summary_data)

    def to_dict(self) -> Dict[str, Any]:
        """
        将回测结果转换为字典格式。

        目的：将回测结果转换为嵌套字典结构，便于JSON序列化和API传输。

        实现方案：
        1. 将回测结果组织为分层字典结构
        2. 分为四个主要类别：performance（性能指标）、trades（交易统计）、portfolio（组合统计）、metadata（元数据）
        3. 自动转换时间格式为ISO字符串，便于序列化

        返回值：
            Dict[str, Any]: 嵌套字典结构的回测结果

        使用方法：
            result = BacktestResult(...)
            data = result.to_dict()

            # 转换为JSON
            import json
            json_data = json.dumps(data)

        字典结构：
            {
                'performance': {
                    'total_return': 0.25,
                    'annual_return': 0.12,
                    'sharpe_ratio': 1.5,
                    ...
                },
                'trades': {
                    'total_trades': 100,
                    'winning_trades': 60,
                    ...
                },
                'portfolio': {
                    'initial_capital': 100000.0,
                    'final_capital': 125000.0,
                    ...
                },
                'metadata': {
                    'strategy_name': 'MyStrategy',
                    'symbols': ['AAPL', 'GOOGL'],
                    ...
                }
            }
        """
        return {
            'performance': {
                'total_return': self.total_return,
                'annual_return': self.annual_return,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'volatility': self.volatility,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss
            },
            'trades': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'avg_trade_return': self.avg_trade_return,
                'avg_trade_duration': str(self.avg_trade_duration),
                'max_consecutive_wins': self.max_consecutive_wins,
                'max_consecutive_losses': self.max_consecutive_losses
            },
            'portfolio': {
                'initial_capital': self.initial_capital,
                'final_capital': self.final_capital,
                'peak_capital': self.peak_capital,
                'trough_capital': self.trough_capital,
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage
            },
            'metadata': {
                'strategy_name': self.strategy_name,
                'symbols': self.symbols,
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'frequency': self.frequency,
                'parameters': self.parameters
            }
        }


class BacktestEngine(abc.ABC):
    """
    回测引擎抽象基类。

    目的：定义所有回测引擎必须实现的统一接口，提供回测的核心框架和通用功能。

    实现方案：
    1. 使用abc.ABC定义抽象基类，强制子类实现核心方法
    2. 提供通用的初始化、数据验证、费用计算等功能
    3. 管理回测状态：资金、持仓、交易记录、净值曲线等
    4. 定义标准的回测执行流程和结果计算接口

    核心抽象方法（子类必须实现）：
    1. run: 执行完整的回测流程
    2. _initialize_backtest: 初始化回测状态
    3. _process_bar: 处理单根K线数据
    4. _execute_trades: 执行交易信号
    5. _update_positions: 更新持仓状态
    6. _calculate_metrics: 计算性能指标

    核心具体方法：
    1. validate_data: 验证输入数据的完整性和有效性
    2. calculate_commission: 计算交易佣金
    3. calculate_slippage: 计算交易滑点
    4. get_summary: 获取回测状态摘要

    使用方法：
        from djinn.core.backtest import BacktestEngine
        import abc

        # 创建自定义回测引擎
        class MyBacktestEngine(BacktestEngine):
            def run(self, strategy, data, start_date, end_date):
                # 实现具体回测逻辑
                pass

            # 实现其他抽象方法...
    """

    def __init__(
            self,
            initial_capital: float = 100000.0,
            commission: float = 0.001,  # 0.1%
            slippage: float = 0.0005,  # 0.05%
            benchmark_symbol: Optional[str] = None,
            risk_free_rate: float = 0.02,  # 2% annual
            mode: BacktestMode = BacktestMode.EVENT_DRIVEN
    ):
        """
        初始化回测引擎。

        目的：设置回测引擎的基本参数和初始状态，为回测执行做准备。

        实现方案：
        1. 设置回测核心参数：初始资金、佣金率、滑点率、无风险利率等
        2. 初始化回测状态变量：当前资金、当前日期、运行状态
        3. 初始化数据存储：交易记录列表、持仓列表、净值曲线、收益率序列
        4. 配置回测执行模式：事件驱动、向量化或混合模式

        参数说明：
            initial_capital: 初始资金，回测开始时的资金量（默认100000.0）
            commission: 佣金率，每笔交易价值的百分比（默认0.001，即0.1%）
            slippage: 滑点率，每笔交易价值的百分比（默认0.0005，即0.05%）
            benchmark_symbol: 基准品种符号，用于对比分析（可选）
            risk_free_rate: 无风险利率，用于计算风险调整收益（默认0.02，即2%年化）
            mode: 回测执行模式，BacktestMode枚举值（默认EVENT_DRIVEN）

        使用方法：
            from djinn.core.backtest import BacktestEngine, BacktestMode

            # 创建回测引擎实例
            engine = BacktestEngine(
                initial_capital=100000,
                commission=0.001,
                slippage=0.0005,
                benchmark_symbol='SPY',
                risk_free_rate=0.02,
                mode=BacktestMode.EVENT_DRIVEN
            )
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        self.mode = mode

        # State variables
        self.current_capital = initial_capital
        self.current_date = None
        self.is_running = False

        # Data storage
        self.trades: List[Trade] = []
        self.positions: List[Position] = []
        self.equity_curve = pd.Series(dtype=float)
        self.returns = pd.Series(dtype=float)

        logger.info(f"Initialized backtest engine with mode: {mode.value}")

    @abc.abstractmethod
    def run(
            self,
            strategy: Strategy,
            data: Dict[str, pd.DataFrame],
            start_date: datetime,
            end_date: datetime,
            **kwargs
    ) -> BacktestResult:
        """
        执行回测。

        目的：运行完整的回测流程，从策略初始化到结果计算，返回包含所有性能指标的回测结果。

        实现方案：
        1. 验证输入数据的完整性和有效性
        2. 初始化策略和回测状态
        3. 按时间顺序处理数据，生成交易信号，执行交易，更新持仓
        4. 计算净值曲线、收益率序列和各项性能指标
        5. 汇总所有结果并返回BacktestResult对象

        参数说明：
            strategy: 交易策略实例，必须实现Strategy接口
            data: 数据字典，映射品种符号到OHLCV数据的DataFrame
            start_date: 回测开始日期
            end_date: 回测结束日期
            **kwargs: 额外参数，供具体引擎实现使用

        返回值：
            BacktestResult: 包含完整回测结果的对象，包括性能指标、交易统计、时序数据等

        使用方法：
            from djinn.core.backtest import BacktestEngine
            from djinn.core.strategy import MyStrategy
            import pandas as pd

            # 创建回测引擎和策略
            engine = BacktestEngine()
            strategy = MyStrategy()

            # 准备数据
            data = {
                'AAPL': pd.DataFrame(...),  # OHLCV数据
                'GOOGL': pd.DataFrame(...)
            }

            # 运行回测
            result = engine.run(
                strategy=strategy,
                data=data,
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2023, 1, 1)
            )

            # 分析结果
            print(f"总收益率: {result.total_return:.2%}")
            print(f"夏普比率: {result.sharpe_ratio:.2f}")
        """
        pass

    @abc.abstractmethod
    def _initialize_backtest(
            self,
            strategy: Strategy,
            data: Dict[str, pd.DataFrame],
            start_date: datetime,
            end_date: datetime,
            frequency: str
    ) -> None:
        """
        初始化回测状态。

        目的：在回测开始前初始化所有必要的状态变量和数据结构，为回测执行做准备。

        实现方案：
        1. 初始化策略：调用策略的initialize方法，传递必要的参数
        2. 重置回测状态：重置资金、持仓、交易记录等状态变量
        3. 准备数据结构：初始化净值曲线、收益率序列等时间序列数据
        4. 验证数据完整性：确保输入数据满足回测要求

        参数说明：
            strategy: 交易策略实例，需要初始化的策略对象
            data: 数据字典，包含所有品种的OHLCV数据
            start_date: 回测开始日期
            end_date: 回测结束日期

        注意事项：
        1. 此方法在run方法开始时自动调用
        2. 子类应在此方法中完成引擎特定的初始化工作
        3. 需要确保所有状态变量都处于正确的初始状态
        """
        pass

    @abc.abstractmethod
    def _process_bar(self, current_date: datetime) -> None:
        """
        处理单根K线数据。

        目的：在事件驱动回测中，按时间顺序逐根处理K线数据，模拟实时交易过程。

        实现方案：
        1. 获取当前时间点的市场数据（价格、成交量等）
        2. 调用策略生成交易信号
        3. 执行交易信号，生成交易记录
        4. 更新持仓状态和组合价值
        5. 记录净值曲线和收益率

        参数说明：
            current_date: 当前处理的K线时间点

        注意事项：
        1. 此方法在事件驱动回测中按时间顺序逐根调用
        2. 向量化回测引擎可能不需要实现此方法，或使用不同的处理方式
        3. 需要处理异常情况，如数据缺失、信号错误等

        典型流程：
            1. 更新当前价格
            2. 检查止损止盈条件
            3. 获取策略信号
            4. 执行交易
            5. 更新持仓
            6. 计算组合价值
        """
        pass

    @abc.abstractmethod
    def _execute_trades(self, signals: Dict[str, float]) -> List[Trade]:
        """
        执行交易信号。

        目的：根据策略生成的交易信号，执行具体的交易操作，生成交易记录。

        实现方案：
        1. 解析交易信号：确定交易方向（买入/卖出）、品种和信号强度
        2. 计算交易数量：根据资金管理规则确定交易数量
        3. 计算交易价格：考虑滑点影响后的实际成交价格
        4. 计算交易成本：佣金和滑点成本计算
        5. 创建交易记录：生成Trade对象记录交易详情
        6. 更新资金状态：扣除或增加资金，反映交易影响

        参数说明：
            signals: 交易信号字典，映射品种符号到信号强度（正值表示买入，负值表示卖出，零表示无信号）

        返回值：
            List[Trade]: 执行的所有交易记录列表

        注意事项：
        1. 需要检查资金是否充足（对于买入交易）
        2. 需要考虑仓位限制和风险控制规则
        3. 需要正确处理做空交易（如果允许）
        4. 需要考虑最小交易单位和交易限制

        交易执行流程：
            1. 遍历所有信号
            2. 计算目标持仓和当前持仓差异
            3. 确定交易数量和方向
            4. 计算成交价格和交易成本
            5. 创建交易记录
            6. 更新资金和持仓状态
        """
        pass

    @abc.abstractmethod
    def _update_positions(self, trades: List[Trade]) -> None:
        """
        更新持仓状态。

        目的：根据已执行的交易记录更新持仓状态，包括持仓数量、成本价、市值和盈亏计算。

        实现方案：
        1. 遍历所有交易记录，按品种分组处理
        2. 更新持仓数量：买入增加持仓，卖出减少持仓
        3. 重新计算平均成本价：加权平均计算新的持仓成本
        4. 计算已实现盈亏：平仓交易产生的实际盈亏
        5. 更新持仓市值和未实现盈亏：基于当前市场价格计算
        6. 管理持仓列表：新增持仓、更新持仓或移除零持仓

        参数说明：
            trades: 已执行的交易记录列表，包含Trade对象

        注意事项：
        1. 需要正确处理做多和做空持仓（正数表示多头，负数表示空头）
        2. 需要精确计算平均成本价，特别是部分平仓的情况
        3. 需要区分已实现盈亏和未实现盈亏
        4. 需要处理持仓归零的情况，从持仓列表中移除

        持仓更新规则：
            1. 开新仓：创建新持仓记录，设置成本价为成交价
            2. 加仓：重新计算加权平均成本价，增加持仓数量
            3. 减仓：更新持仓数量，计算部分已实现盈亏
            4. 平仓：计算全部已实现盈亏，从持仓列表移除
        """
        pass

    @abc.abstractmethod
    def _calculate_metrics(self) -> BacktestResult:
        """
        计算回测性能指标。

        目的：基于回测过程中收集的数据，计算全面的性能指标和统计信息。

        实现方案：
        1. 收集回测数据：净值曲线、交易记录、持仓历史等
        2. 计算基础指标：总收益率、年化收益率、波动率等
        3. 计算风险调整指标：夏普比率、索提诺比率、卡玛比率等
        4. 计算交易统计：胜率、盈亏比、平均交易收益率等
        5. 计算回撤分析：最大回撤、平均回撤、回撤持续时间等
        6. 汇总组合统计：资金使用情况、佣金成本、滑点成本等
        7. 构建回测结果对象：创建BacktestResult实例返回

        返回值：
            BacktestResult: 包含所有计算指标的回测结果对象

        注意事项：
        1. 需要处理空数据情况（如无交易、数据不完整）
        2. 需要确保指标计算的准确性和一致性
        3. 需要考虑不同的年化因子（交易日天数）
        4. 需要正确对齐时间序列数据

        指标计算流程：
            1. 从净值曲线计算收益率序列
            2. 计算收益相关指标（总收益、年化收益等）
            3. 计算风险相关指标（波动率、夏普比率等）
            4. 分析交易记录（胜率、盈亏比等）
            5. 计算回撤分析（最大回撤、回撤持续时间等）
            6. 汇总所有指标到BacktestResult对象
        """
        pass

    def validate_data(
            self,
            data: Dict[str, pd.DataFrame],
            required_columns: List[str] = None
    ) -> None:
        """
        验证回测输入数据。

        目的：确保输入数据的完整性和有效性，防止因数据问题导致回测错误。

        实现方案：
        1. 检查数据非空：确保提供了数据且不为空
        2. 检查数据框完整性：确保每个品种的数据框不为空
        3. 验证必需列：检查OHLCV等必需列是否存在
        4. 检查缺失值：检测并警告数据中的NaN值
        5. 验证时间索引：确保索引为DatetimeIndex类型
        6. 检查重复日期：确保没有重复的时间戳

        参数说明：
            data: 数据字典，映射品种符号到包含OHLCV数据的DataFrame
            required_columns: 必需列名列表，默认为['open', 'high', 'low', 'close', 'volume']

        异常抛出：
            ValidationError: 当数据验证失败时抛出，包含具体的错误信息

        使用方法：
            engine = BacktestEngine()
            data = {
                'AAPL': df_aapl,  # DataFrame with OHLCV data
                'GOOGL': df_googl
            }

            try:
                engine.validate_data(data)
                print("数据验证通过")
            except ValidationError as e:
                print(f"数据验证失败: {e}")

        验证规则：
            1. 数据字典不能为空
            2. 每个DataFrame不能为空
            3. 必须包含指定的列（默认OHLCV）
            4. 索引必须是DatetimeIndex
            5. 不能有重复的日期
        """
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not data:
            raise ValidationError("No data provided for backtest")

        for symbol, df in data.items():
            if df.empty:
                raise ValidationError(f"Empty DataFrame for symbol: {symbol}")

            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValidationError(
                    f"Missing columns for symbol {symbol}: {missing_columns}"
                )

            # Check for NaN values
            nan_count = df[required_columns].isna().sum().sum()
            if nan_count > 0:
                logger.warning(
                    f"Symbol {symbol} has {nan_count} NaN values in required columns"
                )

            # Check date index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValidationError(
                    f"DataFrame index must be DatetimeIndex for symbol: {symbol}"
                )

            # Check for duplicate dates
            if df.index.duplicated().any():
                raise ValidationError(
                    f"Duplicate dates found in data for symbol: {symbol}"
                )

        logger.info(f"Data validation passed for {len(data)} symbols")

    def calculate_commission(
            self,
            quantity: float,
            price: float,
            side: str
    ) -> float:
        """
        计算交易佣金。

        目的：根据交易详情计算佣金费用，支持固定比例和最低佣金两种模式。

        实现方案：
        1. 计算交易价值：交易数量 × 交易价格（取绝对值处理做空情况）
        2. 按比例计算佣金：交易价值 × 佣金率
        3. 应用最低佣金限制：如果计算佣金低于最低佣金，使用最低佣金
        4. 返回佣金金额（正数，表示交易成本）

        参数说明：
            quantity: 交易数量，正数表示买入，负数表示卖出
            price: 交易价格，单位价格
            side: 交易方向，'buy'表示买入，'sell'表示卖出

        返回值：
            float: 佣金金额，始终为非负数

        计算方法：
            佣金 = max(交易价值 × 佣金率, 最低佣金)
            其中：交易价值 = abs(数量) × 价格

        示例：
            # 假设佣金率0.1%，最低佣金$1
            commission = engine.calculate_commission(
                quantity=100,      # 买入100股
                price=150.0,       # 每股$150
                side='buy'         # 买入方向
            )
            # 交易价值 = 100 × 150 = $15,000
            # 比例佣金 = 15,000 × 0.001 = $15
            # 最终佣金 = max($15, $1) = $15
        """
        trade_value = abs(quantity) * price
        commission = trade_value * self.commission

        # Minimum commission check (example: $1 minimum)
        min_commission = 1.0
        if commission < min_commission:
            commission = min_commission

        return commission

    def calculate_slippage(
            self,
            quantity: float,
            price: float,
            side: str
    ) -> float:
        """
        计算交易滑点成本。

        目的：模拟交易执行中的价格滑点，反映市场冲击成本。

        实现方案：
        1. 计算交易价值：交易数量 × 交易价格（取绝对值处理做空情况）
        2. 按比例计算滑点：交易价值 × 滑点率
        3. 滑点始终为成本：对于买方，价格上升；对于卖方，价格下降
        4. 返回滑点金额（正数，表示交易成本）

        参数说明：
            quantity: 交易数量，正数表示买入，负数表示卖出
            price: 交易价格，单位价格
            side: 交易方向，'buy'表示买入，'sell'表示卖出

        返回值：
            float: 滑点金额，始终为非负数（表示成本）

        计算方法：
            滑点 = 交易价值 × 滑点率
            其中：交易价值 = abs(数量) × 价格

        滑点影响：
            1. 买入交易：实际成交价格 = 报价 + 滑点/数量
            2. 卖出交易：实际成交价格 = 报价 - 滑点/数量
            3. 滑点始终增加交易成本，降低交易收益

        示例：
            # 假设滑点率0.05%
            slippage = engine.calculate_slippage(
                quantity=100,      # 买入100股
                price=150.0,       # 每股$150
                side='buy'         # 买入方向
            )
            # 交易价值 = 100 × 150 = $15,000
            # 滑点 = 15,000 × 0.0005 = $7.5
            # 实际成交价格 = 150 + 7.5/100 = $150.075
        """
        trade_value = abs(quantity) * price
        slippage = trade_value * self.slippage

        # Slippage is always a cost for the trader
        # For buys: price increases, for sells: price decreases
        return slippage

    def get_summary(self) -> Dict[str, Any]:
        """
        获取当前回测状态摘要。

        目的：提供回测引擎当前状态的快速概览，用于监控和调试。

        实现方案：
        1. 收集关键状态信息：资金、交易、持仓、运行状态等
        2. 计算累计统计：总交易数、总佣金、总滑点等
        3. 返回结构化字典，便于查看和记录

        返回值：
            Dict[str, Any]: 包含回测状态摘要的字典

        字典字段说明：
            initial_capital: 初始资金
            current_capital: 当前资金（现金部分）
            total_trades: 总交易次数
            total_commission: 累计佣金成本
            total_slippage: 累计滑点成本
            current_positions: 当前持仓数量
            mode: 回测执行模式
            is_running: 回测是否正在运行

        使用方法：
            engine = BacktestEngine()
            # ... 运行部分回测后
            summary = engine.get_summary()
            print(f"当前资金: {summary['current_capital']:.2f}")
            print(f"总交易数: {summary['total_trades']}")
            print(f"累计佣金: {summary['total_commission']:.2f}")

        注意事项：
            1. 此方法可在回测过程中任意时间调用
            2. 返回的是当前快照，状态可能随时间变化
            3. 仅包含摘要信息，详细数据需通过其他方法获取
        """
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_trades': len(self.trades),
            'total_commission': sum(t.commission for t in self.trades),
            'total_slippage': sum(t.slippage for t in self.trades),
            'current_positions': len(self.positions),
            'mode': self.mode.value,
            'is_running': self.is_running
        }

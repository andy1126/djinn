"""
投资组合风险管理模块。

本模块提供风险管理功能，包括头寸规模计算、风险限制设置和风险指标计算。

目的：
    管理投资组合风险，控制头寸规模，监控风险指标，确保投资组合在预设风险边界内运行。

实现方案：
    1. 使用枚举类定义风险指标和头寸规模计算方法
    2. 通过数据类封装风险限制和头寸规模结果
    3. RiskManager类提供完整的风险管理功能
    4. PositionSizer类专注于头寸规模计算

使用方法：
    1. 初始化RiskManager设置初始资本和风险参数
    2. 添加RiskLimit定义风险限制
    3. 使用calculate_position_risk和calculate_portfolio_risk计算风险指标
    4. 使用check_risk_limits检查风险限制违规
    5. 使用calculate_position_size计算最优头寸规模
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from ...utils.exceptions import RiskError, ValidationError
from ...utils.logger import get_logger

logger = get_logger(__name__)


class RiskMetric(Enum):
    """风险指标枚举类。

    目的：
        定义可计算的风险指标类型，用于风险限制和监控。

    指标说明：
        VAR: 风险价值（Value at Risk），在一定置信水平下的最大可能损失
        CVAR: 条件风险价值（Conditional Value at Risk），超过VaR的期望损失
        VOLATILITY: 波动率，资产收益的标准差
        BETA: 贝塔系数，衡量资产相对于市场的系统性风险
        DRAWDOWN: 回撤，从峰值到谷值的最大跌幅
        SHARPE: 夏普比率，风险调整后的收益
        SORTINO: 索提诺比率，只考虑下行风险的风险调整收益

    使用方法：
        在RiskLimit中指定要监控的风险指标类型。
    """
    VAR = "var"  # 风险价值（Value at Risk）
    CVAR = "cvar"  # 条件风险价值（Conditional Value at Risk）
    VOLATILITY = "volatility"
    BETA = "beta"
    DRAWDOWN = "drawdown"
    SHARPE = "sharpe"
    SORTINO = "sortino"


class PositionSizingMethod(Enum):
    """头寸规模计算方法枚举类。

    目的：
        定义不同的头寸规模计算算法，用于风险调整的头寸管理。

    方法说明：
        FIXED_FRACTIONAL: 固定分数法，根据资本固定比例计算头寸
        FIXED_UNITS: 固定单位法，固定数量的股票/合约
        PERCENT_RISK: 百分比风险法，基于止损距离和风险百分比计算
        KELLY: 凯利公式，最大化对数效用函数的最优投注比例
        OPTIMAL_F: 最优f法，基于历史交易数据计算最优风险比例

    使用方法：
        在calculate_position_size方法中指定要使用的计算方法。
    """
    FIXED_FRACTIONAL = "fixed_fractional"
    FIXED_UNITS = "fixed_units"
    PERCENT_RISK = "percent_risk"
    KELLY = "kelly"
    OPTIMAL_F = "optimal_f"


@dataclass
class RiskLimit:
    """风险限制配置数据类。

    目的：
        定义单个风险指标的限制条件，用于风险监控和违规检查。

    属性说明：
        metric: 风险指标类型（RiskMetric枚举）
        threshold: 阈值，超过此值视为违规
        lookback_period: 回溯周期（交易日数），默认252天（一年）
        confidence_level: 置信水平（仅用于VaR/CVaR计算），默认0.95

    使用方法：
        通过RiskManager.add_risk_limit()方法添加到风险管理器中。
    """
    metric: RiskMetric
    threshold: float
    lookback_period: int = 252  # 交易日数
    confidence_level: float = 0.95  # 用于VaR/CVaR计算的置信水平


@dataclass
class PositionSize:
    """头寸规模计算结果数据类。

    目的：
        封装头寸规模计算的详细结果，便于后续使用和分析。

    属性说明：
        symbol: 资产代码
        quantity: 头寸数量（股数/合约数）
        dollar_amount: 头寸美元价值
        portfolio_percentage: 头寸占投资组合比例
        risk_amount: 风险金额（基于波动率估计）
        method: 使用的计算方法（PositionSizingMethod枚举）

    使用方法：
        作为calculate_position_size方法的返回值，包含所有头寸规模相关信息。
    """
    symbol: str
    quantity: float
    dollar_amount: float
    portfolio_percentage: float
    risk_amount: float
    method: PositionSizingMethod


class RiskManager:
    """投资组合风险管理者。

    目的：
        管理投资组合的整体风险，包括风险限制设置、风险指标计算、头寸规模计算和风险违规检查。

    主要功能：
        1. 管理风险限制（RiskLimit）配置
        2. 计算单个头寸和投资组合层面的风险指标
        3. 监控风险指标是否超过预设限制
        4. 提供多种头寸规模计算方法
        5. 记录风险指标历史数据

    实现方案：
        通过内部状态跟踪当前资本、风险指标历史、风险违规记录，提供完整的风险管理流程。

    使用方法：
        1. 初始化设置初始资本和风险参数
        2. 添加风险限制配置
        3. 定期计算风险指标并检查违规
        4. 使用头寸规模计算方法确定交易规模
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_risk: float = 0.02,  # 单个头寸最大风险2%
        max_portfolio_risk: float = 0.10,  # 投资组合最大总风险10%
        max_drawdown: float = 0.20,  # 最大允许回撤20%
        var_confidence: float = 0.95,
        cvar_confidence: float = 0.99
    ):
        """初始化风险管理者。

        目的：
            设置风险管理器的初始状态和风险参数。

        参数说明：
            initial_capital: 初始资本（美元），默认100,000
            max_position_risk: 单个头寸最大风险（资本比例），默认2%
            max_portfolio_risk: 投资组合最大总风险（资本比例），默认10%
            max_drawdown: 最大允许回撤（比例），默认20%
            var_confidence: VaR计算的置信水平，默认0.95
            cvar_confidence: CVaR计算的置信水平，默认0.99

        实现方案：
            初始化内部状态，包括当前资本、风险限制列表、风险指标历史和当前风险状态。
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_risk = max_position_risk
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown = max_drawdown
        self.var_confidence = var_confidence
        self.cvar_confidence = cvar_confidence

        # Risk limits
        self.risk_limits: List[RiskLimit] = []

        # Risk metrics history
        self.risk_history = pd.DataFrame()

        # Current risk state
        self.current_risk_metrics: Dict[str, float] = {}
        self.risk_violations: List[Dict] = []

        logger.info("Initialized risk manager")

    def add_risk_limit(self, limit: RiskLimit) -> None:
        """添加风险限制配置。

        目的：
            将风险限制配置添加到风险管理器中，用于后续的风险监控。

        参数说明：
            limit: RiskLimit实例，包含风险指标类型和阈值

        实现方案：
            将limit添加到risk_limits列表中，并记录日志。

        使用方法：
            在初始化后，根据需要添加多个风险限制配置。
        """
        self.risk_limits.append(limit)
        logger.info(f"Added risk limit: {limit.metric.value} <= {limit.threshold}")

    def calculate_position_risk(
        self,
        symbol: str,
        price: float,
        quantity: float,
        volatility: float,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """计算单个头寸的风险指标。

        目的：
            评估单个头寸的风险水平，为风险管理和头寸调整提供依据。

        参数说明：
            symbol: 资产代码，用于标识头寸
            price: 当前价格（美元）
            quantity: 头寸数量（正数表示多头，负数表示空头）
            volatility: 年化波动率（比例，如0.15表示15%）
            correlation_matrix: 相关性矩阵（DataFrame），用于计算边际风险贡献（可选）

        返回：
            包含风险指标的字典，包括头寸价值、头寸比例、波动率、美元波动率、VaR、CVaR等。

        实现方案：
            1. 计算头寸价值和占资本比例
            2. 计算波动率和美元波动率
            3. 使用正态分布假设计算VaR和CVaR
            4. 如果提供相关性矩阵，可计算边际风险贡献（当前为占位实现）

        使用方法：
            在交易决策前评估单个头寸的风险，或定期监控现有头寸风险。
        """
        position_value = abs(quantity) * price

        # Calculate basic risk metrics
        risk_metrics = {
            'position_value': position_value,
            'position_percentage': position_value / self.current_capital if self.current_capital > 0 else 0,
            'volatility': volatility,
            'dollar_volatility': position_value * volatility,
        }

        # Calculate VaR (simplified)
        var = self._calculate_var(position_value, volatility)
        risk_metrics['var'] = var
        risk_metrics['var_percentage'] = var / self.current_capital if self.current_capital > 0 else 0

        # Calculate CVaR
        cvar = self._calculate_cvar(position_value, volatility)
        risk_metrics['cvar'] = cvar
        risk_metrics['cvar_percentage'] = cvar / self.current_capital if self.current_capital > 0 else 0

        # Calculate marginal risk contribution if correlation matrix provided
        if correlation_matrix is not None and symbol in correlation_matrix.columns:
            # This would require portfolio context
            pass

        return risk_metrics

    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Dict[str, float]],
        returns_data: Optional[pd.DataFrame] = None,
        covariance_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """计算投资组合层面的风险指标。

        目的：
            评估整个投资组合的综合风险，考虑资产间的相关性效应。

        参数说明：
            positions: 头寸字典，键为资产代码，值为包含头寸信息的字典（至少应有'value'键）
            returns_data: 历史收益数据（DataFrame），索引为日期，列为资产代码（可选）
            covariance_matrix: 收益协方差矩阵（DataFrame），用于精确计算组合波动率（可选）

        返回：
            包含投资组合风险指标的字典，包括组合价值、组合波动率、VaR、CVaR、回撤等。

        实现方案：
            1. 计算投资组合总价值
            2. 根据可用数据选择不同方法计算组合波动率：
               a) 有协方差矩阵：使用权重向量计算组合方差
               b) 有收益数据：从历史组合收益计算波动率
               c) 无详细数据：使用加权平均波动率估计
            3. 计算VaR和CVaR
            4. 如有收益数据，计算最大回撤和当前回撤

        使用方法：
            定期监控投资组合整体风险，用于风险报告和合规检查。
        """
        portfolio_metrics = {}

        # Calculate total portfolio value
        portfolio_value = sum(pos.get('value', 0) for pos in positions.values())
        portfolio_metrics['portfolio_value'] = portfolio_value

        if portfolio_value == 0:
            # Empty portfolio
            portfolio_metrics.update({
                'portfolio_volatility': 0.0,
                'portfolio_var': 0.0,
                'portfolio_cvar': 0.0,
                'portfolio_beta': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0
            })
            return portfolio_metrics

        # Calculate portfolio volatility
        if covariance_matrix is not None and returns_data is not None:
            # Use covariance matrix for precise calculation
            weights = self._calculate_portfolio_weights(positions, portfolio_value)
            portfolio_volatility = self._calculate_portfolio_volatility(weights, covariance_matrix)
        elif returns_data is not None:
            # Estimate from historical returns
            portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        else:
            # Simplified estimation
            portfolio_volatility = self._estimate_portfolio_volatility(positions)

        portfolio_metrics['portfolio_volatility'] = portfolio_volatility
        portfolio_metrics['dollar_volatility'] = portfolio_value * portfolio_volatility

        # Calculate VaR and CVaR
        portfolio_metrics['portfolio_var'] = self._calculate_var(portfolio_value, portfolio_volatility)
        portfolio_metrics['portfolio_cvar'] = self._calculate_cvar(portfolio_value, portfolio_volatility)

        # Calculate drawdown if returns data available
        if returns_data is not None:
            portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
            drawdown_metrics = self._calculate_drawdown_metrics(portfolio_returns, portfolio_value)
            portfolio_metrics.update(drawdown_metrics)

        # Update current risk metrics
        self.current_risk_metrics = portfolio_metrics

        return portfolio_metrics

    def check_risk_limits(self, risk_metrics: Dict[str, float]) -> List[Dict]:
        """检查风险限制是否被违反。

        目的：
            监控当前风险指标是否超过预设的风险限制，及时发现风险违规。

        参数说明：
            risk_metrics: 当前风险指标字典，键为风险指标名称（RiskMetric枚举值），值为指标数值

        返回：
            违规列表，每个违规项为字典，包含指标名称、当前值、阈值和违规类型。

        实现方案：
            1. 遍历所有已添加的风险限制
            2. 从risk_metrics中获取对应指标的值
            3. 比较指标值与阈值：
               - 对于回撤指标：当前值 > 阈值视为违规
               - 对于其他指标：当前值 > 阈值视为违规
            4. 记录所有违规项并更新内部状态

        使用方法：
            在计算风险指标后调用此方法，检查是否需要采取风险控制措施。
        """
        violations = []

        for limit in self.risk_limits:
            metric_value = risk_metrics.get(limit.metric.value)

            if metric_value is not None:
                if limit.metric == RiskMetric.DRAWDOWN:
                    # For drawdown, we check if it exceeds threshold
                    if metric_value > limit.threshold:
                        violations.append({
                            'metric': limit.metric.value,
                            'value': metric_value,
                            'threshold': limit.threshold,
                            'violation': 'exceeded'
                        })
                else:
                    # For other metrics, check if value exceeds threshold
                    if metric_value > limit.threshold:
                        violations.append({
                            'metric': limit.metric.value,
                            'value': metric_value,
                            'threshold': limit.threshold,
                            'violation': 'exceeded'
                        })

        self.risk_violations = violations

        if violations:
            logger.warning(f"Found {len(violations)} risk limit violations")

        return violations

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        volatility: float,
        method: PositionSizingMethod = PositionSizingMethod.PERCENT_RISK,
        risk_per_trade: float = 0.01,  # 每笔交易风险1%
        account_risk: float = 0.02,  # 总账户风险2%
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0
    ) -> PositionSize:
        """计算最优头寸规模。

        目的：
            根据选定的头寸规模计算方法，确定风险调整后的最优头寸数量。

        参数说明：
            symbol: 资产代码
            price: 当前价格（美元）
            volatility: 年化波动率（比例）
            method: 头寸规模计算方法（PositionSizingMethod枚举），默认百分比风险法
            risk_per_trade: 每笔交易风险（资本比例），默认1%
            account_risk: 总账户风险（资本比例），默认2%
            win_rate: 历史胜率（仅用于Kelly和Optimal f方法），默认0.5
            avg_win: 平均盈利金额（仅用于Kelly和Optimal f方法），默认1.0
            avg_loss: 平均亏损金额（仅用于Kelly和Optimal f方法），默认1.0

        返回：
            PositionSize对象，包含计算出的头寸数量、美元价值、组合比例、风险金额等信息。

        实现方案：
            根据method参数调用对应的私有计算方法：
            1. FIXED_FRACTIONAL: 固定分数法
            2. PERCENT_RISK: 百分比风险法（考虑波动率调整）
            3. KELLY: 凯利公式（使用半凯利以降低风险）
            4. OPTIMAL_F: 最优f法（简化版本）
            5. FIXED_UNITS: 固定单位法

        使用方法：
            在交易执行前调用，确定应买入/卖出的数量。
        """
        if method == PositionSizingMethod.FIXED_FRACTIONAL:
            quantity = self._fixed_fractional_size(price, risk_per_trade)
        elif method == PositionSizingMethod.PERCENT_RISK:
            quantity = self._percent_risk_size(price, volatility, risk_per_trade)
        elif method == PositionSizingMethod.KELLY:
            quantity = self._kelly_size(price, win_rate, avg_win, avg_loss)
        elif method == PositionSizingMethod.OPTIMAL_F:
            quantity = self._optimal_f_size(price, win_rate, avg_win, avg_loss)
        else:  # FIXED_UNITS
            quantity = self._fixed_units_size(price, risk_per_trade)

        dollar_amount = quantity * price
        portfolio_percentage = dollar_amount / self.current_capital if self.current_capital > 0 else 0
        risk_amount = self._calculate_position_risk_amount(quantity, price, volatility)

        return PositionSize(
            symbol=symbol,
            quantity=quantity,
            dollar_amount=dollar_amount,
            portfolio_percentage=portfolio_percentage,
            risk_amount=risk_amount,
            method=method
        )

    def _fixed_fractional_size(self, price: float, risk_fraction: float) -> float:
        """固定分数法计算头寸规模。

        目的：
            根据资本固定比例计算头寸数量，不考虑资产波动率。

        参数说明：
            price: 资产价格（美元）
            risk_fraction: 风险分数（资本比例）

        返回：
            头寸数量（四舍五入到整数）

        实现方案：
            风险金额 = 当前资本 × 风险分数
            头寸数量 = 风险金额 / 价格（价格>0时）
        """
        risk_amount = self.current_capital * risk_fraction
        quantity = risk_amount / price if price > 0 else 0
        return round(quantity)

    def _percent_risk_size(self, price: float, volatility: float, risk_percent: float) -> float:
        """百分比风险法计算头寸规模。

        目的：
            根据风险百分比计算头寸数量，考虑资产波动率进行调整。

        参数说明：
            price: 资产价格（美元）
            volatility: 年化波动率（比例）
            risk_percent: 风险百分比（资本比例）

        返回：
            头寸数量（四舍五入到整数）

        实现方案：
            1. 风险金额 = 当前资本 × 风险百分比
            2. 波动率调整因子 = 1.0 / max(波动率, 0.01)（避免除零）
            3. 调整后风险 = 风险金额 × 波动率调整因子
            4. 头寸数量 = 调整后风险 / 价格（价格>0时）

        说明：
            波动率越高，调整因子越小，头寸规模越小，以控制风险。
        """
        risk_amount = self.current_capital * risk_percent
        # 使用波动率调整头寸规模
        volatility_adjustment = 1.0 / max(volatility, 0.01)  # 避免除零
        adjusted_risk = risk_amount * volatility_adjustment
        quantity = adjusted_risk / price if price > 0 else 0
        return round(quantity)

    def _kelly_size(
        self,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """凯利公式计算头寸规模。

        目的：
            使用凯利公式计算最大化对数效用函数的最优投注比例。

        参数说明：
            price: 资产价格（美元）
            win_rate: 胜率（0到1之间）
            avg_win: 平均盈利金额（美元）
            avg_loss: 平均亏损金额（美元）

        返回：
            头寸数量（四舍五入到整数）

        实现方案：
            1. 计算赔率 b = 平均盈利 / 平均亏损
            2. 计算凯利分数 f* = (b × 胜率 - (1 - 胜率)) / b
            3. 如果f* ≤ 0，返回0（无正期望）
            4. 使用半凯利（50%）降低风险：f* = f* × 0.5
            5. 风险金额 = 当前资本 × 凯利分数
            6. 头寸数量 = 风险金额 / 价格（价格>0时）

        说明：
            凯利公式假设已知真实胜率和赔率，实践中常用半凯利以降低模型误差风险。
        """
        # 凯利公式: f* = (bp - q) / b
        # 其中 b = 平均盈利/平均亏损, p = 胜率, q = 1-p
        if avg_loss == 0:
            return 0

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        if b * p - q <= 0:
            return 0

        kelly_fraction = (b * p - q) / b
        # 使用半凯利以降低风险
        kelly_fraction = kelly_fraction * 0.5

        risk_amount = self.current_capital * kelly_fraction
        quantity = risk_amount / price if price > 0 else 0
        return round(quantity)

    def _optimal_f_size(
        self,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """最优f法计算头寸规模（简化版本）。

        目的：
            基于历史交易数据计算最优风险比例，最大化几何平均收益。

        参数说明：
            price: 资产价格（美元）
            win_rate: 胜率（0到1之间）
            avg_win: 平均盈利金额（美元）
            avg_loss: 平均亏损金额（美元）

        返回：
            头寸数量（四舍五入到整数）

        实现方案：
            1. 计算期望值 = 胜率 × 平均盈利 - (1 - 胜率) × 平均亏损
            2. 如果期望值 ≤ 0，返回0（无正期望）
            3. 使用保守风险分数（1%）：风险金额 = 当前资本 × 0.01
            4. 头寸数量 = 风险金额 / 价格（价格>0时）

        说明：
            这是最优f法的简化版本，完整实现需要计算历史交易的几何平均并找到最优f值。
        """
        # 简化最优f计算
        if avg_loss == 0:
            return 0

        # 计算几何平均
        # 这是简化版本 - 真实的最优f计算需要更复杂的计算
        expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss

        if expected_value <= 0:
            return 0

        # 使用期望值的一部分
        risk_fraction = 0.01  # 保守的1%
        risk_amount = self.current_capital * risk_fraction
        quantity = risk_amount / price if price > 0 else 0
        return round(quantity)

    def _fixed_units_size(self, price: float, units: float) -> float:
        """固定单位法计算头寸规模。

        目的：
            直接使用固定的股票/合约数量，忽略价格和资本规模。

        参数说明：
            price: 资产价格（美元）（此方法中未使用）
            units: 固定单位数量

        返回：
            头寸数量（直接返回units参数）

        实现方案：
            直接返回units参数，不做任何计算。

        说明：
            适用于需要固定数量头寸的场景，如指数基金定投。
        """
        return units

    def _calculate_position_risk_amount(
        self,
        quantity: float,
        price: float,
        volatility: float
    ) -> float:
        """计算头寸的风险金额。

        目的：
            基于头寸价值和波动率估计头寸的风险金额。

        参数说明：
            quantity: 头寸数量
            price: 资产价格（美元）
            volatility: 年化波动率（比例）

        返回：
            风险金额（美元）

        实现方案：
            头寸价值 = |数量| × 价格
            风险金额 = 头寸价值 × 波动率

        说明：
            风险金额表示在给定波动率下头寸可能面临的风险规模。
        """
        position_value = abs(quantity) * price
        return position_value * volatility

    def _calculate_var(self, value: float, volatility: float) -> float:
        """计算风险价值（Value at Risk）。

        目的：
            计算在一定置信水平下的最大可能损失。

        参数说明：
            value: 资产或头寸价值（美元）
            volatility: 年化波动率（比例）

        返回：
            VaR值（美元）

        实现方案：
            使用正态分布假设：
            VaR = 价值 × 波动率 × z分数
            其中z分数来自标准正态分布的分位数函数，置信水平由self.var_confidence定义。

        说明：
            这是简化VaR计算，假设收益服从正态分布，未考虑肥尾效应。
        """
        # 使用正态分布的简化VaR计算
        # VaR = 价值 × 波动率 × z分数
        from scipy import stats
        z_score = stats.norm.ppf(self.var_confidence)
        return value * volatility * z_score

    def _calculate_cvar(self, value: float, volatility: float) -> float:
        """计算条件风险价值（Conditional Value at Risk）。

        目的：
            计算超过VaR的期望损失，衡量尾部风险。

        参数说明：
            value: 资产或头寸价值（美元）
            volatility: 年化波动率（比例）

        返回：
            CVaR值（美元）

        实现方案：
            使用正态分布假设：
            CVaR = 价值 × 波动率 × (φ(z分数) / (1 - 置信水平))
            其中φ为标准正态分布的概率密度函数，置信水平由self.cvar_confidence定义。

        说明：
            CVaR比VaR更能反映极端损失的风险，但同样基于正态分布假设。
        """
        # 简化CVaR计算
        # CVaR = 价值 × 波动率 × (φ(z分数) / (1 - 置信水平))
        from scipy import stats
        z_score = stats.norm.ppf(self.cvar_confidence)
        phi_z = stats.norm.pdf(z_score)
        cvar = value * volatility * (phi_z / (1 - self.cvar_confidence))
        return cvar

    def _calculate_portfolio_weights(
        self,
        positions: Dict[str, Dict[str, float]],
        portfolio_value: float
    ) -> pd.Series:
        """计算投资组合权重。

        目的：
            根据头寸价值和投资组合总价值计算各资产的权重。

        参数说明：
            positions: 头寸字典，键为资产代码，值为包含头寸信息的字典
            portfolio_value: 投资组合总价值（美元）

        返回：
            pandas Series，索引为资产代码，值为权重（0到1之间）

        实现方案：
            对每个头寸：权重 = 头寸价值 / 投资组合总价值（如果总价值>0）
        """
        weights = {}
        for symbol, pos_info in positions.items():
            position_value = pos_info.get('value', 0)
            if portfolio_value > 0:
                weights[symbol] = position_value / portfolio_value
            else:
                weights[symbol] = 0

        return pd.Series(weights)

    def _calculate_portfolio_volatility(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> float:
        """使用协方差矩阵计算投资组合波动率。

        目的：
            基于资产收益的协方差矩阵精确计算投资组合波动率。

        参数说明：
            weights: 资产权重Series，索引为资产代码
            covariance_matrix: 收益协方差矩阵DataFrame，行列均为资产代码

        返回：
            投资组合年化波动率（比例）

        实现方案：
            1. 将权重向量与协方差矩阵的列对齐，缺失值填0
            2. 计算组合方差 = 权重^T × 协方差矩阵 × 权重
            3. 组合波动率 = sqrt(组合方差)

        说明：
            这是现代投资组合理论的标准计算方法，考虑了资产间的相关性。
        """
        # Ensure weights align with covariance matrix
        aligned_weights = weights.reindex(covariance_matrix.columns).fillna(0)

        # Portfolio variance = w^T * Σ * w
        portfolio_variance = aligned_weights.T @ covariance_matrix @ aligned_weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        return portfolio_volatility

    def _calculate_portfolio_returns(
        self,
        positions: Dict[str, Dict[str, float]],
        returns_data: pd.DataFrame
    ) -> pd.Series:
        """计算历史投资组合收益。

        目的：
            基于资产历史收益和当前头寸权重计算投资组合的历史收益序列。

        参数说明：
            positions: 头寸字典，键为资产代码，值为包含头寸信息的字典（应有'weight'键）
            returns_data: 资产历史收益DataFrame，索引为日期，列为资产代码

        返回：
            pandas Series，索引为日期，值为投资组合日收益

        实现方案：
            1. 初始化全零收益序列
            2. 对每个资产，如果存在于收益数据中，则：组合收益 += 资产收益 × 资产权重

        说明：
            这是简化计算，假设头寸权重在历史期间保持不变。
        """
        # This is a simplified calculation
        # In practice, you would need to track position changes over time
        portfolio_returns = pd.Series(0.0, index=returns_data.index)

        for symbol, pos_info in positions.items():
            if symbol in returns_data.columns:
                weight = pos_info.get('weight', 0)
                portfolio_returns += returns_data[symbol] * weight

        return portfolio_returns

    def _estimate_portfolio_volatility(
        self,
        positions: Dict[str, Dict[str, float]]
    ) -> float:
        """估计投资组合波动率（无详细数据时）。

        目的：
            在没有协方差矩阵或历史收益数据时，估计投资组合波动率。

        参数说明：
            positions: 头寸字典，键为资产代码，值为包含头寸信息的字典（应有'volatility'和'weight'键）

        返回：
            估计的投资组合年化波动率（比例）

        实现方案：
            1. 计算加权平均波动率：总和(波动率 × 权重) / 总权重
            2. 如果总权重为0，返回默认市场波动率（15%）

        说明：
            这是粗略估计，假设资产间完全正相关（最保守估计）。
        """
        # Simplified estimation: weighted average of position volatilities
        total_volatility = 0.0
        total_weight = 0.0

        for pos_info in positions.values():
            volatility = pos_info.get('volatility', 0.15)  # Default 15% volatility
            weight = pos_info.get('weight', 0)
            total_volatility += volatility * weight
            total_weight += weight

        if total_weight > 0:
            return total_volatility / total_weight
        return 0.15  # Default market volatility

    def _calculate_drawdown_metrics(
        self,
        returns: pd.Series,
        current_value: float
    ) -> Dict[str, float]:
        """从收益序列计算回撤指标。

        目的：
            计算投资组合的最大回撤和当前回撤，衡量下行风险。

        参数说明：
            returns: 投资组合收益序列（通常为日收益）
            current_value: 当前投资组合价值（美元）

        返回：
            包含最大回撤和当前回撤的字典。

        实现方案：
            1. 计算累积收益序列
            2. 计算运行最大值序列
            3. 回撤序列 = (运行最大值 - 累积收益) / 运行最大值
            4. 最大回撤 = 回撤序列的最大值
            5. 当前回撤 = (当前运行最大值 - 当前累积收益) / 当前运行最大值

        说明：
            回撤是衡量投资组合风险的重要指标，反映从峰值到谷值的最大跌幅。
        """
        if returns.empty:
            return {'max_drawdown': 0.0, 'current_drawdown': 0.0}

        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cumulative.expanding().max()

        # Calculate drawdown series
        drawdown = (running_max - cumulative) / running_max

        max_drawdown = drawdown.max() if not drawdown.empty else 0.0

        # Current drawdown
        if len(cumulative) > 0:
            current_cumulative = cumulative.iloc[-1]
            current_max = running_max.iloc[-1]
            current_drawdown = (current_max - current_cumulative) / current_max if current_max > 0 else 0.0
        else:
            current_drawdown = 0.0

        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown
        }


class PositionSizer:
    """头寸规模计算器。

    目的：
        提供基于止损的头寸规模计算，专注于风险管理。

    主要功能：
        1. 根据止损距离计算头寸规模
        2. 支持多种头寸规模计算方法
        3. 提供金字塔加仓（pyramiding）计算
        4. 应用头寸规模限制

    实现方案：
        使用RiskManager作为底层风险计算引擎，专注于止损驱动的头寸规模计算。

    使用方法：
        1. 初始化设置账户规模
        2. 使用calculate_size计算基础头寸规模
        3. 使用calculate_pyramiding_size计算加仓规模
    """

    def __init__(self, account_size: float = 100000.0):
        """初始化头寸规模计算器。

        目的：
            设置头寸规模计算器的账户规模，并初始化底层RiskManager。

        参数说明：
            account_size: 账户总规模（美元），默认100,000

        实现方案：
            存储账户规模，创建RiskManager实例用于风险计算。
        """
        self.account_size = account_size
        self.risk_manager = RiskManager(initial_capital=account_size)

    def calculate_size(
        self,
        symbol: str,
        price: float,
        stop_loss: float,
        method: PositionSizingMethod = PositionSizingMethod.PERCENT_RISK,
        risk_per_trade: float = 0.01,
        max_position_size: float = 0.1
    ) -> PositionSize:
        """基于止损计算头寸规模。

        目的：
            根据止损距离和风险参数计算风险调整后的头寸规模。

        参数说明：
            symbol: 资产代码
            price: 入场价格（美元）
            stop_loss: 止损价格（美元）
            method: 头寸规模计算方法，默认百分比风险法
            risk_per_trade: 每笔交易风险（账户比例），默认1%
            max_position_size: 最大头寸规模（账户比例），默认10%

        返回：
            PositionSize对象，包含计算出的头寸数量和相关风险信息。

        实现方案：
            1. 计算每股风险 = |入场价格 - 止损价格|
            2. 计算风险金额 = 账户规模 × 每笔交易风险
            3. 根据方法计算头寸数量：
               - 固定分数法：头寸价值 = 账户规模 × 风险比例，数量 = 头寸价值 / 价格
               - 百分比风险法：数量 = 风险金额 / 每股风险
               - 固定单位法：数量 = 风险比例 × 100（示例）
            4. 应用头寸规模限制：头寸价值 ≤ 账户规模 × 最大头寸比例
            5. 四舍五入到整数股数

        使用方法：
            在设置止损的交易中，用于确定应交易的数量。
        """
        # Calculate risk per share
        risk_per_share = abs(price - stop_loss)

        if risk_per_share <= 0:
            raise RiskError("Stop loss must be different from entry price")

        # Calculate risk amount
        risk_amount = self.account_size * risk_per_trade

        # Calculate position size
        if method == PositionSizingMethod.FIXED_FRACTIONAL:
            position_value = self.account_size * risk_per_trade
            quantity = position_value / price
        elif method == PositionSizingMethod.PERCENT_RISK:
            quantity = risk_amount / risk_per_share
        elif method == PositionSizingMethod.FIXED_UNITS:
            # Fixed number of shares/contracts
            quantity = risk_per_trade * 100  # Example: 1% = 100 shares
        else:
            # Default to percent risk
            quantity = risk_amount / risk_per_share

        # Apply position size limits
        position_value = quantity * price
        max_position_value = self.account_size * max_position_size

        if position_value > max_position_value:
            quantity = max_position_value / price
            position_value = max_position_value

        # Round to nearest whole share
        quantity = round(quantity)

        # Recalculate with rounded quantity
        position_value = quantity * price
        portfolio_percentage = position_value / self.account_size if self.account_size > 0 else 0
        actual_risk = quantity * risk_per_share

        return PositionSize(
            symbol=symbol,
            quantity=quantity,
            dollar_amount=position_value,
            portfolio_percentage=portfolio_percentage,
            risk_amount=actual_risk,
            method=method
        )

    def calculate_pyramiding_size(
        self,
        symbol: str,
        current_price: float,
        entry_prices: List[float],
        current_quantity: float,
        stop_loss: float,
        risk_per_trade: float = 0.01,
        max_adds: int = 3
    ) -> Optional[PositionSize]:
        """计算金字塔加仓（pyramiding）头寸规模。

        目的：
            为盈利头寸计算追加头寸的规模，实现金字塔加仓策略。

        参数说明：
            symbol: 资产代码
            current_price: 当前市场价格（美元）
            entry_prices: 之前的入场价格列表
            current_quantity: 当前头寸数量
            stop_loss: 当前止损价格（美元）
            risk_per_trade: 每笔交易风险（账户比例），默认1%
            max_adds: 最大加仓次数，默认3次

        返回：
            如果允许加仓，返回PositionSize对象（表示追加的头寸）；否则返回None。

        实现方案：
            1. 检查是否达到最大加仓次数，达到则返回None
            2. 检查价格是否朝有利方向移动：当前价格 > 平均入场价格
            3. 计算金字塔加仓风险：风险比例 × (0.5 ^ 已加仓次数)（每次减半）
            4. 计算每股风险 = |当前价格 - 止损价格|
            5. 计算风险金额 = 账户规模 × 金字塔加仓风险
            6. 追加数量 = 风险金额 / 每股风险，四舍五入到整数

        使用方法：
            在趋势跟踪策略中，当价格朝有利方向移动时，用于计算追加头寸的规模。
        """
        if len(entry_prices) >= max_adds:
            return None  # 已达到最大加仓次数

        # 检查价格是否朝有利方向移动
        avg_entry_price = np.mean(entry_prices)
        if current_price <= avg_entry_price:
            return None  # 价格未朝有利方向移动

        # 计算新的风险金额（金字塔加仓降低风险）
        pyramiding_risk = risk_per_trade * (0.5 ** len(entry_prices))  # 每次加仓风险减半

        # 计算新的头寸规模
        risk_per_share = abs(current_price - stop_loss)
        risk_amount = self.account_size * pyramiding_risk
        add_quantity = risk_amount / risk_per_share

        # 四舍五入并计算
        add_quantity = round(add_quantity)
        add_value = add_quantity * current_price

        total_quantity = current_quantity + add_quantity
        total_value = total_quantity * current_price
        portfolio_percentage = total_value / self.account_size if self.account_size > 0 else 0

        return PositionSize(
            symbol=symbol,
            quantity=add_quantity,  # 追加的数量
            dollar_amount=add_value,
            portfolio_percentage=portfolio_percentage,
            risk_amount=add_quantity * risk_per_share,
            method=PositionSizingMethod.PERCENT_RISK
        )
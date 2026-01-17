"""
股票投资组合具体实现模块。

本文件提供了 EquityPortfolio 类的具体实现，这是 Portfolio 抽象基类的第一个具体实现。
EquityPortfolio 专门用于管理股票投资组合，支持完整的投资组合管理功能，包括：
持仓管理、现金管理、绩效跟踪、再平衡操作和风险管理。

实现方案概述：
1. 继承自 Portfolio 抽象基类，实现所有抽象方法
2. 使用字典数据结构管理持仓、资产和目标配置
3. 集成现有的再平衡策略 (rebalancing.py) 和风险管理模块 (risk.py)
4. 遵循项目的异常处理体系，使用 PortfolioError 和 ValidationError
5. 使用 loguru 进行结构化日志记录

主要类：
- EquityPortfolio: 股票投资组合的具体实现类

与系统中其他模块的关系：
1. 依赖 base.py 中的抽象基类和数据结构
2. 使用 rebalancing.py 中的再平衡策略进行权重优化
3. 使用 risk.py 中的 RiskManager 进行风险管理
4. 与回测引擎集成，作为持仓管理组件
5. 使用 utils 模块中的异常处理和日志记录

注意：本实现假设所有资产均为股票类型，价格以组合货币计价。
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ...utils.exceptions import PortfolioError, ValidationError
from ...utils.logger import get_logger
from .base import (
    Portfolio, PortfolioStatus, RebalancingFrequency,
    Asset, PortfolioAllocation, PortfolioHolding, PortfolioSnapshot
)
from .rebalancing import create_rebalancer, RebalancingStrategy
from .risk import RiskManager, PositionSizer

logger = get_logger(__name__)


class EquityPortfolio(Portfolio):
    """
    股票投资组合具体实现类。

    这个类提供了 Portfolio 抽象基类的完整实现，专门用于管理股票投资组合。
    支持多资产持仓管理、自动再平衡、风险控制和绩效跟踪。

    主要功能：
    1. 持仓管理：买入、卖出、持仓更新
    2. 现金管理：现金余额跟踪和分配
    3. 配置管理：目标权重设置和调整
    4. 再平衡：定期或阈值触发再平衡操作
    5. 风险管理：头寸限制、杠杆控制、风险指标计算
    6. 绩效跟踪：组合价值、收益率、波动率等指标

    使用示例：
        portfolio = EquityPortfolio(initial_capital=100000)
        portfolio.add_allocation('AAPL', target_weight=0.3)
        portfolio.update_prices({'AAPL': 150.0})
        portfolio.execute_trade('AAPL', quantity=100, price=150.0)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        name: str = "Equity Portfolio",
        currency: str = 'USD',
        benchmark_symbol: Optional[str] = None,
        rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY,
        rebalancing_strategy: str = 'equal_weight',
        allow_short: bool = False,
        max_leverage: float = 1.0,
        risk_manager: Optional[RiskManager] = None,
        position_sizer: Optional[PositionSizer] = None
    ):
        """
        初始化股票投资组合。

        Args:
            initial_capital: 初始资金
            name: 组合名称
            currency: 货币
            benchmark_symbol: 基准指数代码
            rebalancing_frequency: 再平衡频率
            rebalancing_strategy: 再平衡策略类型
            allow_short: 是否允许做空
            max_leverage: 最大杠杆率
            risk_manager: 风险管理器实例（可选）
            position_sizer: 头寸规模计算器实例（可选）
        """
        # 调用父类初始化
        super().__init__(
            initial_capital=initial_capital,
            name=name,
            currency=currency,
            benchmark_symbol=benchmark_symbol,
            rebalancing_frequency=rebalancing_frequency,
            allow_short=allow_short,
            max_leverage=max_leverage
        )

        # 初始化再平衡策略
        self.rebalancing_strategy_type = rebalancing_strategy
        self.rebalancer = create_rebalancer(rebalancing_strategy)

        # 初始化风险管理和头寸规模计算
        self.risk_manager = risk_manager or RiskManager(initial_capital=initial_capital)
        self.position_sizer = position_sizer or PositionSizer(account_size=initial_capital)

        # 交易历史记录
        self.trade_history: List[Dict] = []

        # 再平衡阈值（默认5%）
        self.rebalancing_threshold = 0.05

        logger.info(f"创建股票投资组合 '{name}'，初始资金: {initial_capital:,.2f} {currency}")
        logger.info(f"再平衡策略: {rebalancing_strategy}，频率: {rebalancing_frequency.value}")

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        更新资产价格。

        这个方法更新持仓资产的当前价格，并重新计算持仓市值、未实现盈亏等指标。

        Args:
            prices: 资产代码到价格的字典映射

        Raises:
            PortfolioError: 价格更新失败时抛出
        """
        try:
            update_count = 0
            for symbol, price in prices.items():
                if symbol in self.holdings:
                    holding = self.holdings[symbol]

                    # 更新价格和市值相关指标
                    holding.current_price = price
                    holding.market_value = holding.quantity * price
                    holding.unrealized_pnl = (price - holding.avg_price) * holding.quantity

                    update_count += 1

            if update_count > 0:
                logger.debug(f"更新了 {update_count} 个资产的价格")
            else:
                logger.warning(f"未找到任何匹配的持仓资产来更新价格")

        except Exception as e:
            raise PortfolioError(
                f"价格更新失败: {str(e)}",
                portfolio_id=self.name,
                operation="update_prices",
                details={"prices": prices}
            ) from e

    def calculate_weights(self) -> Dict[str, float]:
        """
        计算当前投资组合权重。

        权重计算基于持仓市值与组合总价值的比例。
        现金不包含在权重计算中。

        Returns:
            资产代码到权重的字典映射

        Raises:
            PortfolioError: 权重计算失败时抛出
        """
        try:
            total_value = self.get_total_value()
            cash_value = self.current_capital

            # 计算投资部分总价值（排除现金）
            investment_value = total_value - cash_value

            if investment_value <= 0:
                return {}

            weights = {}
            for symbol, holding in self.holdings.items():
                if holding.market_value > 0:
                    weight = holding.market_value / investment_value
                    weights[symbol] = weight

            # 验证权重和为1（允许微小误差）
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 0.001 and weight_sum > 0:
                # 重新归一化
                weights = {k: v / weight_sum for k, v in weights.items()}

            return weights

        except Exception as e:
            raise PortfolioError(
                f"权重计算失败: {str(e)}",
                portfolio_id=self.name,
                operation="calculate_weights"
            ) from e

    def calculate_deviation(self) -> Dict[str, float]:
        """
        计算当前权重与目标权重的偏差。

        偏差计算为当前权重减去目标权重。
        正偏差表示超配，负偏差表示低配。

        Returns:
            资产代码到偏差的字典映射

        Raises:
            PortfolioError: 偏差计算失败时抛出
        """
        try:
            current_weights = self.calculate_weights()
            deviations = {}

            # 将目标配置列表转换为字典
            target_weights = {}
            for allocation in self.target_allocations:
                target_weights[allocation.symbol] = allocation.target_weight

            # 计算偏差
            all_symbols = set(current_weights.keys()) | set(target_weights.keys())
            for symbol in all_symbols:
                current = current_weights.get(symbol, 0.0)
                target = target_weights.get(symbol, 0.0)
                deviation = current - target
                deviations[symbol] = deviation

            return deviations

        except Exception as e:
            raise PortfolioError(
                f"偏差计算失败: {str(e)}",
                portfolio_id=self.name,
                operation="calculate_deviation"
            ) from e

    def needs_rebalancing(self, threshold: float = 0.05) -> bool:
        """
        检查投资组合是否需要再平衡。

        检查标准：
        1. 任何资产的权重偏差超过阈值
        2. 达到再平衡频率时间点
        3. 组合结构发生重大变化

        Args:
            threshold: 再平衡阈值（默认5%）

        Returns:
            如果需要再平衡返回True，否则返回False
        """
        # 检查阈值触发
        deviations = self.calculate_deviation()
        for symbol, deviation in deviations.items():
            if abs(deviation) > threshold:
                logger.info(f"资产 {symbol} 偏差 {deviation:.2%} 超过阈值 {threshold:.2%}，触发再平衡")
                return True

        # 检查时间触发
        if self.last_rebalanced is None:
            # 从未再平衡过，需要第一次再平衡
            return True

        # 根据再平衡频率检查时间
        current_time = datetime.now()
        time_since_rebalance = current_time - self.last_rebalanced

        frequency_days = {
            RebalancingFrequency.DAILY: 1,
            RebalancingFrequency.WEEKLY: 7,
            RebalancingFrequency.MONTHLY: 30,
            RebalancingFrequency.QUARTERLY: 90,
            RebalancingFrequency.YEARLY: 365,
            RebalancingFrequency.NEVER: float('inf')
        }

        required_days = frequency_days.get(self.rebalancing_frequency, 30)
        if time_since_rebalance.days >= required_days:
            logger.info(f"距离上次再平衡 {time_since_rebalance.days} 天，达到 {self.rebalancing_frequency.value} 频率，触发再平衡")
            return True

        return False

    def rebalance(self, target_allocations: List[PortfolioAllocation]) -> List[Dict]:
        """
        执行投资组合再平衡。

        再平衡过程：
        1. 验证目标配置
        2. 计算当前权重和目标权重
        3. 使用再平衡策略计算交易指令
        4. 执行交易指令
        5. 更新再平衡时间戳

        Args:
            target_allocations: 目标配置列表

        Returns:
            执行的交易指令列表

        Raises:
            PortfolioError: 再平衡失败时抛出
        """
        try:
            # 验证目标配置
            for allocation in target_allocations:
                self.validate_allocation(allocation)

            # 更新目标配置
            self.target_allocations = target_allocations

            # 获取当前价格
            prices = {}
            for symbol in self.holdings:
                if symbol in self.holdings and self.holdings[symbol].current_price > 0:
                    prices[symbol] = self.holdings[symbol].current_price

            # 添加目标配置中但当前无持仓的资产价格（需要外部提供）
            for allocation in target_allocations:
                if allocation.symbol not in prices:
                    # 在实际应用中，这里需要从数据源获取价格
                    # 目前设置为0，交易计算时会跳过
                    prices[allocation.symbol] = 0

            # 计算当前权重和目标权重
            current_weights = self.calculate_weights()
            target_weights = {a.symbol: a.target_weight for a in target_allocations}

            # 准备约束条件
            constraints = {
                'min_trade_size': 1,  # 最小交易单位
                'min_weight': 0.0,    # 最小权重
                'max_weight': 1.0,    # 最大权重
                'commission': 0.001,  # 佣金率（0.1%）
            }

            # 使用再平衡策略计算交易
            trades = self.rebalancer.calculate_rebalancing_trades(
                current_weights=current_weights,
                target_weights=target_weights,
                portfolio_value=self.get_total_value(),
                prices=prices,
                constraints=constraints
            )

            # 执行交易
            executed_trades = []
            for trade in trades:
                symbol = trade['symbol']
                quantity = trade['quantity']
                side = trade['side']

                if symbol not in prices or prices[symbol] <= 0:
                    logger.warning(f"资产 {symbol} 无有效价格，跳过交易")
                    continue

                # 根据买卖方向调整数量
                trade_quantity = quantity if side == 'buy' else -quantity

                try:
                    self.execute_trade(
                        symbol=symbol,
                        quantity=trade_quantity,
                        price=prices[symbol],
                        commission=constraints['commission']
                    )

                    executed_trades.append(trade)

                except Exception as trade_error:
                    logger.error(f"执行交易失败: {trade_error}")
                    continue

            # 更新再平衡时间戳
            self.last_rebalanced = datetime.now()

            logger.info(f"再平衡完成，执行了 {len(executed_trades)} 笔交易")
            return executed_trades

        except Exception as e:
            raise PortfolioError(
                f"再平衡失败: {str(e)}",
                portfolio_id=self.name,
                operation="rebalance",
                details={"target_allocations": [a.to_dict() for a in target_allocations] if hasattr(target_allocations[0], 'to_dict') else str(target_allocations)}
            ) from e

    def add_allocation(
        self,
        symbol: str,
        target_weight: float,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        is_core: bool = True
    ) -> None:
        """
        添加或更新目标配置。

        如果资产已存在目标配置，则更新配置；否则添加新配置。

        Args:
            symbol: 资产代码
            target_weight: 目标权重
            min_weight: 最小权重
            max_weight: 最大权重
            is_core: 是否为核心持仓

        Raises:
            PortfolioError: 添加配置失败时抛出
            ValidationError: 参数验证失败时抛出
        """
        try:
            # 创建配置对象
            allocation = PortfolioAllocation(
                symbol=symbol,
                target_weight=target_weight,
                min_weight=min_weight,
                max_weight=max_weight,
                is_core=is_core
            )

            # 验证配置
            self.validate_allocation(allocation)

            # 检查是否已存在
            existing_index = None
            for i, existing in enumerate(self.target_allocations):
                if existing.symbol == symbol:
                    existing_index = i
                    break

            if existing_index is not None:
                # 更新现有配置
                self.target_allocations[existing_index] = allocation
                logger.info(f"更新资产 {symbol} 的目标配置: 权重={target_weight:.2%}")
            else:
                # 添加新配置
                self.target_allocations.append(allocation)
                logger.info(f"添加资产 {symbol} 的目标配置: 权重={target_weight:.2%}")

            # 验证总权重不超过100%
            total_weight = sum(a.target_weight for a in self.target_allocations)
            if total_weight > 1.0:
                logger.warning(f"目标配置总权重 {total_weight:.2%} 超过100%，建议调整")

        except ValidationError as ve:
            raise ve
        except Exception as e:
            raise PortfolioError(
                f"添加目标配置失败: {str(e)}",
                portfolio_id=self.name,
                operation="add_allocation",
                details={"symbol": symbol, "target_weight": target_weight}
            ) from e

    def remove_allocation(self, symbol: str) -> None:
        """
        移除目标配置。

        如果资产存在持仓，仅移除目标配置，不清除持仓。

        Args:
            symbol: 要移除的资产代码

        Raises:
            PortfolioError: 移除配置失败时抛出
        """
        try:
            # 查找配置
            original_count = len(self.target_allocations)
            self.target_allocations = [
                a for a in self.target_allocations if a.symbol != symbol
            ]

            removed_count = original_count - len(self.target_allocations)
            if removed_count > 0:
                logger.info(f"移除资产 {symbol} 的目标配置")
            else:
                logger.warning(f"资产 {symbol} 不存在目标配置，无需移除")

        except Exception as e:
            raise PortfolioError(
                f"移除目标配置失败: {str(e)}",
                portfolio_id=self.name,
                operation="remove_allocation",
                details={"symbol": symbol}
            ) from e

    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float = 0.0
    ) -> None:
        """
        执行交易并更新投资组合。

        交易执行流程：
        1. 验证交易参数
        2. 检查资金充足性（买入）或持仓充足性（卖出）
        3. 计算交易成本
        4. 更新持仓和现金
        5. 记录交易历史

        Args:
            symbol: 资产代码
            quantity: 交易数量（正数为买入，负数为卖出）
            price: 交易价格
            commission: 佣金费用

        Raises:
            PortfolioError: 交易执行失败时抛出
            ValidationError: 交易参数无效时抛出
        """
        try:
            # 参数验证
            if price <= 0:
                raise ValidationError(
                    f"交易价格必须大于0，当前价格: {price}",
                    field="price",
                    value=price,
                    expected="> 0"
                )

            if quantity == 0:
                raise ValidationError(
                    "交易数量不能为0",
                    field="quantity",
                    value=quantity,
                    expected="!= 0"
                )

            # 计算交易金额和费用
            trade_value = quantity * price
            commission_cost = abs(trade_value) * commission

            if quantity > 0:  # 买入
                # 检查资金是否充足
                total_cost = trade_value + commission_cost
                if total_cost > self.current_capital:
                    raise PortfolioError(
                        f"资金不足：需要 {total_cost:.2f}，可用 {self.current_capital:.2f}",
                        portfolio_id=self.name,
                        operation="buy",
                        details={
                            "symbol": symbol,
                            "quantity": quantity,
                            "price": price,
                            "required_capital": total_cost,
                            "available_capital": self.current_capital
                        }
                    )

                # 更新现金
                self.current_capital -= total_cost

                # 更新或创建持仓
                if symbol in self.holdings:
                    holding = self.holdings[symbol]
                    # 计算新的平均价格
                    total_quantity = holding.quantity + quantity
                    total_cost_basis = holding.cost_basis + trade_value
                    new_avg_price = total_cost_basis / total_quantity if total_quantity > 0 else 0

                    holding.quantity = total_quantity
                    holding.avg_price = new_avg_price
                    holding.cost_basis = total_cost_basis
                    holding.current_price = price
                    holding.market_value = holding.quantity * price
                    holding.unrealized_pnl = (price - new_avg_price) * holding.quantity

                else:
                    # 创建新持仓
                    holding = PortfolioHolding(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=price,
                        current_price=price,
                        market_value=quantity * price,
                        cost_basis=trade_value,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        entry_date=datetime.now()
                    )
                    self.holdings[symbol] = holding

                    # 如果还没有资产信息，创建基本资产信息
                    if symbol not in self.assets:
                        self.assets[symbol] = Asset(
                            symbol=symbol,
                            name=symbol,
                            asset_type='stock',
                            currency=self.currency
                        )

                action = "买入"

            else:  # 卖出（quantity < 0）
                # 检查持仓是否充足
                if symbol not in self.holdings:
                    raise PortfolioError(
                        f"没有 {symbol} 的持仓可卖出",
                        portfolio_id=self.name,
                        operation="sell",
                        details={"symbol": symbol, "quantity": quantity}
                    )

                holding = self.holdings[symbol]
                if abs(quantity) > holding.quantity:
                    raise PortfolioError(
                        f"卖出数量超过持仓数量：尝试卖出 {abs(quantity)}，持仓 {holding.quantity}",
                        portfolio_id=self.name,
                        operation="sell",
                        details={
                            "symbol": symbol,
                            "attempted_sell": abs(quantity),
                            "current_holding": holding.quantity
                        }
                    )

                # 计算实现盈亏
                sell_quantity = abs(quantity)
                realized_pnl = (price - holding.avg_price) * sell_quantity

                # 更新现金
                proceeds = trade_value - commission_cost  # 卖出获得现金（负的trade_value）
                self.current_capital += abs(proceeds)

                # 更新持仓
                holding.quantity -= sell_quantity
                if holding.quantity == 0:
                    # 清仓，移除持仓记录
                    del self.holdings[symbol]
                    holding.realized_pnl += realized_pnl
                    # 保留持仓记录用于历史跟踪（可选）
                else:
                    # 部分卖出，更新成本基础
                    proportion_sold = sell_quantity / (holding.quantity + sell_quantity)
                    cost_sold = holding.cost_basis * proportion_sold
                    holding.cost_basis -= cost_sold
                    holding.realized_pnl += realized_pnl
                    holding.current_price = price
                    holding.market_value = holding.quantity * price
                    holding.unrealized_pnl = (price - holding.avg_price) * holding.quantity

                action = "卖出"

            # 记录交易历史
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'commission': commission_cost,
                'action': action,
                'remaining_cash': self.current_capital,
                'portfolio_value': self.get_total_value()
            }
            self.trade_history.append(trade_record)

            # 记录快照
            self.take_snapshot()

            logger.info(
                f"{action} {abs(quantity):.2f} 股 {symbol} @ {price:.2f}，"
                f"价值: {trade_value:,.2f}，佣金: {commission_cost:.2f}，"
                f"剩余现金: {self.current_capital:,.2f}"
            )

        except (PortfolioError, ValidationError):
            raise
        except Exception as e:
            raise PortfolioError(
                f"交易执行失败: {str(e)}",
                portfolio_id=self.name,
                operation="execute_trade",
                details={
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "commission": commission
                }
            ) from e

    def take_snapshot(self) -> PortfolioSnapshot:
        """
        创建投资组合状态快照。

        快照包含投资组合在特定时间点的完整状态，
        用于绩效跟踪和历史分析。

        Returns:
            投资组合快照对象

        Raises:
            PortfolioError: 快照创建失败时抛出
        """
        try:
            # 计算当前权重
            current_weights = self.calculate_weights()

            # 计算绩效指标
            performance = self.get_performance_summary()

            # 创建快照
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=self.get_total_value(),
                cash=self.current_capital,
                holdings=self.holdings.copy(),
                allocations=current_weights,
                performance=performance
            )

            # 添加到快照历史
            self.snapshots.append(snapshot)

            # 更新绩效历史数据框
            snapshot_dict = {
                'timestamp': snapshot.timestamp,
                'total_value': snapshot.total_value,
                'cash': snapshot.cash,
                'num_holdings': len(snapshot.holdings)
            }
            snapshot_dict.update(performance)

            new_row = pd.DataFrame([snapshot_dict])
            self.performance_history = pd.concat([self.performance_history, new_row], ignore_index=True)

            logger.debug(f"创建投资组合快照，总价值: {snapshot.total_value:,.2f}")
            return snapshot

        except Exception as e:
            raise PortfolioError(
                f"快照创建失败: {str(e)}",
                portfolio_id=self.name,
                operation="take_snapshot"
            ) from e

    def get_trade_history(self) -> List[Dict]:
        """
        获取交易历史记录。

        Returns:
            交易历史记录列表
        """
        return self.trade_history.copy()

    def get_asset_info(self, symbol: str) -> Optional[Asset]:
        """
        获取资产信息。

        Args:
            symbol: 资产代码

        Returns:
            资产信息对象，如果不存在则返回None
        """
        return self.assets.get(symbol)

    def get_holding_details(self, symbol: str) -> Optional[PortfolioHolding]:
        """
        获取持仓详细信息。

        Args:
            symbol: 资产代码

        Returns:
            持仓信息对象，如果不存在则返回None
        """
        return self.holdings.get(symbol)

    def get_target_allocation(self, symbol: str) -> Optional[PortfolioAllocation]:
        """
        获取目标配置信息。

        Args:
            symbol: 资产代码

        Returns:
            目标配置对象，如果不存在则返回None
        """
        for allocation in self.target_allocations:
            if allocation.symbol == symbol:
                return allocation
        return None

    def close_position(self, symbol: str, price: float) -> float:
        """
        平仓指定资产的所有持仓。

        Args:
            symbol: 资产代码
            price: 平仓价格

        Returns:
            实现盈亏金额

        Raises:
            PortfolioError: 平仓失败时抛出
        """
        if symbol not in self.holdings:
            raise PortfolioError(
                f"没有 {symbol} 的持仓可平仓",
                portfolio_id=self.name,
                operation="close_position",
                details={"symbol": symbol}
            )

        holding = self.holdings[symbol]
        quantity = holding.quantity

        # 执行卖出交易
        self.execute_trade(
            symbol=symbol,
            quantity=-quantity,  # 卖出全部
            price=price,
            commission=0.0
        )

        # 计算实现盈亏
        realized_pnl = (price - holding.avg_price) * quantity

        logger.info(f"平仓 {symbol}，数量: {quantity}，价格: {price:.2f}，盈亏: {realized_pnl:,.2f}")
        return realized_pnl
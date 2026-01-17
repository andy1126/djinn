"""
Djinn 投资组合管理模块。

这个模块提供了完整的投资组合管理功能，包括：
1. 投资组合抽象基类和数据结构
2. 股票投资组合具体实现
3. 多种再平衡策略（等权重、风险平价、均值-方差优化、Black-Litterman模型）
4. 风险管理和头寸规模计算
5. 工厂函数，支持从配置创建投资组合

主要组件：
- EquityPortfolio: 股票投资组合的具体实现
- Portfolio: 投资组合抽象基类，定义通用接口
- RebalancingStrategy: 再平衡策略抽象基类
- RiskManager: 风险管理器，管理风险限制和指标
- PositionSizer: 头寸规模计算器，计算最优头寸大小

使用示例：
    from djinn.core.portfolio import create_portfolio, EquityPortfolio

    # 从配置创建投资组合
    portfolio = create_portfolio(
        initial_capital=100000,
        name='我的投资组合',
        rebalancing_frequency='monthly'
    )

    # 或直接创建
    portfolio = EquityPortfolio(initial_capital=100000)
    portfolio.add_allocation('AAPL', target_weight=0.3)
    portfolio.update_prices({'AAPL': 150.0})
    portfolio.execute_trade('AAPL', quantity=100, price=150.0)
"""

# 基础类和数据结构
from .base import (
    PortfolioStatus,
    RebalancingFrequency,
    Asset,
    PortfolioAllocation,
    PortfolioHolding,
    PortfolioSnapshot,
    Portfolio,
    RebalancingStrategy
)

# 具体投资组合实现
from .equity import EquityPortfolio

# 再平衡策略
from .rebalancing import (
    EqualWeightRebalancer,
    RiskParityRebalancer,
    MeanVarianceRebalancer,
    BlackLittermanRebalancer,
    create_rebalancer
)

# 风险管理
from .risk import (
    RiskManager,
    PositionSizer,
    RiskMetric,
    PositionSizingMethod,
    RiskLimit,
    PositionSize
)

# 工厂函数
from .factory import (
    create_portfolio,
    create_portfolio_from_config_file
)

# 导出列表
__all__ = [
    # 基础类
    'PortfolioStatus',
    'RebalancingFrequency',
    'Asset',
    'PortfolioAllocation',
    'PortfolioHolding',
    'PortfolioSnapshot',
    'Portfolio',
    'RebalancingStrategy',

    # 具体实现
    'EquityPortfolio',

    # 再平衡策略
    'EqualWeightRebalancer',
    'RiskParityRebalancer',
    'MeanVarianceRebalancer',
    'BlackLittermanRebalancer',
    'create_rebalancer',

    # 风险管理
    'RiskManager',
    'PositionSizer',
    'RiskMetric',
    'PositionSizingMethod',
    'RiskLimit',
    'PositionSize',

    # 工厂函数
    'create_portfolio',
    'create_portfolio_from_config_file'
]
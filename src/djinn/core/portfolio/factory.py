"""
投资组合工厂模块。

本文件提供了 create_portfolio() 工厂函数，用于创建各种类型的投资组合实例。
支持从配置字典、配置对象或直接参数创建投资组合，提供了统一的创建接口。

实现方案概述：
1. 支持多种创建方式：配置驱动、参数驱动、对象驱动
2. 支持多种投资组合类型：目前仅支持 EquityPortfolio，未来可扩展
3. 与配置管理系统无缝集成，支持环境变量覆盖
4. 自动处理依赖注入（风险管理器、头寸规模计算器等）
5. 提供详细的错误处理和验证

主要函数：
- create_portfolio(): 主工厂函数，支持多种输入格式
- _create_from_config(): 从配置字典创建投资组合
- _create_from_params(): 从参数创建投资组合
- _validate_portfolio_config(): 验证投资组合配置

与系统中其他模块的关系：
1. 依赖 equity.py 中的 EquityPortfolio 类
2. 依赖 base.py 中的枚举和数据结构
3. 依赖 rebalancing.py 中的再平衡策略工厂
4. 依赖 risk.py 中的风险管理和头寸规模计算
5. 使用 utils/config.py 中的配置管理功能
6. 使用 utils/exceptions.py 中的异常处理
"""

from typing import Dict, List, Optional, Any, Union
import copy

from ...utils.exceptions import PortfolioError, ConfigurationError, ValidationError
from ...utils.logger import get_logger
from ...utils.config import PortfolioConfig, RebalanceFrequency, AllocationMethod

from .base import Portfolio, RebalancingFrequency, PortfolioStatus
from .equity import EquityPortfolio
from .rebalancing import create_rebalancer
from .risk import RiskManager, PositionSizer

logger = get_logger(__name__)


# 映射配置枚举到内部枚举
_FREQUENCY_MAPPING = {
    'daily': RebalancingFrequency.DAILY,
    'weekly': RebalancingFrequency.WEEKLY,
    'monthly': RebalancingFrequency.MONTHLY,
    'quarterly': RebalancingFrequency.QUARTERLY,
    'yearly': RebalancingFrequency.YEARLY,
    'never': RebalancingFrequency.NEVER
}

# 映射分配方法到再平衡策略
_ALLOCATION_STRATEGY_MAPPING = {
    'equal_weight': 'equal_weight',
    'market_cap': 'equal_weight',  # 市场权重需要额外数据，暂用等权重
    'risk_parity': 'risk_parity',
    'min_variance': 'mean_variance',
    'max_sharpe': 'mean_variance'
}


def create_portfolio(
    config: Optional[Union[Dict[str, Any], PortfolioConfig]] = None,
    portfolio_type: str = 'equity',
    **kwargs
) -> Portfolio:
    """
    创建投资组合实例的工厂函数。

    支持三种创建方式：
    1. 配置字典：从完整的配置字典创建
    2. 配置对象：从 PortfolioConfig 对象创建
    3. 直接参数：通过关键字参数直接指定

    优先级：kwargs > config > 默认值

    Args:
        config: 配置字典或 PortfolioConfig 对象
        portfolio_type: 投资组合类型，目前支持 'equity'
        **kwargs: 直接参数，会覆盖配置中的值

    Returns:
        Portfolio: 投资组合实例

    Raises:
        ConfigurationError: 配置错误或不完整
        PortfolioError: 投资组合创建失败
        ValueError: 参数验证失败

    使用示例：
        # 方式1：从配置字典创建
        config = {
            'initial_capital': 100000,
            'name': '我的投资组合',
            'rebalancing_frequency': 'monthly',
            'rebalancing_strategy': 'equal_weight'
        }
        portfolio = create_portfolio(config)

        # 方式2：直接参数创建
        portfolio = create_portfolio(
            initial_capital=100000,
            name='我的投资组合',
            portfolio_type='equity'
        )

        # 方式3：混合方式（kwargs 覆盖 config）
        portfolio = create_portfolio(config, initial_capital=200000)
    """
    try:
        logger.info(f"开始创建 {portfolio_type} 类型投资组合")

        # 根据 portfolio_type 选择创建函数
        if portfolio_type.lower() == 'equity':
            portfolio = _create_equity_portfolio(config, **kwargs)
        else:
            raise ValueError(f"不支持的投资组合类型: {portfolio_type}")

        logger.info(f"成功创建投资组合 '{portfolio.name}'，初始资金: {portfolio.initial_capital:,.2f}")
        return portfolio

    except (ConfigurationError, PortfolioError, ValueError):
        raise
    except Exception as e:
        raise PortfolioError(
            f"投资组合创建失败: {str(e)}",
            operation="create_portfolio",
            details={
                "portfolio_type": portfolio_type,
                "config_type": type(config).__name__ if config else "None"
            }
        ) from e


def _create_equity_portfolio(
    config: Optional[Union[Dict[str, Any], PortfolioConfig]] = None,
    **kwargs
) -> EquityPortfolio:
    """
    创建股票投资组合实例。

    Args:
        config: 配置字典或 PortfolioConfig 对象
        **kwargs: 直接参数

    Returns:
        EquityPortfolio: 股票投资组合实例
    """
    # 合并配置和参数
    merged_config = _merge_config_and_params(config, kwargs)

    # 提取和验证配置参数
    params = _extract_portfolio_params(merged_config)

    # 创建再平衡策略
    rebalancing_strategy = params.get('rebalancing_strategy', 'equal_weight')

    # 创建风险管理器和头寸规模计算器
    risk_manager = _create_risk_manager(params)
    position_sizer = _create_position_sizer(params)

    # 创建投资组合实例
    portfolio = EquityPortfolio(
        initial_capital=params['initial_capital'],
        name=params['name'],
        currency=params['currency'],
        benchmark_symbol=params.get('benchmark_symbol'),
        rebalancing_frequency=params['rebalancing_frequency'],
        rebalancing_strategy=rebalancing_strategy,
        allow_short=params.get('allow_short', False),
        max_leverage=params.get('max_leverage', 1.0),
        risk_manager=risk_manager,
        position_sizer=position_sizer
    )

    # 设置再平衡阈值
    if 'rebalancing_threshold' in params:
        portfolio.rebalancing_threshold = params['rebalancing_threshold']

    # 如果有初始配置，添加到投资组合
    if 'initial_allocations' in params:
        for allocation in params['initial_allocations']:
            portfolio.add_allocation(**allocation)

    return portfolio


def _merge_config_and_params(
    config: Optional[Union[Dict[str, Any], PortfolioConfig]],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    合并配置和参数。

    合并优先级：params > config > 默认值

    Args:
        config: 配置字典或 PortfolioConfig 对象
        params: 直接参数

    Returns:
        合并后的配置字典
    """
    merged = {}

    # 从配置对象或字典提取配置
    if config is not None:
        if isinstance(config, PortfolioConfig):
            # 从配置对象提取
            config_dict = config.dict(exclude_unset=True)
        else:
            # 从配置字典提取
            config_dict = copy.deepcopy(config)

        # 处理嵌套配置
        if 'portfolio' in config_dict and isinstance(config_dict['portfolio'], dict):
            # 如果配置中有 portfolio 节，提取它
            portfolio_config = config_dict['portfolio']
            merged.update(portfolio_config)
        else:
            # 否则使用整个配置
            merged.update(config_dict)

    # 应用直接参数（覆盖配置）
    merged.update(params)

    return merged


def _extract_portfolio_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从配置字典中提取并验证投资组合参数。

    Args:
        config: 配置字典

    Returns:
        验证后的参数字典

    Raises:
        ConfigurationError: 配置验证失败
        ValidationError: 参数验证失败
    """
    params = {
        'initial_capital': 100000.0,
        'name': 'Equity Portfolio',
        'currency': 'USD',
        'rebalancing_frequency': RebalancingFrequency.MONTHLY,
        'rebalancing_strategy': 'equal_weight',
        'allow_short': False,
        'max_leverage': 1.0,
        'rebalancing_threshold': 0.05
    }

    # 提取参数（带默认值）
    params['initial_capital'] = config.get('initial_capital', params['initial_capital'])
    params['name'] = config.get('name', params['name'])
    params['currency'] = config.get('currency', params['currency'])
    params['allow_short'] = config.get('allow_short', params['allow_short'])
    params['max_leverage'] = config.get('max_leverage', params['max_leverage'])
    params['rebalancing_threshold'] = config.get('rebalancing_threshold', params['rebalancing_threshold'])

    # 处理再平衡频率
    freq_config = config.get('rebalance_frequency', config.get('rebalancing_frequency', 'monthly'))
    if isinstance(freq_config, RebalanceFrequency):
        freq_str = freq_config.value
    else:
        freq_str = str(freq_config)

    params['rebalancing_frequency'] = _FREQUENCY_MAPPING.get(
        freq_str.lower(),
        RebalancingFrequency.MONTHLY
    )

    # 处理再平衡策略
    allocation_method = config.get('allocation_method', 'equal_weight')
    if isinstance(allocation_method, AllocationMethod):
        allocation_str = allocation_method.value
    else:
        allocation_str = str(allocation_method)

    params['rebalancing_strategy'] = _ALLOCATION_STRATEGY_MAPPING.get(
        allocation_str.lower(),
        'equal_weight'
    )

    # 可选参数
    if 'benchmark_symbol' in config:
        params['benchmark_symbol'] = config['benchmark_symbol']

    if 'initial_allocations' in config:
        params['initial_allocations'] = config['initial_allocations']

    # 参数验证
    _validate_portfolio_params(params)

    return params


def _validate_portfolio_params(params: Dict[str, Any]) -> None:
    """
    验证投资组合参数。

    Args:
        params: 参数字典

    Raises:
        ValidationError: 参数验证失败
    """
    # 验证初始资金
    initial_capital = params['initial_capital']
    if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
        raise ValidationError(
            f"初始资金必须为正数，当前值: {initial_capital}",
            field="initial_capital",
            value=initial_capital,
            expected="> 0"
        )

    # 验证名称
    name = params['name']
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(
            "投资组合名称不能为空",
            field="name",
            value=name,
            expected="非空字符串"
        )

    # 验证货币
    currency = params['currency']
    if not isinstance(currency, str) or len(currency) != 3:
        raise ValidationError(
            f"货币代码必须是3个字符，当前值: {currency}",
            field="currency",
            value=currency,
            expected="3字符代码（如USD、CNY）"
        )

    # 验证最大杠杆
    max_leverage = params['max_leverage']
    if not isinstance(max_leverage, (int, float)) or max_leverage < 1.0:
        raise ValidationError(
            f"最大杠杆必须大于等于1.0，当前值: {max_leverage}",
            field="max_leverage",
            value=max_leverage,
            expected=">= 1.0"
        )

    # 验证再平衡阈值
    threshold = params.get('rebalancing_threshold', 0.05)
    if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold >= 1.0:
        raise ValidationError(
            f"再平衡阈值必须在0和1之间，当前值: {threshold}",
            field="rebalancing_threshold",
            value=threshold,
            expected="0 < threshold < 1"
        )


def _create_risk_manager(params: Dict[str, Any]) -> RiskManager:
    """
    创建风险管理器实例。

    Args:
        params: 投资组合参数

    Returns:
        RiskManager: 风险管理器实例
    """
    try:
        risk_manager = RiskManager(
            initial_capital=params['initial_capital'],
            max_position_risk=params.get('max_position_risk', 0.02),
            max_portfolio_risk=params.get('max_portfolio_risk', 0.10),
            max_drawdown=params.get('max_drawdown', 0.20)
        )

        # 添加风险限制（如果配置中提供）
        risk_limits = params.get('risk_limits', [])
        for limit_config in risk_limits:
            # 这里需要根据实际配置创建 RiskLimit 对象
            # 目前使用默认配置
            pass

        return risk_manager

    except Exception as e:
        logger.warning(f"创建风险管理器失败，使用默认配置: {e}")
        return RiskManager(initial_capital=params['initial_capital'])


def _create_position_sizer(params: Dict[str, Any]) -> PositionSizer:
    """
    创建头寸规模计算器实例。

    Args:
        params: 投资组合参数

    Returns:
        PositionSizer: 头寸规模计算器实例
    """
    try:
        position_sizer = PositionSizer(
            account_size=params['initial_capital']
        )
        return position_sizer

    except Exception as e:
        logger.warning(f"创建头寸规模计算器失败，使用默认配置: {e}")
        return PositionSizer(account_size=params['initial_capital'])


def create_portfolio_from_config_file(
    config_file: str = 'backtest_config.yaml',
    config_type: str = 'yaml'
) -> Optional[Portfolio]:
    """
    从配置文件创建投资组合。

    这是一个便捷函数，用于从配置文件直接创建投资组合。
    如果配置中 portfolio.enabled 为 False，则返回 None。

    Args:
        config_file: 配置文件路径
        config_type: 配置类型（yaml 或 json）

    Returns:
        Portfolio: 投资组合实例，如果未启用则返回 None

    Raises:
        ConfigurationError: 配置加载失败
    """
    try:
        from ...utils.config import config_manager

        # 加载配置
        config_data = config_manager.load_config(config_file, config_type)

        # 检查是否启用投资组合
        portfolio_config = config_data.get('portfolio', {})
        if not portfolio_config.get('enabled', False):
            logger.info("投资组合功能未启用")
            return None

        # 合并回测配置中的相关参数
        backtest_config = config_data.get('backtest', {})
        portfolio_config['initial_capital'] = backtest_config.get('initial_capital', 100000.0)
        portfolio_config['currency'] = 'USD'  # 默认货币，可从配置中读取

        # 创建投资组合
        portfolio = create_portfolio(portfolio_config)
        return portfolio

    except Exception as e:
        raise ConfigurationError(
            f"从配置文件创建投资组合失败: {config_file}",
            config_file=config_file,
            details={"error": str(e)}
        ) from e


# 导出
__all__ = [
    'create_portfolio',
    'create_portfolio_from_config_file'
]
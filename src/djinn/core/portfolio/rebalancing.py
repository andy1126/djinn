"""
投资组合再平衡策略模块。

目的：
    提供多种投资组合再平衡策略，用于调整资产权重以达到目标配置。

实现方案：
    - 实现 RebalancingStrategy 抽象基类的具体策略
    - 支持 equal weight（等权重）、risk parity（风险平价）、mean-variance optimization（均值方差优化）、Black-Litterman 等策略
    - 每种策略提供权重优化和交易计算功能

使用方法：
    1. 使用 create_rebalancer() 工厂函数创建策略实例
    2. 调用 optimize_weights() 计算目标权重
    3. 调用 calculate_rebalancing_trades() 生成调仓交易列表
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

from ...utils.exceptions import PortfolioError, ValidationError
from ...utils.logger import get_logger
from .base import RebalancingStrategy

logger = get_logger(__name__)


class EqualWeightRebalancer(RebalancingStrategy):
    """
    等权重再平衡策略。

    目的：
        为投资组合中的每个资产分配相同的权重，实现最简单的分散化。

    实现方案：
        - 计算资产数量的倒数作为每个资产的权重
        - 应用最小/最大权重约束
        - 根据权重差异计算交易数量

    使用方法：
        1. 实例化 EqualWeightRebalancer
        2. 调用 optimize_weights() 获取目标权重
        3. 调用 calculate_rebalancing_trades() 生成交易
    """

    def calculate_rebalancing_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        计算等权重再平衡交易。

        目的：
            根据当前权重和目标权重（等权重）的差异，计算需要执行的交易列表。

        实现方案：
            1. 计算每个资产的目标权重（资产数量的倒数）
            2. 遍历每个资产，计算权重差异
            3. 如果权重差异小于阈值（0.001），跳过该资产
            4. 计算目标价值和当前价值，得出交易价值
            5. 根据价格计算交易数量，应用最小交易规模约束
            6. 确定交易方向（buy/sell）并构建交易字典

        参数：
            current_weights: 当前投资组合权重字典 {资产符号: 权重}
            target_weights: 目标权重字典（等权重策略下与资产数量相关）
            portfolio_value: 投资组合总价值
            prices: 当前资产价格字典 {资产符号: 价格}
            constraints: 交易约束字典，如 min_trade_size（最小交易规模）

        返回：
            交易订单列表，每个订单包含 symbol, quantity, side, current_weight, target_weight, trade_value
        """
        trades = []

        # 计算每个资产的等权重
        num_assets = len(target_weights)
        equal_weight = 1.0 / num_assets if num_assets > 0 else 0

        for symbol in target_weights.keys():
            current_weight = current_weights.get(symbol, 0)
            target_weight = equal_weight

            # 计算权重差异
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < 0.001:  # 小阈值
                continue

            # 计算目标价值和当前价值
            target_value = portfolio_value * target_weight
            current_value = portfolio_value * current_weight
            trade_value = target_value - current_value

            if symbol not in prices or prices[symbol] <= 0:
                logger.warning(f"No valid price for {symbol}, skipping")
                continue

            # 计算交易数量
            quantity = trade_value / prices[symbol]

            # 应用取整约束
            min_trade_size = constraints.get('min_trade_size', 1)
            if abs(quantity) < min_trade_size:
                continue

            quantity = round(quantity)

            # 确定交易方向
            side = 'buy' if quantity > 0 else 'sell'

            trades.append({
                'symbol': symbol,
                'quantity': abs(quantity),
                'side': side,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'trade_value': abs(trade_value)
            })

        return trades

    def optimize_weights(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        计算等权重。

        目的：
            为所有资产分配相同的权重，忽略预期收益和协方差矩阵。

        实现方案：
            1. 从 expected_returns 获取资产列表
            2. 计算每个资产的权重：1 / 资产数量
            3. 应用最小/最大权重约束
            4. 归一化权重确保总和为1

        参数：
            expected_returns: 预期收益字典 {资产符号: 预期收益}（等权重策略中未使用）
            covariance_matrix: 协方差矩阵（等权重策略中未使用）
            constraints: 优化约束字典，包含 min_weight（最小权重）和 max_weight（最大权重）

        返回：
            等权重字典 {资产符号: 权重}
        """
        symbols = list(expected_returns.keys())
        num_assets = len(symbols)

        if num_assets == 0:
            return {}

        equal_weight = 1.0 / num_assets

        weights = {symbol: equal_weight for symbol in symbols}

        # 应用约束（如果有）
        min_weight = constraints.get('min_weight', 0)
        max_weight = constraints.get('max_weight', 1.0)

        for symbol in weights:
            weights[symbol] = max(min_weight, min(max_weight, weights[symbol]))

        # 归一化权重确保总和为1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class RiskParityRebalancer(RebalancingStrategy):
    """
    风险平价再平衡策略。

    目的：
        根据资产风险分配权重，高风险资产权重较低，低风险资产权重较高，实现每个资产对组合风险的贡献相等。

    实现方案：
        - 从协方差矩阵提取波动率作为风险度量
        - 计算逆波动率权重（权重与波动率成反比）
        - 应用权重约束并归一化

    使用方法：
        1. 实例化 RiskParityRebalancer
        2. 调用 optimize_weights() 获取目标权重
        3. 调用 calculate_rebalancing_trades() 生成交易
    """

    def calculate_rebalancing_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        计算风险平价再平衡交易。

        目的：
            根据当前权重和目标权重（风险平价权重）的差异，计算需要执行的交易列表。

        实现方案：
            1. 遍历每个资产，计算权重差异
            2. 如果权重差异小于阈值（0.001），跳过该资产
            3. 计算目标价值和当前价值，得出交易价值
            4. 根据价格计算交易数量，应用最小交易规模约束
            5. 确定交易方向（buy/sell）并构建交易字典

        参数：
            current_weights: 当前投资组合权重字典 {资产符号: 权重}
            target_weights: 目标权重字典（风险平价权重）
            portfolio_value: 投资组合总价值
            prices: 当前资产价格字典 {资产符号: 价格}
            constraints: 交易约束字典，如 min_trade_size（最小交易规模）

        返回：
            交易订单列表，每个订单包含 symbol, quantity, side, current_weight, target_weight, trade_value
        """
        trades = []

        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)

            # 计算权重差异
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < 0.001:  # 小阈值
                continue

            # 计算目标价值和当前价值
            target_value = portfolio_value * target_weight
            current_value = portfolio_value * current_weight
            trade_value = target_value - current_value

            if symbol not in prices or prices[symbol] <= 0:
                logger.warning(f"No valid price for {symbol}, skipping")
                continue

            # 计算交易数量
            quantity = trade_value / prices[symbol]

            # 应用取整约束
            min_trade_size = constraints.get('min_trade_size', 1)
            if abs(quantity) < min_trade_size:
                continue

            quantity = round(quantity)

            # 确定交易方向
            side = 'buy' if quantity > 0 else 'sell'

            trades.append({
                'symbol': symbol,
                'quantity': abs(quantity),
                'side': side,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'trade_value': abs(trade_value)
            })

        return trades

    def optimize_weights(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        计算风险平价权重。

        目的：
            根据资产风险（波动率）分配权重，使每个资产对组合风险的贡献相等。

        实现方案：
            1. 从协方差矩阵提取每个资产的波动率（方差开平方）
            2. 计算逆波动率权重（权重 = 1 / 波动率）
            3. 归一化权重确保总和为1
            4. 应用最小/最大权重约束并重新归一化

        参数：
            expected_returns: 预期收益字典 {资产符号: 预期收益}（风险平价策略中未使用）
            covariance_matrix: 协方差矩阵，用于提取波动率
            constraints: 优化约束字典，包含 min_weight（最小权重）和 max_weight（最大权重）

        返回：
            风险平价权重字典 {资产符号: 权重}
        """
        symbols = list(expected_returns.keys())

        if len(symbols) == 0:
            return {}

        # 从协方差矩阵提取波动率
        volatilities = {}
        for symbol in symbols:
            if symbol in covariance_matrix.columns:
                variance = covariance_matrix.loc[symbol, symbol]
                volatilities[symbol] = np.sqrt(variance)
            else:
                volatilities[symbol] = 0.15  # Default volatility

        # 计算逆波动率权重
        inv_volatilities = {k: 1 / max(v, 0.01) for k, v in volatilities.items()}
        total_inv_vol = sum(inv_volatilities.values())

        if total_inv_vol == 0:
            # 回退到等权重
            equal_weight = 1.0 / len(symbols)
            weights = {symbol: equal_weight for symbol in symbols}
        else:
            weights = {k: v / total_inv_vol for k, v in inv_volatilities.items()}

        # 应用约束
        min_weight = constraints.get('min_weight', 0)
        max_weight = constraints.get('max_weight', 1.0)

        for symbol in weights:
            weights[symbol] = max(min_weight, min(max_weight, weights[symbol]))

        # 归一化权重确保总和为1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class MeanVarianceRebalancer(RebalancingStrategy):
    """
    均值方差优化再平衡策略。

    目的：
        使用马科维茨均值方差优化模型，在给定风险水平下最大化预期收益，找到最优的风险收益权衡。

    实现方案：
        - 使用预期收益和协方差矩阵构建优化问题
        - 通过二次规划（QP）求解最优权重
        - 支持风险厌恶参数调节风险偏好

    使用方法：
        1. 实例化 MeanVarianceRebalancer，可选参数 risk_aversion（风险厌恶系数）
        2. 调用 optimize_weights() 获取目标权重
        3. 调用 calculate_rebalancing_trades() 生成交易
    """

    def __init__(self, risk_aversion: float = 1.0):
        """
        初始化均值方差再平衡器。

        目的：
            设置风险厌恶系数，用于调节优化过程中的风险偏好。

        参数：
            risk_aversion: 风险厌恶参数，值越高表示越厌恶风险
        """
        self.risk_aversion = risk_aversion

    def calculate_rebalancing_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        计算均值方差优化再平衡交易。

        目的：
            根据当前权重和目标权重（均值方差最优权重）的差异，计算需要执行的交易列表。

        实现方案：
            1. 遍历每个资产，计算权重差异
            2. 如果权重差异小于阈值（0.001），跳过该资产
            3. 计算目标价值和当前价值，得出交易价值
            4. 根据价格计算交易数量，应用最小交易规模约束
            5. 确定交易方向（buy/sell）并构建交易字典

        参数：
            current_weights: 当前投资组合权重字典 {资产符号: 权重}
            target_weights: 目标权重字典（均值方差最优权重）
            portfolio_value: 投资组合总价值
            prices: 当前资产价格字典 {资产符号: 价格}
            constraints: 交易约束字典，如 min_trade_size（最小交易规模）

        返回：
            交易订单列表，每个订单包含 symbol, quantity, side, current_weight, target_weight, trade_value
        """
        trades = []

        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)

            # 计算权重差异
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < 0.001:  # 小阈值
                continue

            # 计算目标价值和当前价值
            target_value = portfolio_value * target_weight
            current_value = portfolio_value * current_weight
            trade_value = target_value - current_value

            if symbol not in prices or prices[symbol] <= 0:
                logger.warning(f"No valid price for {symbol}, skipping")
                continue

            # 计算交易数量
            quantity = trade_value / prices[symbol]

            # 应用取整约束
            min_trade_size = constraints.get('min_trade_size', 1)
            if abs(quantity) < min_trade_size:
                continue

            quantity = round(quantity)

            # 确定交易方向
            side = 'buy' if quantity > 0 else 'sell'

            trades.append({
                'symbol': symbol,
                'quantity': abs(quantity),
                'side': side,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'trade_value': abs(trade_value)
            })

        return trades

    def optimize_weights(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        计算均值方差最优权重。

        目的：
            基于马科维茨均值方差优化理论，在给定风险厌恶系数下最大化预期收益。

        实现方案：
            1. 准备预期收益向量和协方差矩阵
            2. 检查协方差矩阵是否可逆，若奇异则回退到等权重
            3. 使用二次规划（CVXOPT）求解最优权重，若无CVXOPT则使用解析解
            4. 应用权重约束并归一化

        参数：
            expected_returns: 预期收益字典 {资产符号: 预期收益}
            covariance_matrix: 协方差矩阵
            constraints: 优化约束字典，包含 min_weight（最小权重）和 max_weight（最大权重）

        返回：
            最优权重字典 {资产符号: 权重}
        """
        symbols = list(expected_returns.keys())

        if len(symbols) == 0:
            return {}

        # 准备优化数据
        mu = np.array([expected_returns[s] for s in symbols])
        Sigma = covariance_matrix.loc[symbols, symbols].values

        # 检查协方差矩阵是否有效
        if np.linalg.matrix_rank(Sigma) < len(symbols):
            logger.warning("Covariance matrix is singular, using equal weights")
            equal_weight = 1.0 / len(symbols)
            return {symbol: equal_weight for symbol in symbols}

        try:
            # 求解均值方差优化
            # Objective: maximize μ'w - (λ/2) * w'Σw
            # 其中 λ 是风险厌恶参数

            # 如果可用，使用二次规划求解器
            try:
                import cvxopt
                from cvxopt import matrix, solvers

                # 设置 QP 问题
                n = len(symbols)

                # Objective: minimize (1/2) * w'Σw - μ'w
                P = matrix(Sigma * self.risk_aversion)
                q = matrix(-mu)

                # Constraints: sum(w) = 1, w >= 0
                A = matrix(1.0, (1, n))
                b = matrix(1.0)

                G = matrix(-np.eye(n))
                h = matrix(0.0, (n, 1))

                # 求解 QP
                solvers.options['show_progress'] = False
                solution = solvers.qp(P, q, G, h, A, b)

                if solution['status'] == 'optimal':
                    weights = np.array(solution['x']).flatten()
                else:
                    raise ValueError("QP solver did not find optimal solution")

            except ImportError:
                # 无约束情况下的解析解回退方案
                logger.warning("CVXOPT not available, using analytical solution")
                Sigma_inv = np.linalg.inv(Sigma)
                ones = np.ones(len(symbols))

                # 最小方差组合权重
                w_mvp = Sigma_inv @ ones / (ones.T @ Sigma_inv @ ones)

                # 切线组合权重
                w_tangency = Sigma_inv @ mu / (ones.T @ Sigma_inv @ mu)

                # 基于风险厌恶系数组合
                weights = w_mvp + (1 / self.risk_aversion) * w_tangency

                # 归一化
                weights = weights / weights.sum()

            # 转换为字典
            weights_dict = {symbol: float(weights[i]) for i, symbol in enumerate(symbols)}

        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            # 回退到等权重
            equal_weight = 1.0 / len(symbols)
            weights_dict = {symbol: equal_weight for symbol in symbols}

        # 应用约束
        min_weight = constraints.get('min_weight', 0)
        max_weight = constraints.get('max_weight', 1.0)

        for symbol in weights_dict:
            weights_dict[symbol] = max(min_weight, min(max_weight, weights_dict[symbol]))

        # 归一化权重确保总和为1
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {k: v / total for k, v in weights_dict.items()}

        return weights_dict


class BlackLittermanRebalancer(RebalancingStrategy):
    """
    Black-Litterman 模型再平衡策略。

    目的：
        结合市场均衡收益和投资者观点，生成更合理的资产预期收益和权重。

    实现方案：
        - 使用市场权重计算隐含均衡收益
        - 融合投资者观点及其置信度
        - 计算后验收益和协方差矩阵
        - 通过均值方差优化得到最终权重

    使用方法：
        1. 实例化 BlackLittermanRebalancer，可选参数 tau（先验不确定性）和 risk_aversion（风险厌恶系数）
        2. 调用 optimize_weights()，可传入市场权重、观点和置信度
        3. 调用 calculate_rebalancing_trades() 生成交易
    """

    def __init__(self, tau: float = 0.05, risk_aversion: float = 3.07):
        """
        初始化 Black-Litterman 再平衡器。

        目的：
            设置先验不确定性参数 tau 和风险厌恶系数。

        参数：
            tau: 先验估计的不确定性，通常取值 0.05-0.1
            risk_aversion: 风险厌恶参数
        """
        self.tau = tau
        self.risk_aversion = risk_aversion

    def calculate_rebalancing_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        计算 Black-Litterman 再平衡交易。

        目的：
            根据当前权重和目标权重（Black-Litterman 最优权重）的差异，计算需要执行的交易列表。

        实现方案：
            1. 遍历每个资产，计算权重差异
            2. 如果权重差异小于阈值（0.001），跳过该资产
            3. 计算目标价值和当前价值，得出交易价值
            4. 根据价格计算交易数量，应用最小交易规模约束
            5. 确定交易方向（buy/sell）并构建交易字典

        参数：
            current_weights: 当前投资组合权重字典 {资产符号: 权重}
            target_weights: 目标权重字典（Black-Litterman 最优权重）
            portfolio_value: 投资组合总价值
            prices: 当前资产价格字典 {资产符号: 价格}
            constraints: 交易约束字典，如 min_trade_size（最小交易规模）

        返回：
            交易订单列表，每个订单包含 symbol, quantity, side, current_weight, target_weight, trade_value
        """
        trades = []

        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)

            # 计算权重差异
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < 0.001:  # 小阈值
                continue

            # 计算目标价值和当前价值
            target_value = portfolio_value * target_weight
            current_value = portfolio_value * current_weight
            trade_value = target_value - current_value

            if symbol not in prices or prices[symbol] <= 0:
                logger.warning(f"No valid price for {symbol}, skipping")
                continue

            # 计算交易数量
            quantity = trade_value / prices[symbol]

            # 应用取整约束
            min_trade_size = constraints.get('min_trade_size', 1)
            if abs(quantity) < min_trade_size:
                continue

            quantity = round(quantity)

            # 确定交易方向
            side = 'buy' if quantity > 0 else 'sell'

            trades.append({
                'symbol': symbol,
                'quantity': abs(quantity),
                'side': side,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'trade_value': abs(trade_value)
            })

        return trades

    def optimize_weights(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any],
        market_weights: Optional[Dict[str, float]] = None,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        计算 Black-Litterman 最优权重。

        目的：
            结合市场均衡收益和投资者观点，生成更合理的资产预期收益，进而计算最优权重。

        实现方案：
            1. 使用市场权重计算隐含均衡收益（Π = δ * Σ * w_mkt）
            2. 如果没有观点，直接返回市场权重
            3. 处理投资者观点，构建选择矩阵 P 和不确定性矩阵 Ω
            4. 计算后验收益：E[R] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} * [(τΣ)^{-1}Π + P'Ω^{-1}Q]
            5. 计算后验协方差：Σ_bl = Σ + [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
            6. 使用均值方差优化计算最优权重

        参数：
            expected_returns: 预期收益字典 {资产符号: 预期收益}
            covariance_matrix: 协方差矩阵
            constraints: 优化约束字典
            market_weights: 市场权重字典（可选，默认等权重）
            views: 投资者观点字典 {资产符号: 预期收益}
            view_confidences: 观点置信度字典 {资产符号: 置信度}

        返回：
            Black-Litterman 最优权重字典 {资产符号: 权重}
        """
        symbols = list(expected_returns.keys())

        if len(symbols) == 0:
            return {}

        # 如果提供了市场权重则使用，否则使用等权重
        if market_weights is None:
            market_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}

        # 准备数据
        n = len(symbols)
        Sigma = covariance_matrix.loc[symbols, symbols].values
        w_mkt = np.array([market_weights.get(symbol, 0) for symbol in symbols])

        # Calculate implied equilibrium returns
        # Π = δ * Σ * w_mkt
        Pi = self.risk_aversion * Sigma @ w_mkt

        # 如果没有提供观点，返回市场均衡权重
        if views is None or len(views) == 0:
            weights_dict = {symbol: float(w_mkt[i]) for i, symbol in enumerate(symbols)}
        else:
            try:
                # 处理观点
                view_assets = list(views.keys())
                view_returns = np.array([views[asset] for asset in view_assets])

                # 创建选择矩阵 P
                P = np.zeros((len(view_assets), n))
                for i, asset in enumerate(view_assets):
                    if asset in symbols:
                        idx = symbols.index(asset)
                        P[i, idx] = 1.0

                # 创建不确定性矩阵 Ω（对角）
                if view_confidences is None:
                    # 默认置信度：使用 tau * P * Σ * P'
                    omega = self.tau * P @ Sigma @ P.T
                    Omega = np.diag(np.diag(omega))
                else:
                    # 使用提供的置信度
                    confidences = np.array([view_confidences.get(asset, 1.0) for asset in view_assets])
                    Omega = np.diag(1.0 / confidences)

                # 计算后验收益
                # E[R] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} * [(τΣ)^{-1}Π + P'Ω^{-1}Q]
                tau_sigma_inv = np.linalg.inv(self.tau * Sigma)
                omega_inv = np.linalg.inv(Omega)

                M = tau_sigma_inv + P.T @ omega_inv @ P
                v = tau_sigma_inv @ Pi + P.T @ omega_inv @ view_returns

                mu_bl = np.linalg.solve(M, v)

                # 计算后验协方差
                # Σ_bl = Σ + [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
                sigma_bl = Sigma + np.linalg.inv(M)

                # 使用均值方差优化计算最优权重
                # w* = (δΣ_bl)^{-1} * μ_bl
                weights = np.linalg.solve(self.risk_aversion * sigma_bl, mu_bl)

                # 归一化权重确保总和为1
                weights = weights / weights.sum()

                weights_dict = {symbol: float(weights[i]) for i, symbol in enumerate(symbols)}

            except Exception as e:
                logger.error(f"Black-Litterman optimization failed: {e}")
                # 回退到市场权重
                weights_dict = {symbol: float(w_mkt[i]) for i, symbol in enumerate(symbols)}

        # 应用约束
        min_weight = constraints.get('min_weight', 0)
        max_weight = constraints.get('max_weight', 1.0)

        for symbol in weights_dict:
            weights_dict[symbol] = max(min_weight, min(max_weight, weights_dict[symbol]))

        # 归一化权重确保总和为1
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {k: v / total for k, v in weights_dict.items()}

        return weights_dict


def create_rebalancer(
    strategy_type: str,
    **kwargs
) -> RebalancingStrategy:
    """
    工厂函数，创建再平衡策略实例。

    目的：
        根据策略类型字符串创建对应的再平衡策略对象。

    实现方案：
        1. 将策略类型转换为小写
        2. 根据策略类型字符串匹配相应的策略类
        3. 传递额外参数给策略构造函数
        4. 返回策略实例

    参数：
        strategy_type: 策略类型字符串，支持 'equal_weight', 'risk_parity', 'mean_variance', 'black_litterman'
        **kwargs: 额外参数，将传递给策略构造函数

    返回：
        RebalancingStrategy 实例

    使用示例：
        rebalancer = create_rebalancer('mean_variance', risk_aversion=2.0)
    """
    strategy_type = strategy_type.lower()

    if strategy_type == 'equal_weight':
        return EqualWeightRebalancer(**kwargs)
    elif strategy_type == 'risk_parity':
        return RiskParityRebalancer(**kwargs)
    elif strategy_type == 'mean_variance':
        return MeanVarianceRebalancer(**kwargs)
    elif strategy_type == 'black_litterman':
        return BlackLittermanRebalancer(**kwargs)
    else:
        raise ValueError(f"Unknown rebalancing strategy: {strategy_type}")
"""
股票回测技术指标模块。

本模块提供交易策略中常用的技术指标计算，所有指标均采用向量化操作以提高性能。

目的：
1. 提供标准化、高性能的技术指标计算工具
2. 支持趋势、动量、波动率、成交量、振荡器等各类技术指标
3. 为交易策略开发提供可靠的技术分析基础

实现方案：
1. 使用pandas和numpy进行向量化计算，确保计算效率
2. 统一的IndicatorResult数据结构封装指标计算结果
3. 静态方法设计，便于独立使用和组合调用
4. 包含数据验证和错误处理机制

使用方法：
1. 使用TechnicalIndicators类的静态方法：sma = TechnicalIndicators.simple_moving_average(prices, window=20)
2. 使用便捷函数：sma = calculate_sma(prices, window=20)
3. 批量计算所有指标：indicators = TechnicalIndicators.calculate_all_indicators(data)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ...utils.exceptions import IndicatorError, ValidationError
from ...utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorType(Enum):
    """
    技术指标类型枚举。

    定义技术指标的分类，便于指标管理和策略组合。

    目的：
    1. 标准化指标分类，便于按类型筛选和使用指标
    2. 提供指标元数据，支持指标特性分析
    3. 为指标可视化和管理提供分类依据

    实现方案：
    1. 继承Enum枚举类，定义五种主要指标类型
    2. TREND：趋势指标（如移动平均线、MACD）
    3. MOMENTUM：动量指标（如RSI、随机指标）
    4. VOLATILITY：波动率指标（如布林带、ATR）
    5. VOLUME：成交量指标（如OBV、VWAP）
    6. OSCILLATOR：振荡器指标（如RSI、随机指标）

    使用方法：
    1. 作为指标元数据的一部分：metadata={"type": IndicatorType.TREND}
    2. 按类型筛选指标：trend_indicators = [k for k,v in indicators.items() if v.metadata["type"] == IndicatorType.TREND]
    """
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"


@dataclass
class IndicatorResult:
    """
    技术指标计算结果容器。

    统一封装技术指标的计算结果，包括指标值、信号和元数据。

    目的：
    1. 标准化指标输出格式，便于结果处理和传递
    2. 将指标值、交易信号和元数据封装在单一对象中
    3. 支持复杂的多输出指标（如MACD、布林带等）

    实现方案：
    1. 使用@dataclass装饰器简化数据类定义
    2. values字段：存储指标计算值，可以是Series或DataFrame
    3. signals字段：可选交易信号Series，基于指标生成的买卖信号
    4. metadata字段：可选元数据字典，包含指标类型、参数等信息

    使用方法：
    1. 从指标方法获取结果：result = TechnicalIndicators.simple_moving_average(prices, window=20)
    2. 访问指标值：sma_values = result.values
    3. 访问信号：signals = result.signals
    4. 访问元数据：metadata = result.metadata
    """
    values: Union[pd.Series, pd.DataFrame]
    signals: Optional[pd.Series] = None
    metadata: Optional[Dict[str, Any]] = None


class TechnicalIndicators:
    """
    技术指标计算工具类。

    包含交易策略中常用的各种技术指标计算方法，所有方法均为静态方法，可独立使用。

    目的：
    1. 提供完整的技术指标计算库，覆盖趋势、动量、波动率、成交量等各类指标
    2. 采用向量化计算确保高性能，适合大规模历史数据回测
    3. 统一的接口设计，便于集成到各种交易策略中

    实现方案：
    1. 所有方法均为静态方法，无需实例化即可使用
    2. 使用pandas的rolling、ewm等函数进行向量化计算
    3. 每个方法返回IndicatorResult对象，包含指标值、信号和元数据
    4. 包含数据验证机制，确保输入数据满足计算要求

    使用方法：
    1. 直接调用静态方法：sma = TechnicalIndicators.simple_moving_average(prices, window=20)
    2. 批量计算：indicators = TechnicalIndicators.calculate_all_indicators(data)
    3. 访问结果：sma_values = sma.values, sma_signals = sma.signals
    """

    @staticmethod
    def simple_moving_average(
        prices: pd.Series,
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> IndicatorResult:
        """
        计算简单移动平均线（SMA）。

        目的：
        1. 平滑价格序列，识别价格趋势方向
        2. 作为基础技术指标，用于趋势判断和支撑阻力位识别
        3. 为其他指标（如MACD、布林带）提供计算基础

        参数：
            prices: 价格序列，通常为收盘价
            window: 滚动窗口大小，决定移动平均的周期
            min_periods: 最小观测值数量，未达到时返回NaN

        返回：
            IndicatorResult: 包含SMA值、元数据的IndicatorResult对象

        实现方案：
        1. 使用pandas的rolling().mean()进行向量化计算
        2. 验证数据长度是否满足窗口要求
        3. 设置默认min_periods=window，确保初始阶段有足够数据
        4. 在元数据中记录指标类型和参数

        使用方法：
        1. 计算20日SMA：sma_result = TechnicalIndicators.simple_moving_average(close_prices, window=20)
        2. 获取SMA值：sma_values = sma_result.values
        3. 判断趋势：current_price > sma_values.iloc[-1] 表示价格在均线之上
        """
        if len(prices) < window:
            raise IndicatorError(
                f"Insufficient data for SMA with window {window}. "
                f"Got {len(prices)} observations."
            )

        if min_periods is None:
            min_periods = window

        sma = prices.rolling(window=window, min_periods=min_periods).mean()

        return IndicatorResult(
            values=sma,
            metadata={
                "type": IndicatorType.TREND,
                "window": window,
                "min_periods": min_periods
            }
        )

    @staticmethod
    def exponential_moving_average(
        prices: pd.Series,
        span: int = 20,
        adjust: bool = True
    ) -> IndicatorResult:
        """
        计算指数移动平均线（EMA）。

        目的：
        1. 对近期价格给予更高权重，更快反应价格变化
        2. 识别短期趋势，常用于快速响应交易策略
        3. 作为MACD等指标的计算基础

        参数：
            prices: 价格序列，通常为收盘价
            span: EMA的时间跨度，决定衰减因子
            adjust: 是否使用pandas的adjust=True参数，修正初始值偏差

        返回：
            IndicatorResult: 包含EMA值、元数据的IndicatorResult对象

        实现方案：
        1. 使用pandas的ewm(span=span, adjust=adjust).mean()计算
        2. 验证数据长度是否满足跨度要求
        3. 在元数据中记录指标类型和参数

        使用方法：
        1. 计算12日EMA：ema_result = TechnicalIndicators.exponential_moving_average(close_prices, span=12)
        2. 获取EMA值：ema_values = ema_result.values
        3. 与SMA比较判断趋势强度：ema上升速度快于SMA表示强势上涨
        """
        if len(prices) < span:
            raise IndicatorError(
                f"Insufficient data for EMA with span {span}. "
                f"Got {len(prices)} observations."
            )

        ema = prices.ewm(span=span, adjust=adjust).mean()

        return IndicatorResult(
            values=ema,
            metadata={
                "type": IndicatorType.TREND,
                "span": span,
                "adjust": adjust
            }
        )

    @staticmethod
    def moving_average_convergence_divergence(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> IndicatorResult:
        """
        计算移动平均收敛发散指标（MACD）。

        目的：
        1. 识别价格趋势的变化和动量强度
        2. 通过快慢EMA的差值（MACD线）和信号线交叉产生交易信号
        3.  histogram显示MACD线与信号线的差值，反映趋势加速或减速

        参数：
            prices: 价格序列，通常为收盘价
            fast_period: 快速EMA周期，默认12
            slow_period: 慢速EMA周期，默认26
            signal_period: 信号线EMA周期，默认9

        返回：
            IndicatorResult: 包含MACD线、信号线、histogram的DataFrame，以及交易信号

        实现方案：
        1. 计算快速EMA和慢速EMA，差值得到MACD线
        2. 对MACD线计算EMA得到信号线
        3. MACD线与信号线差值得到histogram
        4. 生成信号：MACD线上穿信号线（金叉）为看涨，下穿（死叉）为看跌
        5. 验证数据长度满足最大周期要求

        使用方法：
        1. 计算MACD：macd_result = TechnicalIndicators.moving_average_convergence_divergence(close_prices)
        2. 获取各分量：macd_df = macd_result.values, signals = macd_result.signals
        3. 判断信号：macd_df['macd'] > macd_df['signal'] 表示看涨
        4. histogram为正表示趋势加速，为负表示趋势减速
        """
        if len(prices) < max(fast_period, slow_period, signal_period):
            raise IndicatorError(
                f"Insufficient data for MACD. "
                f"Got {len(prices)} observations, need at least "
                f"{max(fast_period, slow_period, signal_period)}."
            )

        # Calculate EMAs
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

        # MACD line
        macd_line = fast_ema - slow_ema

        # Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        # Create DataFrame with all components
        macd_df = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

        # Generate signals based on MACD crossovers
        signals = pd.Series(0, index=prices.index)
        signals[macd_line > signal_line] = 1  # Bullish
        signals[macd_line < signal_line] = -1  # Bearish

        return IndicatorResult(
            values=macd_df,
            signals=signals,
            metadata={
                "type": IndicatorType.MOMENTUM,
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period
            }
        )

    @staticmethod
    def relative_strength_index(
        prices: pd.Series,
        period: int = 14
    ) -> IndicatorResult:
        """
        计算相对强弱指数（RSI）。

        目的：
        1. 衡量价格变动速度和幅度，识别超买超卖状态
        2. 评估资产近期表现强度，判断趋势动量
        3. 通过超买（>70）超卖（<30）区域产生交易信号

        参数：
            prices: 价格序列，通常为收盘价
            period: RSI计算周期，默认14

        返回：
            IndicatorResult: 包含RSI值、交易信号和元数据的IndicatorResult对象

        实现方案：
        1. 计算价格变化，分离上涨和下跌幅度
        2. 计算平均上涨幅度和平均下跌幅度
        3. 计算相对强度RS = 平均上涨 / 平均下跌
        4. 计算RSI = 100 - (100 / (1 + RS))
        5. 生成信号：RSI<30超卖（看涨），RSI>70超买（看跌）
        6. 验证数据长度满足period+1要求

        使用方法：
        1. 计算14日RSI：rsi_result = TechnicalIndicators.relative_strength_index(close_prices, period=14)
        2. 获取RSI值：rsi_values = rsi_result.values
        3. 判断超买超卖：rsi_values.iloc[-1] > 70 表示超买，<30 表示超卖
        4. 背离分析：价格创新高但RSI未创新高可能预示趋势反转
        """
        if len(prices) < period + 1:
            raise IndicatorError(
                f"Insufficient data for RSI with period {period}. "
                f"Got {len(prices)} observations, need at least {period + 1}."
            )

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals = pd.Series(0, index=prices.index)
        signals[rsi < 30] = 1  # Oversold, bullish signal
        signals[rsi > 70] = -1  # Overbought, bearish signal

        return IndicatorResult(
            values=rsi,
            signals=signals,
            metadata={
                "type": IndicatorType.OSCILLATOR,
                "period": period,
                "oversold_threshold": 30,
                "overbought_threshold": 70
            }
        )

    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> IndicatorResult:
        """
        计算布林带（Bollinger Bands）。

        目的：
        1. 衡量价格波动性和识别超买超卖水平
        2. 通过上下轨提供动态支撑阻力位
        3. 识别波动率收缩（带宽收窄）和扩张（带宽扩大）

        参数：
            prices: 价格序列，通常为收盘价
            window: 滚动窗口大小，默认20
            num_std: 标准差倍数，决定上下轨宽度，默认2.0

        返回：
            IndicatorResult: 包含中轨、上轨、下轨、%B指标、带宽的DataFrame，以及交易信号

        实现方案：
        1. 中轨 = 价格的SMA（简单移动平均）
        2. 标准差 = 价格的滚动标准差
        3. 上轨 = 中轨 + num_std * 标准差
        4. 下轨 = 中轨 - num_std * 标准差
        5. %B指标 = (价格 - 下轨) / (上轨 - 下轨)，衡量价格在布林带中的位置
        6. 带宽 = (上轨 - 下轨) / 中轨，衡量波动率
        7. 生成信号：价格低于下轨（超卖看涨），价格高于上轨（超买看跌）

        使用方法：
        1. 计算布林带：bb_result = TechnicalIndicators.bollinger_bands(close_prices, window=20, num_std=2.0)
        2. 获取各分量：bb_df = bb_result.values, signals = bb_result.signals
        3. 判断价格位置：bb_df['percent_b'].iloc[-1] > 1 表示价格在上轨之上，<0 表示在下轨之下
        4. 观察带宽变化：带宽收窄可能预示波动率突破
        """
        if len(prices) < window:
            raise IndicatorError(
                f"Insufficient data for Bollinger Bands with window {window}. "
                f"Got {len(prices)} observations."
            )

        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=window).mean()

        # Calculate standard deviation
        std = prices.rolling(window=window).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        # Calculate %B indicator
        percent_b = (prices - lower_band) / (upper_band - lower_band)

        # Calculate bandwidth
        bandwidth = (upper_band - lower_band) / middle_band

        # Generate signals
        signals = pd.Series(0, index=prices.index)
        signals[prices < lower_band] = 1  # Price below lower band, bullish
        signals[prices > upper_band] = -1  # Price above upper band, bearish

        # Create DataFrame with all components
        bands_df = pd.DataFrame({
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band,
            'percent_b': percent_b,
            'bandwidth': bandwidth
        })

        return IndicatorResult(
            values=bands_df,
            signals=signals,
            metadata={
                "type": IndicatorType.VOLATILITY,
                "window": window,
                "num_std": num_std
            }
        )

    @staticmethod
    def average_true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> IndicatorResult:
        """
        计算平均真实波幅（ATR）。

        目的：
        1. 衡量价格波动性，反映市场波动程度
        2. 为止损和仓位大小计算提供波动率参考
        3. 识别波动率变化，判断市场情绪

        参数：
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: ATR计算周期，默认14

        返回：
            IndicatorResult: 包含ATR值和元数据的IndicatorResult对象

        实现方案：
        1. 计算真实波幅（TR）：取以下三者最大值
           - 当日最高价 - 当日最低价
           - 当日最高价 - 前日收盘价的绝对值
           - 当日最低价 - 前日收盘价的绝对值
        2. 对真实波幅计算简单移动平均得到ATR
        3. 验证各价格序列长度满足周期要求

        使用方法：
        1. 计算14日ATR：atr_result = TechnicalIndicators.average_true_range(high_prices, low_prices, close_prices, period=14)
        2. 获取ATR值：atr_values = atr_result.values
        3. 设置动态止损：stop_loss = current_price - 2 * atr_values.iloc[-1]
        4. 波动率分析：ATR上升表示波动加剧，下降表示市场平静
        """
        if len(high) < period or len(low) < period or len(close) < period:
            raise IndicatorError(
                f"Insufficient data for ATR with period {period}. "
                f"Got {len(high)} observations."
            )

        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        atr = true_range.rolling(window=period).mean()

        return IndicatorResult(
            values=atr,
            metadata={
                "type": IndicatorType.VOLATILITY,
                "period": period
            }
        )

    @staticmethod
    def on_balance_volume(
        close: pd.Series,
        volume: pd.Series
    ) -> IndicatorResult:
        """
        计算能量潮指标（OBV）。

        目的：
        1. 将成交量与价格变化结合，识别资金流向
        2. 验证价格趋势的成交量支撑，判断趋势强度
        3. 通过OBV与价格背离预测趋势反转

        参数：
            close: 收盘价序列
            volume: 成交量序列

        返回：
            IndicatorResult: 包含OBV值、交易信号和元数据的IndicatorResult对象

        实现方案：
        1. 计算价格变化：当日收盘价 - 前日收盘价
        2. OBV计算规则：
           - 价格上涨：OBV增加当日成交量
           - 价格下跌：OBV减少当日成交量
           - 价格不变：OBV不变
        3. 计算OBV的累积和
        4. 生成信号：OBV上升为看涨，下降为看跌
        5. 验证收盘价和成交量序列长度一致

        使用方法：
        1. 计算OBV：obv_result = TechnicalIndicators.on_balance_volume(close_prices, volume_series)
        2. 获取OBV值：obv_values = obv_result.values
        3. 分析背离：价格创新高但OBV未创新高，可能预示上涨乏力
        4. 趋势确认：价格上涨伴随OBV上升，表示趋势有成交量支持
        """
        if len(close) != len(volume):
            raise IndicatorError(
                f"Close and volume series must have same length. "
                f"Got close: {len(close)}, volume: {len(volume)}."
            )

        # Calculate price change
        price_change = close.diff()

        # Initialize OBV
        obv = pd.Series(0, index=close.index)

        # Calculate OBV
        obv[price_change > 0] = volume[price_change > 0]
        obv[price_change < 0] = -volume[price_change < 0]
        obv[price_change == 0] = 0

        # Cumulative sum
        obv = obv.cumsum()

        # Generate signals based on OBV trend
        signals = pd.Series(0, index=close.index)

        # Simple signal: OBV increasing = bullish, decreasing = bearish
        obv_change = obv.diff()
        signals[obv_change > 0] = 1
        signals[obv_change < 0] = -1

        return IndicatorResult(
            values=obv,
            signals=signals,
            metadata={
                "type": IndicatorType.VOLUME
            }
        )

    @staticmethod
    def stochastic_oscillator(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> IndicatorResult:
        """
        计算随机指标（Stochastic Oscillator）。

        目的：
        1. 识别超买超卖状态，衡量收盘价在近期价格区间中的位置
        2. 通过%K和%D线的交叉产生交易信号
        3. 捕捉短期价格动量变化

        参数：
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            k_period: %K计算周期，默认14
            d_period: %D计算周期（平滑周期），默认3
            smooth_k: %K平滑因子，默认3

        返回：
            IndicatorResult: 包含%K快线、%D慢线的DataFrame，以及交易信号

        实现方案：
        1. 计算%K原始值：100 * ((收盘价 - 最近N期最低价) / (最近N期最高价 - 最近N期最低价))
        2. 对%K进行平滑得到%K快线
        3. 对%K快线进行移动平均得到%D慢线（信号线）
        4. 生成信号：%K从下方上穿%D且处于超卖区（<20）为看涨，%K从上方下穿%D且处于超买区（>80）为看跌
        5. 验证数据长度满足周期要求

        使用方法：
        1. 计算随机指标：stoch_result = TechnicalIndicators.stochastic_oscillator(high_prices, low_prices, close_prices)
        2. 获取指标值：stoch_df = stoch_result.values, signals = stoch_result.signals
        3. 判断超买超卖：stoch_df['k_fast'].iloc[-1] > 80 超买，<20 超卖
        4. 观察金叉死叉：%K上穿%D为金叉（看涨），下穿为死叉（看跌）
        """
        if len(high) < k_period or len(low) < k_period or len(close) < k_period:
            raise IndicatorError(
                f"Insufficient data for Stochastic Oscillator. "
                f"Got {len(high)} observations, need at least {k_period}."
            )

        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_raw = 100 * ((close - lowest_low) / (highest_high - lowest_low))

        # Smooth %K if requested
        if smooth_k > 1:
            k_fast = k_raw.rolling(window=smooth_k).mean()
        else:
            k_fast = k_raw

        # Calculate %D (signal line)
        d_slow = k_fast.rolling(window=d_period).mean()

        # Create DataFrame
        stoch_df = pd.DataFrame({
            'k_fast': k_fast,
            'd_slow': d_slow
        })

        # Generate signals
        signals = pd.Series(0, index=close.index)

        # Bullish: %K crosses above %D from oversold (<20)
        bullish_cross = (k_fast.shift() < d_slow.shift()) & (k_fast > d_slow) & (k_fast < 20)
        signals[bullish_cross] = 1

        # Bearish: %K crosses below %D from overbought (>80)
        bearish_cross = (k_fast.shift() > d_slow.shift()) & (k_fast < d_slow) & (k_fast > 80)
        signals[bearish_cross] = -1

        return IndicatorResult(
            values=stoch_df,
            signals=signals,
            metadata={
                "type": IndicatorType.OSCILLATOR,
                "k_period": k_period,
                "d_period": d_period,
                "smooth_k": smooth_k,
                "oversold_threshold": 20,
                "overbought_threshold": 80
            }
        )

    @staticmethod
    def volume_weighted_average_price(
        close: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> IndicatorResult:
        """
        计算成交量加权平均价格（VWAP）。

        目的：
        1. 衡量在特定成交量下的平均交易价格
        2. 识别机构资金的平均成本，作为支撑阻力参考
        3. 比较当前价格与VWAP的相对位置判断买卖压力

        参数：
            close: 收盘价序列（由于缺乏日内数据，使用收盘价代替典型价格）
            volume: 成交量序列
            window: 滚动窗口大小，默认20

        返回：
            IndicatorResult: 包含VWAP值、交易信号和元数据的IndicatorResult对象

        实现方案：
        1. 计算典型价格（由于只有收盘价，使用收盘价作为典型价格）
        2. 计算滚动窗口内的累计（典型价格 * 成交量）
        3. 计算滚动窗口内的累计成交量
        4. VWAP = 累计（价格*成交量） / 累计成交量
        5. 生成信号：价格高于VWAP看涨，低于VWAP看跌
        6. 验证序列长度一致且满足窗口要求

        使用方法：
        1. 计算VWAP：vwap_result = TechnicalIndicators.volume_weighted_average_price(close_prices, volume_series, window=20)
        2. 获取VWAP值：vwap_values = vwap_result.values
        3. 判断价格位置：current_price > vwap_values.iloc[-1] 表示价格在VWAP之上，买方占优
        4. 作为动态支撑阻力：价格回调至VWAP可能获得支撑，反弹至VWAP可能遇到阻力
        """
        if len(close) != len(volume):
            raise IndicatorError(
                f"Close and volume series must have same length. "
                f"Got close: {len(close)}, volume: {len(volume)}."
            )

        if len(close) < window:
            raise IndicatorError(
                f"Insufficient data for VWAP with window {window}. "
                f"Got {len(close)} observations."
            )

        # Calculate typical price (average of high, low, close)
        # Since we only have close, we'll use close as typical price
        typical_price = close

        # Calculate VWAP
        cumulative_tp_volume = (typical_price * volume).rolling(window=window).sum()
        cumulative_volume = volume.rolling(window=window).sum()
        vwap = cumulative_tp_volume / cumulative_volume

        # Generate signals based on price vs VWAP
        signals = pd.Series(0, index=close.index)
        signals[close > vwap] = 1  # Price above VWAP, bullish
        signals[close < vwap] = -1  # Price below VWAP, bearish

        return IndicatorResult(
            values=vwap,
            signals=signals,
            metadata={
                "type": IndicatorType.VOLUME,
                "window": window
            }
        )

    @staticmethod
    def ichimoku_cloud(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        conversion_period: int = 9,
        base_period: int = 26,
        leading_span_b_period: int = 52,
        displacement: int = 26
    ) -> IndicatorResult:
        """
        计算一目均衡表（Ichimoku Cloud）指标。

        目的：
        1. 提供全面的趋势分析，包含趋势方向、支撑阻力、动量等多个维度
        2. 通过云层（Kumo）识别关键支撑阻力区域
        3. 多时间框架分析，捕捉短期、中期、长期趋势

        参数：
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            conversion_period: 转换线（Tenkan-sen）周期，默认9
            base_period: 基准线（Kijun-sen）周期，默认26
            leading_span_b_period: 先行带B（Senkou Span B）周期，默认52
            displacement: 延迟跨度（Chikou Span）位移，默认26

        返回：
            IndicatorResult: 包含转换线、基准线、先行带A、先行带B、延迟跨度的DataFrame，以及交易信号

        实现方案：
        1. 转换线 = (最近N期最高价 + 最近N期最低价) / 2
        2. 基准线 = (最近M期最高价 + 最近M期最低价) / 2
        3. 先行带A = (转换线 + 基准线) / 2，位移到未来
        4. 先行带B = (最近L期最高价 + 最近L期最低价) / 2，位移到未来
        5. 延迟跨度 = 收盘价，位移到过去
        6. 云层 = 先行带A和先行带B之间的区域
        7. 生成信号：价格在云层之上且转换线>基准线为看涨，价格在云层之下且转换线<基准线为看跌

        使用方法：
        1. 计算一目均衡表：ichimoku_result = TechnicalIndicators.ichimoku_cloud(high_prices, low_prices, close_prices)
        2. 获取各分量：ichimoku_df = ichimoku_result.values, signals = ichimoku_result.signals
        3. 判断趋势：价格在云层之上为上升趋势，之下为下降趋势
        4. 观察交叉：转换线上穿基准线为金叉（看涨），下穿为死叉（看跌）
        5. 云层厚度：云层厚表示强支撑阻力，薄表示可能突破
        """
        periods = [conversion_period, base_period, leading_span_b_period]
        max_period = max(periods)

        if len(high) < max_period or len(low) < max_period or len(close) < max_period:
            raise IndicatorError(
                f"Insufficient data for Ichimoku Cloud. "
                f"Got {len(high)} observations, need at least {max_period}."
            )

        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=conversion_period).max() +
                     low.rolling(window=conversion_period).min()) / 2

        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=base_period).max() +
                    low.rolling(window=base_period).min()) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=leading_span_b_period).max() +
                         low.rolling(window=leading_span_b_period).min()) / 2).shift(displacement)

        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)

        # Create DataFrame
        ichimoku_df = pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })

        # Generate signals
        signals = pd.Series(0, index=close.index)

        # Bullish: Price above cloud, Tenkan-sen above Kijun-sen
        bullish = (close > senkou_span_a) & (close > senkou_span_b) & (tenkan_sen > kijun_sen)
        signals[bullish] = 1

        # Bearish: Price below cloud, Tenkan-sen below Kijun-sen
        bearish = (close < senkou_span_a) & (close < senkou_span_b) & (tenkan_sen < kijun_sen)
        signals[bearish] = -1

        return IndicatorResult(
            values=ichimoku_df,
            signals=signals,
            metadata={
                "type": IndicatorType.TREND,
                "conversion_period": conversion_period,
                "base_period": base_period,
                "leading_span_b_period": leading_span_b_period,
                "displacement": displacement
            }
        )

    @staticmethod
    def calculate_all_indicators(
        data: pd.DataFrame,
        close_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        volume_col: Optional[str] = None
    ) -> Dict[str, IndicatorResult]:
        """
        批量计算所有可用的技术指标。

        目的：
        1. 一次性计算多种技术指标，提高效率
        2. 为策略开发提供全面的技术分析数据
        3. 便于比较不同指标的表现和信号一致性

        参数：
            data: 包含OHLCV数据的DataFrame
            close_col: 收盘价列名，默认'close'
            high_col: 最高价列名，默认'high'
            low_col: 最低价列名，默认'low'
            volume_col: 成交量列名，可选

        返回：
            Dict[str, IndicatorResult]: 字典，键为指标名称，值为IndicatorResult对象

        实现方案：
        1. 从DataFrame中提取收盘价、最高价、最低价序列
        2. 计算不依赖成交量的基础指标：SMA、EMA、MACD、RSI、布林带、ATR、随机指标、一目均衡表
        3. 如果提供成交量列，额外计算OBV和VWAP
        4. 使用预定义的参数组合计算各指标（如SMA_20、SMA_50、EMA_20等）

        使用方法：
        1. 批量计算指标：indicators = TechnicalIndicators.calculate_all_indicators(ohlcv_df)
        2. 访问特定指标：sma_20_result = indicators['sma_20']
        3. 分析多指标信号：bullish_count = sum(1 for indicator in indicators.values() if indicator.signals.iloc[-1] == 1)
        4. 可视化多指标：fig, axes = plt.subplots(len(indicators), 1, figsize=(12, 4*len(indicators)))
        """
        results = {}

        # Extract series
        close = data[close_col]
        high = data[high_col]
        low = data[low_col]

        # Calculate indicators that don't require volume
        results['sma_20'] = TechnicalIndicators.simple_moving_average(close, window=20)
        results['sma_50'] = TechnicalIndicators.simple_moving_average(close, window=50)
        results['ema_20'] = TechnicalIndicators.exponential_moving_average(close, span=20)
        results['macd'] = TechnicalIndicators.moving_average_convergence_divergence(close)
        results['rsi'] = TechnicalIndicators.relative_strength_index(close)
        results['bollinger'] = TechnicalIndicators.bollinger_bands(close)
        results['atr'] = TechnicalIndicators.average_true_range(high, low, close)
        results['stochastic'] = TechnicalIndicators.stochastic_oscillator(high, low, close)
        results['ichimoku'] = TechnicalIndicators.ichimoku_cloud(high, low, close)

        # Calculate volume-based indicators if volume column is provided
        if volume_col is not None and volume_col in data.columns:
            volume = data[volume_col]
            results['obv'] = TechnicalIndicators.on_balance_volume(close, volume)
            results['vwap'] = TechnicalIndicators.volume_weighted_average_price(close, volume)

        return results


# Convenience functions for common indicator calculations
def calculate_sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    SMA计算便捷函数。

    目的：
    1. 提供简单移动平均线（SMA）的简化调用接口
    2. 直接返回SMA值Series，无需处理IndicatorResult对象

    参数：
        prices: 价格序列，通常为收盘价
        window: 滚动窗口大小，默认20

    返回：
        pd.Series: SMA值序列

    实现方案：
        调用TechnicalIndicators.simple_moving_average并提取values字段

    使用方法：
        1. 计算SMA：sma_values = calculate_sma(close_prices, window=20)
        2. 直接用于分析或绘图
    """
    return TechnicalIndicators.simple_moving_average(prices, window).values


def calculate_ema(prices: pd.Series, span: int = 20) -> pd.Series:
    """
    EMA计算便捷函数。

    目的：
    1. 提供指数移动平均线（EMA）的简化调用接口
    2. 直接返回EMA值Series，无需处理IndicatorResult对象

    参数：
        prices: 价格序列，通常为收盘价
        span: EMA的时间跨度，默认20

    返回：
        pd.Series: EMA值序列

    实现方案：
        调用TechnicalIndicators.exponential_moving_average并提取values字段

    使用方法：
        1. 计算EMA：ema_values = calculate_ema(close_prices, span=20)
        2. 直接用于分析或绘图
    """
    return TechnicalIndicators.exponential_moving_average(prices, span).values


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI计算便捷函数。

    目的：
    1. 提供相对强弱指数（RSI）的简化调用接口
    2. 直接返回RSI值Series，无需处理IndicatorResult对象

    参数：
        prices: 价格序列，通常为收盘价
        period: RSI计算周期，默认14

    返回：
        pd.Series: RSI值序列

    实现方案：
        调用TechnicalIndicators.relative_strength_index并提取values字段

    使用方法：
        1. 计算RSI：rsi_values = calculate_rsi(close_prices, period=14)
        2. 直接用于分析或绘图
    """
    return TechnicalIndicators.relative_strength_index(prices, period).values


def calculate_macd(prices: pd.Series) -> pd.DataFrame:
    """
    MACD计算便捷函数。

    目的：
    1. 提供移动平均收敛发散指标（MACD）的简化调用接口
    2. 直接返回MACD各分量DataFrame，无需处理IndicatorResult对象

    参数：
        prices: 价格序列，通常为收盘价

    返回：
        pd.DataFrame: 包含MACD线、信号线、histogram的DataFrame

    实现方案：
        调用TechnicalIndicators.moving_average_convergence_divergence并提取values字段

    使用方法：
        1. 计算MACD：macd_df = calculate_macd(close_prices)
        2. 直接用于分析或绘图
    """
    return TechnicalIndicators.moving_average_convergence_divergence(prices).values
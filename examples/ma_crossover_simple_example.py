#!/usr/bin/env python3
"""
双均线策略回测示例 - 使用 SimpleStrategy

本示例演示如何:
1. 使用 Yahoo Finance 加载市场数据（带本地缓存）
2. 使用 SimpleStrategy 框架定义双均线交叉策略（约15行代码）
3. 运行事件驱动回测
4. 分析回测结果

与原 MovingAverageCrossover 策略（约500行）相比，
使用 SimpleStrategy 只需约15行代码即可定义相同策略。
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib

# 添加父目录到路径以导入 djinn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from djinn import SimpleStrategy, param
from djinn.core.backtest import EventDrivenBacktestEngine
from djinn.data.providers.yahoo_finance import YahooFinanceProvider
from src.djinn.data.market_data import AdjustmentType
from src.djinn.utils.logger import setup_logger

# 设置日志
setup_logger(level="INFO")

# 缓存配置（与 basic_backtest.py 保持一致）
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')
CACHE_EXPIRY_DAYS = 1  # 缓存1天后过期


def ensure_cache_dir():
    """确保缓存目录存在。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def get_cache_key(symbol, start_date, end_date, interval="1d", adjustment="adj"):
    """
    为数据请求生成唯一的缓存键。

    Args:
        symbol: 股票代码
        start_date: 开始日期（datetime 或 string）
        end_date: 结束日期（datetime 或 string）
        interval: 数据间隔
        adjustment: 价格调整类型

    Returns:
        str: 唯一缓存键
    """
    # 转换日期为字符串
    if isinstance(start_date, datetime):
        start_str = start_date.strftime("%Y-%m-%d")
    else:
        start_str = str(start_date)

    if isinstance(end_date, datetime):
        end_str = end_date.strftime("%Y-%m-%d")
    else:
        end_str = str(end_date)

    # 创建参数字符串
    key_str = f"{symbol}_{start_str}_{end_str}_{interval}_{adjustment}"

    # 生成 SHA256 哈希作为文件名
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def get_cache_filepath(symbol, start_date, end_date, interval="1d", adjustment="adj"):
    """
    获取给定参数的缓存文件路径。

    Returns:
        str: 缓存文件的完整路径
    """
    ensure_cache_dir()
    cache_key = get_cache_key(symbol, start_date, end_date, interval, adjustment)
    filename = f"{symbol}_{cache_key}.csv"
    return os.path.join(CACHE_DIR, filename)


def is_cache_valid(filepath, max_age_days=CACHE_EXPIRY_DAYS):
    """
    检查缓存文件是否存在且未过期。

    Args:
        filepath: 缓存文件路径
        max_age_days: 缓存最大有效期（天）

    Returns:
        bool: 缓存有效返回 True，否则返回 False
    """
    if not os.path.exists(filepath):
        return False

    # 检查文件年龄
    file_mtime = os.path.getmtime(filepath)
    file_age = (time.time() - file_mtime) / (60 * 60 * 24)  # 转换为天

    return file_age <= max_age_days


def load_from_cache(filepath):
    """
    从缓存文件加载数据。

    Args:
        filepath: 缓存文件路径

    Returns:
        pandas.DataFrame or None: 加载的数据，失败返回 None
    """
    try:
        # 加载 CSV 文件，解析日期并使用第一列作为索引
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"  从缓存加载: {filepath}")
        return data
    except Exception as e:
        print(f"  警告: 无法加载缓存 {filepath}: {e}")
        return None


def save_to_cache(filepath, data):
    """
    保存数据到缓存文件。

    Args:
        filepath: 缓存文件路径
        data: 要保存的 DataFrame
    """
    try:
        # 保存 DataFrame 到 CSV 文件（包含索引）
        data.to_csv(filepath, index=True)
        print(f"  保存到缓存: {filepath}")
    except Exception as e:
        print(f"  警告: 无法保存缓存 {filepath}: {e}")


def get_symbol_data_with_cache(symbol, start_date, end_date, interval="1d", adjustment="adj"):
    """
    获取带缓存的市场数据。

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        interval: 数据间隔
        adjustment: 价格调整类型

    Returns:
        pandas.DataFrame: 该股票的市场数据
    """
    print(f"  正在获取 {symbol} 的数据...")

    # 获取缓存文件路径
    cache_file = get_cache_filepath(symbol, start_date, end_date, interval, adjustment)

    # 首先尝试从缓存加载
    if is_cache_valid(cache_file):
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            return cached_data

    print(f"    缓存不可用或已过期，从 Yahoo Finance 下载...")

    # 创建 Yahoo Finance 数据提供程序
    provider = YahooFinanceProvider(
        cache_enabled=False,  # 我们自行处理缓存
        cache_ttl=3600
    )

    try:
        # 下载数据
        market_data = provider.get_ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            adjustment=AdjustmentType(adjustment)
        )
        df = market_data.to_dataframe()

        # 保存到缓存
        if not df.empty:
            save_to_cache(cache_file, df)

        print(f"    已下载 {len(df)} 行数据")
        return df

    except Exception as e:
        print(f"    下载 {symbol} 时出错: {e}")
        raise


def download_market_data():
    """从 Yahoo Finance 下载示例市场数据（带本地缓存）。"""
    print("正在获取市场数据（带本地缓存）...")

    # 定义股票代码和日期范围
    symbols = ["NVDA"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2年数据

    # 获取每个股票的数据
    data = {}
    for symbol in symbols:
        try:
            df = get_symbol_data_with_cache(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                adjustment="adj"
            )
            df.index = pd.to_datetime(df.index, utc=True)
            data[symbol] = df

        except Exception as e:
            print(f"    获取 {symbol} 数据时出错: {e}")

    return data


class MACrossoverStrategy(SimpleStrategy):
    """
    双均线交叉策略 - 使用 SimpleStrategy 框架

    当快速均线上穿慢速均线时买入（信号=1）
    当快速均线下穿慢速均线时卖出（信号=-1）

    代码简洁，仅需约15行即可定义完整策略！
    """
    # 使用 param() 声明策略参数
    fast = param(10, min=2, max=100, description="快速均线周期")
    slow = param(30, min=5, max=200, description="慢速均线周期")

    def signals(self, data):
        """
        生成交易信号

        Args:
            data: 包含 OHLCV 数据的 DataFrame

        Returns:
            pd.Series: 信号值 (1=买入/多头, -1=卖出/空头, 0=无信号)
        """
        # 计算快速和慢速移动平均线
        fast_ma = data['close'].rolling(window=self.params.fast).mean()
        slow_ma = data['close'].rolling(window=self.params.slow).mean()

        # 生成信号：快线在慢线之上为买入信号(1)，否则为卖出信号(-1)
        signal = np.where(fast_ma > slow_ma, 1, -1)

        return pd.Series(signal, index=data.index)


def run_backtest(data, strategy):
    """运行事件驱动回测。"""
    print("\n" + "="*60)
    print("运行事件驱动回测")
    print("="*60)

    # 创建事件驱动回测引擎
    engine = EventDrivenBacktestEngine(
        initial_capital=100000.0,  # 初始资金10万美元
        commission=0.001,          # 0.1% 手续费
        slippage=0.0005,           # 0.05% 滑点
        allow_short=False,         # 不允许做空
        max_position_size=0.5,     # 最大仓位50%
        stop_loss=0.1,             # 10% 止损
        take_profit=0.2            # 20% 止盈
    )

    # 仅使用第一个股票进行回测（简化）
    symbol = list(data.keys())[0]
    symbol_data = data[symbol]

    # 准备数据字典
    backtest_data = {symbol: symbol_data}

    # 定义日期范围
    start_date = symbol_data.index[0]
    end_date = symbol_data.index[-1]

    print(f"回测股票: {symbol}")
    print(f"回测期间: {start_date.date()} 至 {end_date.date()}")
    print(f"数据点数: {len(symbol_data)}")
    print(f"策略参数: 快速均线={strategy.params.fast}, 慢速均线={strategy.params.slow}")

    # 运行回测
    result = engine.run(
        strategy=strategy,
        data=backtest_data,
        start_date=start_date,
        end_date=end_date,
        frequency='daily'
    )

    return result


def analyze_results(result):
    """分析并显示回测结果。"""
    print("\n" + "="*60)
    print("回测结果分析")
    print("="*60)

    # 基本性能指标
    print("\n性能指标:")
    print(f"  总收益率: {result.total_return:.2%}")
    print(f"  年化收益率: {result.annual_return:.2%}")
    print(f"  夏普比率: {result.sharpe_ratio:.3f}")
    print(f"  最大回撤: {result.max_drawdown:.2%}")
    print(f"  波动率: {result.volatility:.2%}")
    print(f"  索提诺比率: {result.sortino_ratio:.3f}")
    print(f"  卡尔玛比率: {result.calmar_ratio:.3f}")

    # 交易统计
    print("\n交易统计:")
    print(f"  总交易次数: {result.total_trades}")
    print(f"  盈利交易: {result.winning_trades}")
    print(f"  亏损交易: {result.losing_trades}")
    print(f"  胜率: {result.win_rate:.2%}")
    print(f"  盈亏比: {result.profit_factor:.3f}")
    print(f"  平均交易收益: {result.avg_trade_return:.2%}")

    # 资金统计
    print("\n资金统计:")
    print(f"  初始资金: ${result.initial_capital:,.2f}")
    print(f"  最终资金: ${result.final_capital:,.2f}")
    print(f"  峰值资金: ${result.peak_capital:,.2f}")
    print(f"  谷值资金: ${result.trough_capital:,.2f}")
    print(f"  总手续费: ${result.total_commission:,.2f}")
    print(f"  总滑点: ${result.total_slippage:,.2f}")

    return result


def print_trade_summary(result):
    """打印交易摘要信息。"""
    print("\n" + "="*60)
    print("交易摘要")
    print("="*60)

    if not result.trades:
        print("回测期间未执行任何交易。")
        return

    print(f"\n总交易次数: {len(result.trades)}")
    print("-" * 80)

    # 打印表头
    print(f"{'日期':<12} {'时间':<8} {'股票':<8} {'方向':<6} {'数量':<10} {'价格':<10} {'手续费':<10}")
    print("-" * 80)

    # 打印每笔交易
    for trade in result.trades[:10]:  # 仅显示前10笔交易
        trade_date = trade.timestamp.strftime("%Y-%m-%d")
        trade_time = trade.timestamp.strftime("%H:%M:%S")
        quantity_str = f"{trade.quantity:,.2f}"
        price_str = f"${trade.price:,.2f}"
        commission_str = f"${trade.commission:,.2f}"

        print(f"{trade_date:<12} {trade_time:<8} {trade.symbol:<8} {trade.side:<6} {quantity_str:<10} {price_str:<10} {commission_str:<10}")

    if len(result.trades) > 10:
        print(f"\n... 还有 {len(result.trades) - 10} 笔交易 ...")

    # 打印汇总
    print("-" * 80)
    total_buys = sum(1 for t in result.trades if t.side.lower() == 'buy')
    total_sells = sum(1 for t in result.trades if t.side.lower() == 'sell')
    total_commission = sum(t.commission for t in result.trades)
    total_slippage = sum(t.slippage for t in result.trades)

    print(f"\n交易汇总:")
    print(f"  买入次数: {total_buys}")
    print(f"  卖出次数: {total_sells}")
    print(f"  总手续费: ${total_commission:,.2f}")
    print(f"  总滑点: ${total_slippage:,.2f}")


def plot_results(result, title="回测结果"):
    """绘制回测结果图表。"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 权益曲线
        ax1 = axes[0]
        result.equity_curve.plot(ax=ax1, title=f'{title} - 权益曲线', color='blue')
        ax1.set_ylabel('组合价值 ($)')
        ax1.grid(True, alpha=0.3)

        # 回撤
        ax2 = axes[1]
        result.drawdown.plot(ax=ax2, title='回撤', color='red')
        ax2.set_ylabel('回撤 (%)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"警告: 无法生成图表: {e}")
        print("matplotlib 可能未安装。使用以下命令安装: pip install matplotlib")


def main():
    """主函数 - 运行双均线策略回测示例。"""
    print("="*60)
    print("Djinn 量化回测框架 - 双均线策略示例 (SimpleStrategy)")
    print("="*60)

    # 1. 获取市场数据
    print("\n1. 正在获取市场数据...")
    try:
        data = download_market_data()
        if not data:
            print("   警告: 未下载到数据")
            return
    except Exception as e:
        print(f"   下载数据时出错: {e}")
        return

    # 2. 创建策略
    print("\n2. 正在创建交易策略...")
    strategy = MACrossoverStrategy(
        fast=10,  # 10日快速均线
        slow=30   # 30日慢速均线
    )
    print(f"   策略已创建: 双均线交叉策略")
    print(f"   策略参数: fast={strategy.params.fast}, slow={strategy.params.slow}")
    print(f"   （与原 MovingAverageCrossover 策略约500行代码相比，")
    print(f"    使用 SimpleStrategy 仅需约15行代码！）")

    # 3. 运行回测
    print("\n3. 正在运行回测...")
    result = run_backtest(data, strategy)

    # 4. 分析结果
    print("\n4. 正在分析结果...")
    analyze_results(result)

    # 5. 打印交易摘要
    print("\n5. 交易详情...")
    print_trade_summary(result)

    # 6. 绘制结果（可选）
    # print("\n6. 正在生成图表...")
    # plot_results(result, title="双均线策略回测")

    print("\n" + "="*60)
    print("示例运行成功完成！")
    print("="*60)

    return {
        'result': result,
        'data': data,
        'strategy': strategy
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n结果保存在 'results' 字典中")
    except KeyboardInterrupt:
        print("\n\n示例被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

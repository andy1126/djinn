# 策略框架改进设计文档

## 概述

本设计文档描述 Djinn 策略框架的简化改进，目标是让简单策略（如双均线）可以用 < 20 行代码实现，同时保留对复杂策略的扩展能力。

## 现状问题

当前双均线策略需要近 500 行代码，包含：
- 大量重复性的初始化、验证代码
- 需要理解继承、抽象方法等概念
- 文档和注释占比过高

## 目标

1. 简单策略代码量从 500 行降至 20 行以内
2. 保留对复杂策略的扩展能力
3. 与改进后的回测引擎无缝集成

## 设计

### SimpleStrategy 基础设计

**核心思想：** 约定优于配置，自动推导指标和信号逻辑

```python
from djinn import SimpleStrategy, param

class MACrossover(SimpleStrategy):
    # 参数声明（自动验证、文档生成）
    fast = param(10, min=2, max=100, description="快速均线周期")
    slow = param(30, min=5, max=200, description="慢速均线周期")

    def signals(self, data):
        """
        返回信号序列：
        1  = 买入/持有
        -1 = 卖出/空仓
        0  = 无信号（保持当前状态）
        """
        fast_ma = data.close.rolling(self.params.fast).mean()
        slow_ma = data.close.rolling(self.params.slow).mean()

        return np.where(fast_ma > slow_ma, 1, -1)
```

**对比：**

| 项目 | 当前方式 | 简化方式 |
|------|---------|----------|
| 代码行数 | ~500行 | ~15行 |
| 需要实现的方法 | 5+个 | 1个 |
| 参数处理 | 手动验证 | 声明式验证 |
| 指标计算 | 手动管理 | 自动缓存 |

### 内部机制

**参数系统 `param()`：**

```python
def param(default, *, min=None, max=None, description=None, choices=None):
    """参数声明，支持自动验证和文档生成"""
    return Parameter(
        default=default,
        min=min, max=max,
        description=description,
        choices=choices
    )
```

**SimpleStrategy 基类自动处理：**

```python
class SimpleStrategy:
    def __init__(self, **kwargs):
        # 1. 收集所有 param 声明的参数
        self.params = self._collect_params()

        # 2. 用传入值覆盖默认值
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        # 3. 自动验证参数
        self._validate_params()

        # 4. 初始化指标缓存
        self._indicator_cache = {}

    def _collect_params(self):
        """从类属性收集参数声明"""
        params = {}
        for name in dir(self):
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, Parameter):
                params[name] = attr.default
        return SimpleNamespace(**params)

    def calculate_signals_vectorized(self, data):
        """适配向量化回测引擎"""
        return self.signals(data)

    def calculate_signal(self, symbol, data, current_date):
        """适配事件驱动回测引擎"""
        signals = self.signals(data)
        if current_date in signals.index:
            return signals.loc[current_date]
        return 0.0
```

### 扩展机制 - 支持复杂策略

**场景1: 简单趋势跟踪**

```python
class MACrossover(SimpleStrategy):
    fast = param(10)
    slow = param(30)

    def signals(self, data):
        return np.where(
            data.close.rolling(self.params.fast).mean() >
            data.close.rolling(self.params.slow).mean(), 1, -1
        )
```

**场景2: 多资产/多因子策略**

```python
class MultiFactorStrategy(SimpleStrategy):
    # 声明关注的资产
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    # 多因子评分
    factors = {
        'momentum': lambda d: d.close.pct_change(20),
        'value': lambda d: 1 / d.pe_ratio,
        'quality': lambda d: d.roe,
    }

    def signals(self, data):
        # data 是多资产 DataFrame，包含 symbol 列
        scores = (
            self.factors['momentum'](data) * 0.4 +
            self.factors['value'](data) * 0.3 +
            self.factors['quality'](data) * 0.3
        )

        # 返回每个资产的信号，选取得分前2的买入
        return scores.rank() <= 2
```

**场景3: 需要状态管理的策略**

```python
class StatefulStrategy(SimpleStrategy):
    lookback = param(20)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 自定义状态
        self.state = {
            'last_signal': None,
            'consecutive_count': 0,
        }

    def signals(self, data):
        # 访问上次信号状态
        if self.state['last_signal'] == 1:
            # 前一周期是买入信号，可能需要调整逻辑
            pass

        # 计算新信号
        signal = self._calculate(data)

        # 更新状态
        self.state['last_signal'] = signal
        return signal
```

**渐进式复杂度支持：**

```python
# SimpleStrategy - 简单场景，声明式
class SimpleStrategy:
    """适合：单资产、纯信号、无状态"""
    pass

# AdvancedStrategy - 中等复杂度
class AdvancedStrategy(SimpleStrategy):
    """适合：多资产、需要状态、简单订单管理"""

    def on_bar(self, context):
        """
        每根K线调用，可以：
        - 访问当前持仓
        - 提交/取消订单
        - 读取历史信号
        """
        if not context.has_position('AAPL'):
            if self.should_buy(context.data):
                context.buy('AAPL', size=100)
        else:
            if self.should_sell(context.data):
                context.sell('AAPL')

    def should_buy(self, data):
        # 子类实现
        pass

# FullStrategy - 原 Strategy 基类
class FullStrategy(Strategy):
    """适合：高频、复杂订单类型、多事件处理"""
    pass
```

### 完整整合

```python
# 完整使用示例

from djinn import SimpleStrategy, param, BacktestEngine

# 1. 定义策略（15行代码）
class MACrossover(SimpleStrategy):
    fast = param(10, min=2, max=100)
    slow = param(30, min=5, max=200)

    def signals(self, data):
        fast_ma = data.close.rolling(self.params.fast).mean()
        slow_ma = data.close.rolling(self.params.slow).mean()
        return np.where(fast_ma > slow_ma, 1,
               np.where(fast_ma < slow_ma, -1, 0))

# 2. 运行回测
engine = BacktestEngine(
    initial_cash=100000,
    commission=0.001,
    slippage_model='fixed',  # 或 'dynamic'
)

result = engine.run(
    strategy=MACrossover(fast=5, slow=20),
    symbols=['AAPL', 'MSFT'],
    start='2020-01-01',
    end='2023-12-31',
    timeframe='1d'
)

# 3. 查看结果
print(result.metrics)  # 收益率、夏普比率、最大回撤等
result.plot()          # 绘制权益曲线
```

**核心架构：**

```
User Code Layer
  SimpleStrategy ──┐
  AdvancedStrategy─┼── 统一接口 ──> BacktestEngine.run()
  FullStrategy  ──┘
          │
          ▼
Strategy Adapter Layer
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ SimpleAdapter │  │AdvancedAdapter│  │  FullAdapter  │
  │  (signals)   │  │  (on_bar)    │  │ (events)     │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         └─────────────────┼─────────────────┘
                           │
                           ▼
              Backtest Engine Core
  EventBus ──> OrderManager ──> Portfolio ──> Metrics
```

## 实现优先级

1. **Phase 1**: Parameter 类和参数验证
2. **Phase 2**: SimpleStrategy 基类
3. **Phase 3**: StrategyAdapter 层（统一接口）
4. **Phase 4**: AdvancedStrategy 扩展
5. **Phase 5**: 内置策略库（MACrossover, RSI, Bollinger 等）

---

*设计日期: 2026-02-12*

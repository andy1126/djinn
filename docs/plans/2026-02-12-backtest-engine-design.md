# 回测引擎改进设计文档

## 概述

本设计文档描述 Djinn 回测引擎的核心改进，包括交易执行模拟、多资产回测和事件驱动架构优化。

## 目标

1. 支持限价单、止损单、止损限价单的真实交易模拟
2. 支持 2-10 个资产的多资产组合回测
3. 实现精确的事件排序，支持多时间粒度混合使用

## 架构设计

### 整体架构

改进后的回测引擎采用**分层事件驱动架构**，核心组件包括：

**EventBus（事件总线）**
- 优先队列按 `(timestamp, priority, sequence)` 排序，确保事件严格有序
- 支持 Tick/分钟/K线 多种时间粒度的事件注册
- 提供 `register_event()`, `emit()`, `process_next()` 接口

**OrderManager（订单管理器）**
- 统一管理限价单、止损单、止损限价单
- 维护活跃订单簿，在每个价格事件时检查触发条件
- 支持部分成交，返回 `FillEvent` 包含成交量和成交均价

**MultiAssetEngine（多资产引擎）**
- 每个资产独立的 `AssetContext`，维护自己的持仓和订单
- 资产间通过 `Portfolio` 共享资金池
- 支持独立回测和组合回测两种模式

**数据流示意**：
```
数据源 → EventBus → 策略处理 → OrderManager → 成交模拟 → 更新持仓/PnL
```

### 订单管理系统

**核心订单类型实现：**

```python
class Order:
    id: str
    symbol: str
    side: Buy/Sell
    quantity: Decimal
    status: Pending/Partial/Complete/Cancelled

class LimitOrder(Order):
    limit_price: Decimal          # 限价，只有价格触及才成交

class StopOrder(Order):
    stop_price: Decimal           # 触发价，触及后转为市价单
    triggered: bool = False       # 是否已触发

class StopLimitOrder(Order):
    stop_price: Decimal           # 触发价
    limit_price: Decimal          # 触发后的限价
    triggered: bool = False
```

**订单生命周期：**

1. **提交** → 进入活跃订单簿，状态为 Pending
2. **检查触发** → 每个价格事件时，`OrderManager` 检查：
   - 限价单：当价格 ≤ 买限价 / ≥ 卖限价时成交
   - 止损单：价格触及 stop_price 后转为市价单立即成交
   - 止损限价单：触发后转为限价单等待成交
3. **部分成交** → 根据市场深度模拟（配置 `fill_ratio` 参数）
4. **成交确认** → 生成 `FillEvent`，更新持仓和资金

**滑点模型：**
```python
class SlippageModel:
    def apply(self, order: Order, market_price: Decimal) -> Decimal:
        # 基于订单规模和市场波动率调整成交价
        pass
```

### 多资产引擎

**AssetContext（资产上下文）：**
```python
class AssetContext:
    symbol: str
    position: Position              # 当前持仓
    active_orders: List[Order]      # 该资产的活跃订单
    history: List[Trade]            # 成交历史
    price_cache: MarketData         # 最新市场数据
```

**MultiAssetEngine 核心逻辑：**
```python
class MultiAssetEngine:
    contexts: Dict[str, AssetContext]
    portfolio: Portfolio            # 共享资金池
    mode: BacktestMode              # INDEPENDENT 或 PORTFOLIO

    def process_bar(self, timestamp, symbol, ohlcv):
        ctx = self.contexts[symbol]
        # 1. 检查并执行该资产的活跃订单
        fills = self.order_manager.check_orders(ctx, ohlcv)
        # 2. 调用策略生成信号
        signals = self.strategy.generate(ctx, ohlcv)
        # 3. 提交新订单
        for signal in signals:
            order = self.create_order(signal)
            self.order_manager.submit(order)
```

**两种回测模式：**

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **INDEPENDENT** | 每个资产独立运行策略，最终汇总各资产收益曲线 | 多品种趋势跟踪 |
| **PORTFOLIO** | 策略统一决策资金分配，资产间再平衡 | 资产配置策略 |

**事件处理示例（AAPL 和 TSLA 同时回测）：**
```
09:30:00.000  AAPL 开盘事件 → 策略处理 → 可能提交订单
09:30:00.000  TSLA 开盘事件 → 策略处理 → 可能提交订单
09:31:00.000  AAPL 分钟K线 → 检查订单 → 策略处理...
```

### 数据流与事件处理

**事件优先级定义（数值越小优先级越高）：**
```python
class EventPriority:
    MARKET_OPEN = 1      # 市场开盘
    PRICE_UPDATE = 2     # 价格更新（Tick/Bar）
    ORDER_FILL = 3       # 订单成交
    SIGNAL = 4           # 策略信号
    MARKET_CLOSE = 5     # 市场收盘
```

**完整事件处理流程：**

```
1. 数据加载器读取多资产数据，生成事件序列
   (BarEvent, TickEvent 按时间戳合并排序)
              ↓
2. EventBus 按优先级取出下一个事件
   同一时间点：MarketOpen → Bar → Fill → Signal → Close
              ↓
3. 策略处理 BarEvent
   - 更新技术指标
   - 检查止损/止盈条件
   - 生成交易信号
              ↓
4. OrderManager 处理订单
   - 检查活跃限价单/止损单是否触发
   - 模拟成交，应用滑点模型
   - 生成 FillEvent 更新持仓
              ↓
5. Portfolio 更新资金与风险指标
   - 计算已实现/未实现盈亏
   - 检查风险限制（最大回撤、仓位上限）
```

**时间粒度混合支持：**
- 策略可订阅不同粒度：`subscribe('AAPL', '1m')` 和 `subscribe('SPY', '1d')`
- EventBus 自动对齐时间戳，日级事件在日内只触发一次

## 错误处理

**异常类型设计：**
```python
class BacktestError(Exception): pass
class OrderError(BacktestError): pass      # 订单参数错误、资金不足
class DataError(BacktestError): pass       # 数据缺失、时间戳异常
class RiskError(BacktestError): pass       # 触发风控限制
```

**关键错误处理点：**

| 场景 | 处理方式 |
|------|----------|
| 资金不足 | 拒单并记录 `OrderRejected` 事件 |
| 数据时间戳乱序 | 抛出 `DataError`，停止回测 |
| 订单价格非法 | 提交时校验，抛出 `OrderError` |
| 触发最大回撤 | 根据配置：仅警告 / 停止开仓 / 停止回测 |

## 测试策略

```python
# 单元测试
test_order_lifecycle.py       # 订单提交→触发→成交完整流程
test_event_ordering.py        # 验证事件优先级和排序
test_multiasset_sync.py       # 多资产事件同步处理

# 集成测试
test_backtest_scenarios.py    # 典型场景：趋势跟踪、均值回归、跨市场套利

# 验证测试
test_fill_accuracy.py         # 对比已知结果验证成交逻辑正确性
```

## 性能基准（2-10资产）

- 日级数据 10 资产 10 年回测 < 5 秒
- 分钟级数据 10 资产 1 年回测 < 30 秒

## 实现优先级

1. **Phase 1**: EventBus 基础架构 + 事件优先级排序
2. **Phase 2**: OrderManager 限价单支持
3. **Phase 3**: 止损单和止损限价单
4. **Phase 4**: MultiAssetEngine 多资产支持
5. **Phase 5**: 滑点模型与性能优化

---

*设计日期: 2026-02-12*

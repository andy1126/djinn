# 计划 A:回测内核 —— 撮合、费用与交易约束正确性

> 目标读者:执行模型。本文档自包含,每项改动给出:问题、修改方案(文件/行号/代码)、测试验证。
> 总原则:改动后必须 `pytest -n auto -m "not network and not slow and not benchmark"` 全绿;新增行为必须新增测试。
> 项目约定:black line-length 88;mypy --strict;UI 文案/注释中文,标识符英文;Decimal 记账硬性不变量(现金/股数/费用用 `decimal.Decimal`,禁止 float 记账)。

## 总览

| # | 改进点 | 严重度 | 预估工作量 |
|---|---|---|---|
| A1 | 港股印花税双边化 + A 股过户费仅沪市 | P0 | 0.5 天 |
| A2 | A 股卖出豁免整手取整(零股可卖) | P0 | 0.5 天 |
| A3 | 撮合批次先卖后买(两阶段执行) | P1 | 1 天 |
| A4 | target_percent 估值口径统一(equity 用开盘价) | P1 | 0.5 天 |
| A5 | 死字段处理:fill_ref / limit_price / initial_alloc_on_start | P1 | 1 天 |
| A6 | 停牌续挂结构化(去中文字符串匹配) | P2 | 0.5 天 |
| A7 | 基准曲线前导 NaN 传染 | P1 | 0.25 天 |
| A8 | 进阶分配器在引擎再平衡路径静默退化等权 → 启动校验 | P1 | 0.5 天 |
| A9 | 退市/长期无数据标的处理(强制平仓或核销) | P1 | 1.5 天 |
| A10 | adjust=none 时公司行为(分红)接入主循环 | P2 | 1 天 |
| A11 | 成交量上限约束 | P2 | 1 天 |
| A12 | 删除或接入 EventBus(死代码) | P2 | 0.25 天 |

---

## A1. 港股印花税双边化 + A 股过户费仅沪市

### 问题
`src/djinn/engine/commission.py:55` 基类 `ConservativeCommissionModel.cost()`:

```python
stamp = amount * self.stamp_duty_rate if side == "sell" else D(0)
```

印花税只收卖出单边。而 `HKCommissionModel`(同文件 :90-104)的 docstring 与费率均按**双边** 1.3‰ 设定(港股印花税买卖双边征收,2023-11 起税率 0.1%,此前 0.13%),导致港股买入方向漏收印花税,HK 回测成本被低估约一半。

另外 `ChinaCommissionModel`(:60-80 附近)的过户费 `transfer_fee_rate` 对所有 A 股双边收取,但过户费实际仅**沪市**(600/601/603/605/688 开头)收取,深市(000/002/300/301 开头)不收 → 深市回测成本高估。

### 修改方案

**文件:`src/djinn/engine/commission.py`**

1. 把印花税的单边/双边行为参数化。在 `ConservativeCommissionModel.__init__` 增加字段:

```python
class ConservativeCommissionModel(CommissionModel):
    def __init__(
        self,
        rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.0005,
        transfer_fee_rate: float = 0.00002,
        *,
        stamp_duty_sides: str = "sell",   # "sell"(A股) / "both"(港股)
    ) -> None:
        ...
        self.stamp_duty_sides = stamp_duty_sides
```

2. `cost()` 中印花税分支改为:

```python
if self.stamp_duty_sides == "both":
    stamp = amount * self.stamp_duty_rate
else:
    stamp = amount * self.stamp_duty_rate if side == "sell" else D(0)
```

3. `HKCommissionModel.__init__` 调 `super().__init__(..., stamp_duty_sides="both")`,并把 `stamp_duty_rate` 默认值从 `0.0013` 更新为 `0.001`(2023-11-17 起现行 0.1%;保留参数可覆盖以回测更早区间)。

4. `ChinaCommissionModel` 增加沪市判定。`cost()` 的签名目前只有 `(side, price, qty)`,不知道标的 → 需要扩展。最小侵入做法:给 `CommissionModel.cost()` 增加可选关键字参数 `symbol: str | None = None`(基类与所有子类签名同步加,默认 None 保持兼容),`Broker.execute()` 在 `broker.py:136` 调用处传 `symbol=order.symbol`。`ChinaCommissionModel.cost()` 内:

```python
def cost(self, side, price, qty, *, symbol=None):
    amount = D(qty) * D(price)
    commission = max(amount * self.rate, self.min_commission)
    stamp = amount * self.stamp_duty_rate if side == "sell" else D(0)
    # 过户费仅沪市(60xxxx/68xxxx)收取,深市不收
    transfer = (
        amount * self.transfer_fee_rate
        if symbol is not None and symbol.startswith(("60", "68"))
        else D(0)
    )
    return q_money(commission + stamp + transfer)
```

注意:A 股代码在 akshare 链路为 6 位数字(如 `600519`);若带 `.SH`/`.SZ` 后缀(CSV 数据源常见),判定改为 `symbol.split(".")[0].startswith(("60","68"))`。

5. `make_commission()` 工厂(同文件底部)无需改动;但需在 docstring 更新"印花税单边/双边按市场"说明。

### 测试验证
文件:`tests/unit/test_commission_slippage.py`(已存在,追加用例)。

- `test_hk_stamp_duty_both_sides`:构造 `HKCommissionModel()`,`cost("buy", 10.0, 10000)` 与 `cost("sell", 10.0, 10000)` 的印花税额应**相等且都 > 0**;总额 = `max(amount*rate, min) + amount*0.001`(双边)。
- `test_cn_stamp_duty_sell_only`:`ChinaCommissionModel()` 买入无印花税、卖出有。
- `test_cn_transfer_fee_sh_only`:`cost(..., symbol="600519")` 含过户费;`cost(..., symbol="000001")` 不含;`cost(..., symbol="300750.SZ")` 不含(后缀剥离)。
- 回归:跑一个含港股标的的最小回测(可复用现有 yahoo stub 或 CSV fixture),断言总费用较改动前上升(买入方向新增印花税)。

---

## A2. A 股卖出豁免整手取整(零股可卖)

### 问题
`src/djinn/engine/constraints.py:76-81`(在 `check_constraints()` 内):

```python
lot = constraints.market.lot_size if constraints.enforce_lot else 1
if constraints.enforce_lot and lot > 1:
    qty = floor_shares(qty, lot)
    if qty <= 0:
        return CheckResult(False, reason=f"不足最小手 {lot} 股")
```

买卖统一向下取整到 100 股整数倍。A 股规则:**买入必须整手,卖出允许零股**(零股来源于分红再投资碎股 `account.py:214`、或历史整手拆分)。当前实现下,一旦产生 <100 股尾差,卖出单被 floor 后剩尾差,尾差再卖被"不足最小手"拒单 → 头寸永远无法清零。

### 修改方案

**文件:`src/djinn/engine/constraints.py`**

在最小手校验分支加方向判断:

```python
lot = constraints.market.lot_size if constraints.enforce_lot else 1
if constraints.enforce_lot and lot > 1 and side == "buy":
    qty = floor_shares(qty, lot)
    if qty <= 0:
        return CheckResult(False, reason=f"不足最小手 {lot} 股")
elif constraints.enforce_lot and lot > 1 and side == "sell":
    # 卖出:整手部分之外的零股允许一次性卖出(不得超过可用股数);
    # 不向下取整,由后续 Account.sell 的 available 校验兜底。
    qty = min(raw_qty, <可用股数>)  # 见下
```

注意:`check_constraints()` 当前不接收持仓可用股数。方案:给 `check_constraints()` 增加可选参数 `available_qty: Decimal | None = None`,`Broker.execute()`(broker.py:99-107)调用处传 `available_qty=pos.available if pos else None`(从 `self.account.positions.get(order.symbol)` 取)。卖出分支:

```python
if available_qty is not None and raw_qty > available_qty:
    qty = available_qty  # 夹到可用(零股全出)
else:
    qty = raw_qty        # 不取整,允许零股
```

同时删除卖出方向的"不足最小手"拒单逻辑(仅保留买入)。

### 测试验证
文件:`tests/unit/test_constraints.py`(已存在)。

- `test_sell_odd_lot_allowed`:constraints 为 CN(lot=100),卖出 raw_qty=37 → `check.ok` 为 True 且 `adjusted_qty == 37`。
- `test_sell_clamped_to_available`:raw_qty=150,available=120 → adjusted=120(不取整到 100)。
- `test_buy_still_floored`:买入 raw_qty=150 → adjusted=100 不变(回归)。
- 端到端:`tests/unit/test_account.py` 或新增引擎级用例——买入 100 股 → 分红再投资产生 3 股碎股(直接调 `Account.receive_dividend(..., reinvest=True, price=...)`)→ 全仓卖出(target_percent=0)→ 断言 `pos.qty == 0` 且无"不足最小手"拒单。

---

## A3. 撮合批次先卖后买(两阶段执行)

### 问题
`src/djinn/engine/event_engine.py:158-172` 按 `pending_orders` 列表 FIFO 顺序撮合。`Rebalancer._build_orders`(`portfolio/rebalance.py:113-128`)按 symbols 顺序混排买卖意图。当买单先于卖单成交时,卖单回笼的现金尚未到账,买单触发 `_maybe_shrink_buy` 缩减甚至被拒——而随后的卖单本可提供资金。**同一批订单的成交结果依赖列表顺序**,回测不可复现地受 symbols 排序影响。

### 修改方案

**文件:`src/djinn/engine/event_engine.py`**(:158-172 撮合块)

把单循环改为两阶段:

```python
if pending_orders:
    equity_now = account.equity_float(prices_mtm)
    # 两阶段撮合:先全部卖单(回笼现金),再全部买单,消除顺序依赖。
    sells = [o for o in pending_orders if o.side == "sell"]
    buys = [o for o in pending_orders if o.side == "buy"]
    still_pending: list[Order] = []
    for order in sells + buys:
        bar = bars.get(order.symbol)
        if bar is None:
            still_pending.append(order)
            continue
        result = broker.execute(order, bar, prev_close.get(order.symbol), equity_now)
        if isinstance(result, Rejection) and result.retryable:  # 见 A6
            still_pending.append(order)
    pending_orders = still_pending
```

注意点:
- `equity_now` 在两阶段共用同一份(以开盘前估值为基准),买单的 target_percent 解析不因此漂移;这与 A4 的口径统一是配套的。
- 阶段内仍保持原相对顺序(稳定排序),保证可复现。
- `Order.side` 类型是 `Literal["buy","sell"]`,过滤无遗漏。

### 测试验证
文件:`tests/unit/test_strategy.py` 或新增 `tests/unit/test_engine_ordering.py`。

- `test_sell_before_buy_same_batch`:构造组合策略,同一调仓日发出"卖 A(回笼 10 万)+ 买 B(需 10 万)",初始现金不足覆盖买 B。断言:B 买单**全额成交**且无缩减拒单;对比改动前行为(缩减)。
- `test_ordering_deterministic`:打乱 symbols 顺序跑两次,断言 fills 序列逐笔一致(符号、数量、价格)。
- 回归:现有全部引擎级测试不应有非预期差异(除本就依赖 FIFO 缩减语义的用例,若有失败逐一确认是新语义更优后更新断言)。

---

## A4. target_percent 估值口径统一

### 问题
`event_engine.py:159` 计算 `equity_now = account.equity_float(prices_mtm)` 用的是**今日收盘价**(`prices_mtm` 来自 `bars[s].close`),而 `broker.py:65` 的成交价参考 `ref_price = bar.open`。目标市值、当前市值、成交价三个参考点不一致:开盘大幅跳空时,`delta_mv = target*equity(收盘价口径) − cur_mv(开盘价口径)` 偏差明显。

### 修改方案

**文件:`src/djinn/engine/event_engine.py`**

在撮合块内,为 `equity_now` 单独构造**开盘价口径**的价格表:

```python
if pending_orders:
    # 撮合口径统一为开盘价:equity/当前市值/成交价三处一致
    prices_open = {s: b.open for s, b in bars.items() if b is not None and b.open > 0}
    for s, pc in prev_close.items():
        prices_open.setdefault(s, pc)  # 当日无行情标的沿用昨收
    equity_now = account.equity_float(prices_open)
```

(`prices_mtm` 仍用于 PortfolioView 与 mark-to-market,不动。)

### 测试验证
- 新增 `tests/unit/test_engine_ordering.py::test_target_percent_uses_open_price`:构造跳空场景(昨收 100、今开 110),target_percent=0.5,初始现金已知;断言买入股数 == `0.5 * equity_open / 110`(手算期望值),而非按 100 算出的值。
- 回归:现有回测集成测试的 fills 数量/股数会有小幅变化属预期,逐一核对确认方向合理(跳空高开时买单股数减少)。

---

## A5. 死字段处理:fill_ref / limit_price / initial_alloc_on_start

### 问题
三个字段定义了但从未被读取,用户设置后**静默无效**:
- `EngineConfig.fill_ref`(`event_engine.py:48`,注释称支持 "open"/"close"/"vwap")——成交价恒为 `bar.open`(broker.py:65);
- `EngineConfig.initial_alloc_on_start`(`event_engine.py:50`);
- `Order.limit_price` / `OrderIntent.limit_price`(`engine/events.py:43`、`strategy/signal.py:48`)——撮合从不读取。

静默忽略比报错更危险:用户以为设了限价单,实际按市价成交。

### 修改方案(实现 fill_ref + limit_price;删除 initial_alloc_on_start)

**1. fill_ref 实现** —— 三处改动:

a. `broker.py:65` 的 `ref_price` 计算提取为方法并尊重配置。给 `Broker` 增加字段 `fill_ref: str = "open"`(dataclass 字段),`EventDrivenEngine.run()` 构造 Broker 时(event_engine.py:121-123)传 `fill_ref=cfg.fill_ref`。

b. broker.py:

```python
def _ref_price(self, bar: Bar) -> float:
    if self.fill_ref == "close":
        return bar.close if bar.close > 0 else bar.open
    if self.fill_ref == "vwap":
        # Bar 无 vwap 字段:用 amount/volume 近似;无成交量时退化 open
        return bar.amount / bar.volume if bar.volume > 0 else bar.open
    return bar.open if bar.open > 0 else bar.close
```

`execute()` 首行改为 `ref_price = self._ref_price(bar)`。

c. `config/models.py` 检查 `CostsConfig`/引擎相关模型是否暴露 fill_ref;若无,在 `BacktestConfig` 增加 `fill_ref: Literal["open","close","vwap"] = "open"` 并在 `cli/runner.py` 的 `build_engine_config()` 传递。

**2. limit_price 实现(限价单语义)** —— `broker.py` `execute()` 在滑点之后、约束校验之前插入:

```python
# 限价单:买限价 ≥ 成交价才成交;卖限价 ≤ 成交价才成交;否则挂起(非拒单)
if order.limit_price is not None:
    lp = float(order.limit_price)
    if (order.side == "buy" and price > lp) or (order.side == "sell" and price < lp):
        return Rejection(
            order_id=order.id, timestamp=bar.timestamp, symbol=order.symbol,
            side=order.side, reason=f"限价未达(limit={lp}, ref={price:.4f})",
            requested_qty=float(qty), tag=order.tag, retryable=True,  # retryable 见 A6
        )
```

依赖 A6 的 `retryable` 字段先落地(限价未达的订单应挂起等次日,而非丢弃)。`Context.buy/sell`(strategy/base.py:152-198)需加 `limit: float | None = None` 参数透传到 `OrderIntent`。

**3. initial_alloc_on_start 删除** —— 该字段与 `portfolio.rebalance.period` 语义重叠(首日建仓已由 `rebalance` + `BuyAndHold` 类策略覆盖)。直接从 `EngineConfig` 删除字段;检查 `cli/runner.py`、`config/models.py`、前端 `types/index.ts` 无引用(grep 确认)后删净。

### 测试验证
- `test_fill_ref_close`:同一策略同一数据,fill_ref="close" 的成交价 == 当日 close(断言 fills[0].price)。
- `test_fill_ref_vwap`:amount/volume 已知时断言 price == amount/volume(±滑点)。
- `test_limit_buy_not_filled`:限价低于当日开盘价 → 当日无 fill、订单次日在 pending 中继续;次日价格落入限价 → 成交。
- `test_limit_sell_not_filled` 对称。
- `grep -rn initial_alloc_on_start src/ frontend/src/` 输出为空。

---

## A6. 停牌续挂结构化(去中文字符串匹配)

### 问题
`event_engine.py:169`:`if isinstance(result, Rejection) and "停牌" in result.reason` —— 续挂逻辑与中文文案硬耦合,改文案即破坏行为;A5 的限价未达也需要同样的"挂起而非拒单"语义。

### 修改方案

**文件:`src/djinn/engine/events.py` + `constraints.py` + `broker.py` + `event_engine.py`**

1. `Rejection` dataclass(events.py:60 附近)增加字段:`retryable: bool = False`。
2. `constraints.py` 停牌拒单处(:72-73)构造 CheckResult 时不含 retryable 信息 → 改法:`Broker.execute()` 在生成停牌 Rejection 的通用处(broker.py:108-119)无法区分原因;最简方案是 `check_constraints` 的 `CheckResult` 也加 `retryable: bool = False` 字段,停牌分支(:72-73)置 True,Broker 把它透传到 Rejection:

```python
rej = Rejection(..., reason=check.reason, retryable=check.retryable, ...)
```

3. A5 的限价未达 Rejection 置 `retryable=True`。
4. `event_engine.py:169` 改为 `if isinstance(result, Rejection) and result.retryable:`。
5. 序列化检查:`report_store.py` 与 `io/export.py` 用 getattr 读 Rejection 字段,新增字段不破坏;但 `tests/unit/test_api.py` 中若有 Rejection 的精确 dict 断言需补字段。

### 测试验证
- `test_suspension_retryable_flag`:构造停牌 bar(is_suspended=1)→ execute → 返回 Rejection 且 `retryable is True`;涨停拒单 `retryable is False`。
- `test_suspension_order_persist`:引擎级,停牌日订单留在 pending,复牌日成交(此用例可能已存在,改为断言不依赖中文串后仍绿)。
- `grep -n '"停牌" in' src/` 输出为空。

---

## A7. 基准曲线前导 NaN 传染

### 问题
`event_engine.py:231-232`:

```python
bm = benchmark.df["close"].reindex(idx).ffill()
benchmark_curve = (bm / bm.iloc[0]) * equity_curve.iloc[0]
```

基准数据起点晚于策略首个交易日时,`bm.iloc[0]` 为 NaN → 整条基准曲线 NaN;`compare_benchmark` 静默返回全 0(`trades.py:133-134` 的 len<2 分支)。

### 修改方案
`event_engine.py:231` 一行修复:

```python
bm = benchmark.df["close"].reindex(idx).ffill().bfill()
```

(bfill 用首个有效值回填前导;归一化基准起点即非 NaN。语义:策略早期基准视为持平,可接受。)

另在 `trades.py:130-134` 的 `compare_benchmark`:`s`/`b` dropna 后应再取交集(当前先 dropna 各自、再 pct_change 再交集,前导 NaN 已在 bfill 后消除,此函数无需改;但加一行防御:`if b.isna().all(): return BenchmarkStats()`)。

### 测试验证
- `test_benchmark_starts_late`:策略数据 2020-01 起,基准 2020-06 起 → benchmark_curve 无 NaN,`bm.iloc[0] == 第一个有效值`;`compare_benchmark` 的 beta/alpha 非全 0。
- 回归:现有基准对齐测试(test_metrics.py / 引擎集成)不破坏。

---

## A8. 进阶分配器静默退化 → 启动校验

### 问题
`event_engine.py:188-190` 引擎再平衡路径只传 `prices` 给 `Rebalancer.maybe_rebalance`,`Rebalancer._build_orders`(`rebalance.py:112`)再调 `allocation.target_weights(symbols, prices=prices)` —— 永远拿不到 `scores`/`cov`。`allocation.py` 中 `ScoreWeight`/`RiskParityWeight`/`MinVarianceWeight`/`MeanVarianceWeight` 在缺参时全部走 `_equal_weights` 分支(:136-137, 166-168, 184-186, 210-211),**无告警**。用户配置 `allocation: min_variance` + 引擎再平衡,得到的是等权。

(注:`FactorPortfolioStrategy` 路径不受影响——它自行再平衡并传 scores/cov,见 `strategy/library/factor_portfolio.py:99-100`。)

### 修改方案

**文件:`src/djinn/portfolio/allocation.py` + `src/djinn/engine/event_engine.py`**

1. 给各 Allocation 子类增加类属性 `requires: frozenset[str] = frozenset()`;`ScoreWeight.requires = {"scores"}`、`RiskParityWeight`/`MinVarianceWeight`/`MeanVarianceWeight.requires = {"cov"}`。
2. `Allocation.target_weights()` 在缺参退化时改为**显式告警一次**(而非静默):

```python
if self.requires and not provided:
    _log.warning("%s 需要 %s 但调用方未提供,退化为等权", type(self).__name__, self.requires)
```

(`_log.warning` 自带去重困难 → 在 Allocation 实例上加 `_warned: bool` 实例属性只警一次。)

3. `EventDrivenEngine.run()` 启动时(event_engine.py:139-141 附近)校验:

```python
if rebalancer is not None and getattr(allocation, "requires", frozenset()):
    raise ValueError(
        f"allocation={type(allocation).__name__} 需要 {allocation.requires},"
        "引擎再平衡路径无法提供;请改用 FactorPortfolioStrategy(scope=portfolio)"
        "或 allocation=equal/market_cap/custom"
    )
```

(选择"直接拒绝"而非告警:配置错误应尽早暴露。)

### 测试验证
- `test_engine_rejects_cov_allocation`:EngineConfig(allocation=MinVarianceWeight(), rebalance=Rebalancer(...)) → `run()` 抛 ValueError 且消息含 "FactorPortfolioStrategy"。
- `test_allocation_warns_once`:直接调 `MinVarianceWeight().target_weights(syms, prices=...)` 两次,caplog 中只有一条 warning。
- 回归:`allocation=equal` + rebalance 的现有集成测试不受影响。

---

## A9. 退市/长期无数据标的处理

### 问题
union 日历下(选股回测),已退市/长期停牌的持仓标的被 `prev_close` 永久前向填充估值(event_engine.py:147-151,注释已自知),按最后价永远计值;其 pending 订单无限期挂起(:163-164)。后果:净值曲线高估(死仓位按成本价计入)、资金永远占用。

### 修改方案

**文件:`src/djinn/engine/event_engine.py` + `EngineConfig`**

1. `EngineConfig` 增加:`delist_grace_days: int = 30`(连续无行情超过 N 个交易日判定退市)。
2. 主循环维护 `last_bar_date: dict[str, date]`(在 :218-220 更新 prev_close 的同一循环里更新)。
3. 每日撮合前(:157 之前)检查:

```python
for s in list(account.positions.keys()):
    pos = account.positions.get(s)
    if pos is None or pos.qty <= 0:
        continue
    last = last_bar_date.get(s)
    if last is None or (ts_date - last).days <= cfg.delist_grace_days * 1.5:  # 日历日粗判
        continue
    # 强制按最近可得价清仓(免滑点,计佣金),订单走正常 Broker 路径
    pc = prev_close.get(s)
    if pc and pc > 0:
        _log.warning("标的 %s 超 %d 天无行情,按退市强制平仓 @%s", s, cfg.delist_grace_days, ts_date)
        qty = float(pos.available) if con.enforce_t_plus_1 else float(pos.qty)
        if qty > 0:
            comm_fee = comm.cost("sell", pc, qty)
            account.sell(s, D(qty), pc, comm_fee, timestamp=ts_date, tag="delist")
            broker.fills.append(Fill(0, ts_date, s, "sell", qty, pc, float(comm_fee), "delist"))
```

4. 同处清理该标的的 pending 订单:`pending_orders = [o for o in pending_orders if o.symbol != s]`。
5. 注意 `pos.available` 在 T+1 下可能小于 `pos.qty`(当日买入冻结),取 available 即可;剩余冻结部分次日解冻后再清(循环会再次命中)。简化:若 `qty <= 0` 则跳过等次日。

### 测试验证
- `test_delist_forced_liquidation`:构造 union 日历回测,标的 X 行情止于 T,之后 40 天只有其他标的有行情 → 断言 T+31 附近出现 tag="delist" 的卖单,持仓归零,回收现金 == 最后价 × 股数 − 佣金。
- `test_delist_pending_purged`:X 的未成交买单在强平日后被清除。
- 净值连续性:强平当日 equity 与前后日衔接无跳变(除佣金损耗)。

---

## A10. adjust=none 时公司行为(分红)接入主循环

### 问题
`Bar.dividend`/`split_ratio` 字段存在(`data/schema.py:137-138`),`Account.receive_dividend` 实现存在(`account.py:194-216`),但**引擎从不调用**——全依赖复权价格隐含分红。`adjust=none` 时现金分红凭空消失,净值曲线在除息日出现假回撤。另外 `receive_dividend(reinvest=True)` 不摊薄 `avg_cost`(:216 注释"视为零成本增量"),导致后续 realized_pnl 高估。

### 修改方案

**文件:`src/djinn/engine/event_engine.py` + `portfolio/account.py`**

1. 主循环 MARKET_OPEN 阶段(:154 解冻之后)插入:

```python
if cfg_adjust_none:  # 由 runner 经 EngineConfig 传入 adjust 信息,见下
    for s, b in bars.items():
        if b is not None and b.dividend > 0 and account.positions.get(s):
            account.receive_dividend(s, D(str(b.dividend)), reinvest=False)
        if b is not None and b.split_ratio not in (0.0, 1.0) and account.positions.get(s):
            account.apply_split(s, D(str(b.split_ratio)))  # 新方法,见 3
```

2. `EngineConfig` 增加 `process_corporate_actions: bool = False`;`cli/runner.py` 的 `build_engine_config()` 在 `cfg.adjust == "none"` 时置 True。
3. `account.py` 新增 `apply_split(symbol, ratio)`:`pos.qty *= ratio; pos.available *= ratio; pos.avg_cost /= ratio`(全部 q_shares/q_money 量化)。
4. 修 `receive_dividend` 的 reinvest 分支摊薄成本:

```python
extra = q_shares(amt / D(price))
# 再投资股数按 0 成本进入 → 摊薄均价(总成本不变、股数增加)
pos.avg_cost = q_money(pos.avg_cost * pos.qty / (pos.qty + extra))
pos.qty = q_shares(pos.qty + extra)
```

### 数据来源
`Bar.dividend`/`split_ratio` 的填充依赖 provider:yfinance `history(actions=True)` 的 Dividends/Splits 列;akshare 日线接口不直接含分红列——检查 `data/providers/akshare.py` 与 `data/adjust.py` 是否在 `adjust=none` 路径填充了这两列;若未填充,CSV/数据库离线源可通过 `dividend`/`split_ratio` 列提供。本项只负责"引擎消费已存在的列",provider 填充若缺失另列任务(可先对 yahoo 路径验证)。

### 测试验证
- `test_dividend_cash_when_unadjusted`:adjust=none、bar 含 dividend=0.5、持仓 1000 股 → 除息日 cash 增加 500,equity 无假回撤(equity_before == equity_after − 500 + 价格自然变动)。
- `test_split_applied`:split_ratio=2 → 股数翻倍、avg_cost 减半、市值不变。
- `test_reinvest_dilutes_cost`:reinvest=True 后 avg_cost == 原成本 × 原股数 / 新股数。
- 回归:adjust=backward(默认)路径零行为变化(开关默认关)。

---

## A11. 成交量上限约束

### 问题
`VolumeShareSlippage`(engine/slippage.py:67-74)只按成交量占比加价,不限制可成交股数——一笔订单可吃掉超过全天成交量任意倍,小盘股回测失真。

### 修改方案

**文件:`src/djinn/engine/constraints.py` + `TradeConstraints` + `config/models.py`**

1. `TradeConstraints` 增加 `max_volume_share: float = 0.0`(0 = 不限制,保持兼容;建议选股回测设 0.1)。
2. `check_constraints()` 在资金校验后追加:

```python
if constraints.max_volume_share > 0 and bar.volume > 0:
    cap = D(str(bar.volume)) * D(str(constraints.max_volume_share))
    if qty > cap:
        qty = floor_shares(cap, lot) if side == "buy" and lot > 1 else q_shares(cap)
        if qty <= 0:
            return CheckResult(False, reason="超过成交量上限")
```

(返回 adjusted_qty=qty 走既有缩减通道;与 A2 的卖出不取整规则一致。)

3. `config/models.py` 的 `CostsConfig` 加 `max_volume_share: float = 0.0`(pydantic,`ge=0, le=1`);`cli/runner.py` 传入 `TradeConstraints`。

### 测试验证
- `test_volume_cap_shrinks_order`:bar.volume=1e6,max_share=0.1,买 50 万股 → adjusted == 10 万(±lot 取整)。
- `test_volume_cap_zero_disabled`:默认 0 时不生效(回归)。
- 配置 YAML 加 `max_volume_share: 0.1` 经 `load_config` 后生效(loader 测试)。

---

## A12. 删除 EventBus 死代码

### 问题
`src/djinn/engine/event_bus.py`(EventBus/EventPriority)定义了完整优先级事件队列,但 `grep` 全 src 零引用——引擎主循环是硬编码顺序。死代码误导读者以为引擎是事件总线架构。

### 修改方案
删除 `src/djinn/engine/event_bus.py`;若 `engine/__init__.py` 有 re-export 一并删除。`events.py` 中的 `Event`/优先级枚举(:11-31)若同样零引用一并删除(保留 `Fill`/`Order`/`Rejection` —— 它们在用了)。

### 测试验证
- `grep -rn "EventBus\|EventPriority" src/ tests/` 输出为空;`pytest` 全绿;mypy --strict 通过。

---

## 验收清单(全部完成后)

1. `ruff check src/djinn tests` / `black --check` / `mypy --strict src/djinn` 全过。
2. `pytest -n auto -m "not network and not slow and not benchmark"` 全绿(含本计划新增 ~25 个用例)。
3. 手工端到端:用 `configs/backtest.example.yaml` 跑一次 NVDA 回测(CLI),确认结果与基线合理;A 股组合配置(portfolio.example.yaml)跑一次 CSI300 选股回测,确认无"不足最小手"异常拒单、无零股残留。

# 计划 D:性能优化

> 目标读者:执行模型。覆盖回测内核热路径、因子引擎、数据层缓存、API 响应。
> 验证方法:除单测外,每项给出基准测量方式。项目已有 `pytest-benchmark`(`--benchmark-only`)。
> 基准场景(全计划共用):
> - **S1**:单标的 5 年日频(NVDA 2020-2024,~1250 交易日);
> - **S2**:CSI300 等权池 3 年(300 标的 × ~750 交易日,union 日历,FactorPortfolio 月调仓);
> - **S3**:signals 型策略(MACrossover)在 100 标的 × 1000 日。

## 总览

| # | 改进点 | 量级 | 预估工作量 |
|---|---|---|---|
| D1 | 默认 on_bar 每日全历史重算 signals → 向量化预计算 | O(T²·N)→O(T·N) | 1.5 天 |
| D2 | PortfolioView.weight() 每次调用重算 equity | O(n²)/日 → O(n) | 0.5 天 |
| D3 | FactorPortfolioStrategy 调仓日全量重算因子 → lookback 截断 | O(T²/freq) | 1 天 |
| D4 | DataView 每日全历史切片 → 视图化/尾部窗口 | O(T²) 复制 | 1 天 |
| D5 | 引擎每日快照稠密矩阵 → 稀疏记录 | 内存/传输 | 0.5 天 |
| D6 | DataCache:LRU 32 可调 + covers() 永久 miss 修复 + TTL | 命中率 | 1 天 |
| D7 | 数据拉取并发(预拉循环线程池) | 墙钟 ÷N | 0.5 天 |
| D8 | _bars_at 双重索引查找 + bar_at dataclass 构造 | 常数项 | 0.5 天 |
| D9 | top_n / DynamicUniverse.symbols_on 线性扫 → bisect | O(T²) | 0.25 天 |
| D10 | replace(0, pd.NA) object dtype 退化 | 常数项 | 0.25 天 |
| D11 | /stocks/{symbol} 4 次 Ticker.info → 1 次复用 | N+1 | 0.5 天 |
| D12 | winsorize/neutralize 逐日 Python 循环优化 | 常数项 | 0.5 天(可选) |

---

## D1. 默认 on_bar 每日全历史重算 signals → 向量化预计算

### 问题
`strategy/base.py:267-279`:默认 `on_bar` 每个交易日对每个标的调用 `self.signals(df)`,其中 `df = ctx.data[symbol]` 是 **≤now 的完整历史切片**;典型 signals 实现内部 `close.rolling(slow).mean()` 等对全历史向量化计算 → 每日 O(T),全回测 **O(T²×N)**。S3 场景下这是绝对主瓶颈(粗估 80%+ 耗时)。

### 修改方案(signals 预计算 + 每日查表)

核心观察:signals-only 策略的 `signals(data)` 是**无状态纯函数**,给定完整历史一次性算出的信号序列,与逐日切片重算的末值完全一致(rolling/shift 类运算只依赖过去)。因此可以**在回测启动时对每个标的预计算一次全量信号**,主循环每日 O(1) 查表。

**文件:`src/djinn/strategy/base.py` + `src/djinn/engine/event_engine.py`**

1. `Strategy` 增加类属性 `precompute_signals: bool = True`(signals-only 策略默认开;`on_bar` 覆写类不受影响)。
2. 引擎 `run()` 启动时(event_engine.py:125 日历对齐之后):

```python
presignals: dict[str, pd.Series] = {}
if (
    getattr(strategy, "precompute_signals", False)
    and type(strategy).on_bar is Strategy.on_bar  # 未覆写 on_bar(走默认适配)
    and strategy.scope == SCOPE_PER_SYMBOL
):
    for sym, md in data.items():
        presignals[sym] = strategy.signals(md.df)  # 一次性全量
```

3. 默认 `on_bar`(base.py:256-279)增加快速路径:

```python
def on_bar(self, ctx):
    ...
    ps = getattr(self, "_presignals", None)  # 引擎注入
    for symbol in ctx.data.symbols:
        if ps is not None and symbol in ps:
            ser = ps[symbol]
            today_sig = int(ser.asof(ctx.now)) if len(ser) else 0  # asof: 当日或之前最近值,防未来函数
        else:
            df = ctx.data[symbol]
            ...  # 现状慢路径
```

(`Series.asof(pd.Timestamp)` 返回 ≤now 的最近非 NaN 值;引擎在 run() 里把 presignals 挂到 `strategy._presignals`。)

4. **防未来函数审查(关键)**:预计算用完整 df 计算,但 rolling/ewm/shift 类指标 t 日值只依赖 ≤t 数据,与慢路径逐日切片末值**数学等价** —— 必须防止用户 signals 里写了依赖未来数据的运算(如 `df.close.shift(-1)` 或全样本统计量 `df.close.mean()`)。对策:
   - 文档(docstring + `docs/add-factor.md` 同级新增 `docs/write-strategy.md` 或在 base.py 模块 docstring)明确:signals 必须是**因果运算**(t 日输出仅依赖 ≤t 输入);
   - 提供自检工具 `strategy/check.py::check_causal(signals_fn, df, n_probe=5)`:随机取 5 个截断点,对比截断序列末值与全量序列对应值,不等则告警"signals 含非因果运算,请用 on_bar"。引擎在 debug 模式(`DJINN_DEBUG=1`)下启动时跑一次。

5. **等价性保险**:引擎配置加 `EngineConfig.verify_presignal: bool = False`;开启时每 N 日(如 50 日)抽样对比快/慢路径信号,不一致则 fallback 慢路径并 warning。默认关(零开销)。

### 测试验证
- `test_presignal_equivalence`:MACrossover 等 3 个内置 signals 策略,同一数据分别跑(1)预计算路径(2)强制慢路径,断言 fills 序列逐笔相等、equity_curve 逐日相等(1e-12)。
- `test_asof_no_lookahead`:构造信号在 T 日突变,断言 T-1 日 asof 取不到 T 值。
- `test_check_causal_detects_shift_neg`:含 `shift(-1)` 的 signals 被 check_causal 检出。
- **基准**:S3 场景 `pytest --benchmark-only` 对比,预期 ≥10× 提速。

---

## D2. PortfolioView.weight() 每次调用重算 equity

### 问题
`strategy/base.py:112-120`:`weight()` 内 `eq = self.equity`(遍历全部持仓);`event_engine.py:211-212` 每日对每个 symbol 调一次;`weights()`(:122-126)同样逐 symbol 调 weight → **O(n²)/日**。S2 场景(300 标的)每日 9 万次内层循环。

### 修改方案
**文件:`src/djinn/strategy/base.py`**

1. `PortfolioView.__init__` 增加惰性缓存:`self._equity_cache: float | None = None`;`equity` property 改为:

```python
@property
def equity(self) -> float:
    if self._equity_cache is None:
        self._equity_cache = float(self._account.equity(self._prices))
    return self._equity_cache
```

(PortfolioView 每交易日由引擎新建,:176 → 缓存天然按日失效,无一致性问题。)

2. `weights()` 改为先算一次 equity 再统一除:

```python
def weights(self) -> dict[str, float]:
    eq = self.equity
    if eq <= 0:
        return {s: 0.0 for s in self._account.positions}
    return {
        s: float(p.qty) * self._prices.get(s, 0.0) / eq
        for s, p in self._account.positions.items() if p.qty > 0
    }
```

3. 引擎 :211-212 的逐 symbol `portfolio_view.weight(s)` 循环改为一次 `w_all = portfolio_view.weights()` 后查表(缺失 symbol 补 0.0)。

### 测试验证
- `test_weights_consistency`:`weights()` 与逐 symbol `weight()` 结果一致;持仓为空/eq=0 边界。
- 基准:S2 场景引擎主循环耗时段(cProfile)中 equity 计算占比从高位降至噪声级;预期整体 1.3~2×。

---

## D3. FactorPortfolioStrategy 调仓日全量重算因子 → lookback 截断

### 问题
`strategy/library/factor_portfolio.py:77-84`:每个调仓日对 ≤now 的**全历史**面板调 `f.compute()`(内部 rolling O(T))→ 全回测 O(T²/freq)。

### 修改方案
1. 每个 Factor 子类声明最大回看窗口:`Factor` ABC(base.py)加 `max_lookback: int = 252`(保守默认);各因子按参数覆写(如 `MomentumFactor` → `self.window + self.skip + 5`;`VolatilityFactor` → `window + 5`;基本面直读类 → 1)。实例属性在 `__init__` 按实际参数算。
2. `FactorPortfolioStrategy` 调仓处:

```python
lb = max(getattr(f, "max_lookback", 252) for f, _w in self.factor_weights_list)
cutoff = now - timedelta(days=int(lb * 1.6) + 30)  # 交易日→日历日放大 + 余量
prices_win = prices.loc[prices.index >= pd.Timestamp(cutoff)]
# 对 prices_win 调 f.compute(...),取末行截面
```

3. 正确性论证:rolling(window=w) 在 t 日的值只依赖 [t−w+1, t];截断 ≥ max_lookback + buffer 时,截断面板的末行与全历史面板的末行**逐值相等**。测试必须证明这一点(见下);buffer 放大系数保证 ewm(alpha=1/14, min_periods=14) 类递归指标 Warm-up 充分(ewm adjust=False 理论上依赖全历史,但 5×halflife 后权重 <3% —— buffer 取 max(5×window, 60) 并把 RSI 类 ewm 因子(如用了 ewm 的自定义因子)的 max_lookback 声明为 window×5)。

### 测试验证
- `test_lookback_truncation_equal`:对每个内置价格类因子,全历史面板末行 vs 截断面板末行 `pd.testing.assert_series_equal`(rtol=1e-9;RSI/ewm 类用 rtol=1e-6 并在文档注明)。
- 基准:S2 场景调仓日耗时 ÷(T/lb);预期整体 3~5×。

---

## D4. DataView 每日全历史切片 → 视图化/尾部窗口

### 问题
`strategy/base.py:54`:`DataView.__getitem__` 缓存 `df.loc[:now]` 全历史切片;每个交易日新建 DataView(引擎 :175)→ O(T²) 复制(750 日 × 300 股 × 平均 375 行 ≈ 8400 万行复制)。

### 修改方案(两阶段:先做低成本高收益的;D1 落地后大部分策略不再访问 DataView)

1. **`.loc[:ts]` 是视图但后续 rolling 等运算会复制** —— 真正的成本在策略把切片当新 df 用。改法:`DataView.__getitem__` 返回 `df.iloc[:self._pos_cache[symbol]+1]`(iloc 行切片同样是视图,但配合 2 的 tail 更关键)。
2. 增加尾部窗口 API:`DataView.tail(symbol, n) -> pd.DataFrame`(返回最近 n 行);`history()` 已有,改文档引导。`Strategy` 层无法强制用户只用 tail,故:
3. **DataView 实例按日复用改为按回测复用 + now 推进**:引擎把 DataView 创建挪出主循环(run 开头创建一次),每日 `data_view._advance(ts_date)`;`__getitem__` 的缓存从"按日切片"改为"始终返回完整 df 的 ≤now 前缀视图"——实际上 `df.loc[:now]` 在 DatetimeIndex 单调时是 O(log n) 的 searchsorted + 视图构造,本身不贵;**真正的复制发生在调用方**。因此本项的核心改动是:
   - `__getitem__` 缓存键从 symbol 改为不需要缓存(每日新建视图,searchsorted 即可):
     ```python
     def __getitem__(self, symbol):
         df = self._datas[symbol].df
         pos = df.index.searchsorted(self._now, side="right")
         return df.iloc[:pos]
     ```
     (消除每日 dict 缓存与切片常驻内存;searchsorted O(log n)。)
4. 收益主要在配合 D1 后:signals 策略不再触碰 DataView;on_bar 策略(TurtleATR 等)若只用尾部窗口,文档引导用 `ctx.data.tail(symbol, n)`。

### 测试验证
- `test_dataview_searchsorted_equal`:逐日 `__getitem__` 结果与旧 `df.loc[:now]` 完全相等(随机 20 日抽查)。
- 内存基准:S2 场景 tracemalloc 峰值下降(切片不再每日缓存全历史副本)。

---

## D5. 引擎每日快照稠密矩阵 → 稀疏记录

### 问题
`event_engine.py:206-214`:每日对**全部 symbols** 记录 positions/weights 快照 → days×symbols 双稠密 DataFrame,300 股池绝大多数 cell 为 0;`weights_curve` 还随报告序列化传输(report_store JSON 体积)。

### 修改方案
1. 引擎侧:只在**持仓变动日**记录非零项。`positions_hist` 改为 `list[dict]` 但 dict 只含当日非零持仓;构造 DataFrame 时 `pd.DataFrame(positions_hist, index=idx).fillna(0.0)` —— 注意 fillna 后仍是稠密 DataFrame(内存相同),但构造速度提升。真正收益在序列化层:
2. `report_store.py` 的 `serialize_report`:positions/weights 面板转 JSON 前做稀疏化 —— 只输出变动行(diff 非零的日期)+ 首行;`rebuild_report` 对称重建(reindex + ffill + fillna(0))。新增字段 `positions_sparse: {"dates": [...], "rows": [{date, values:{sym: qty}}]}`,保留旧字段兼容(或版本号 `v: 2`)。
3. 前端报告的 PositionAreaChart 消费 rebuild 后的完整面板,无感知。

### 测试验证
- `test_sparse_roundtrip`:serialize→rebuild 后面板与原 `pd.testing.assert_frame_equal`;稀疏体积断言(S2 规模 JSON 体积降 ≥70%)。
- API 报告端点 stub 测试不破坏(test_api_alpha.py)。

---

## D6. DataCache:LRU 可调 + covers() 修复 + TTL

### 问题(三件套)
1. `data/cache.py:28`:`_MEMORY_LRU_SIZE = 32` 硬编码 —— 300 标的池预拉即逐出全部,内存层形同虚设。
2. `data/cache.py:163-170` `covers()`:要求 `df.index[-1].date() >= end` —— 请求 end 晚于最近交易日(周末/节假日/盘中)时**永远 miss → 永远重拉打网络**,合并后末日期不变,下次依然 miss。
3. universe 类缓存(spot_a/code_name_sina/industry_map)读取不传 `max_age_days`(`akshare.py:217,270,398`)→ 写一次永久有效;quote 缓存无 staleness 概念。

### 修改方案
**文件:`src/djinn/data/cache.py` + provider 调用点**

1. LRU 容量:`DataCache.__init__(self, cache_dir=..., mem_size: int | None = None)`;默认从环境变量 `DJINN_CACHE_MEM_SIZE` 读(默认 128);`deps.get_cache` 与 CLI 入口可注入。
2. `covers()` 语义修正:

```python
@staticmethod
def covers(df, start, end, *, today: date | None = None) -> bool:
    if df is None or len(df) == 0:
        return False
    today = today or date.today()
    effective_end = min(end, today)  # 请求的 end 晚于今天不可能有数据
    return df.index[0].date() <= start and df.index[-1].date() >= effective_end
```

**进一步(关键)**:end=today 但今天是周末/假期时仍 miss —— provider 层(`akshare.py:128`、`yahoo.py:106` 附近的 miss 分支)在拉取后应做**软命中**:若 `covers()` 为 False 但 `df.index[-1]` 距 end ≤ 7 个自然日(覆盖春节/国庆长假的 A 股取 12 天),视为命中并直接返回缓存(加 `_log.debug`)。在 `DataCache` 加方法 `covers_soft(df, start, end, slack_days=7)` 供 provider 使用。

3. TTL:`get_universe`/`get` 调用点统一传 `max_age_days`:universe 类(spot_a/code_name_sina/industry_map)7 天;quote 保持"覆盖即命中、不覆盖增量合并"(配合 2 的 soft hit,不会天天打网络);基本面 history 30 天。`akshare.py:217,270,398` 三处补上 `max_age_days=7`。

### 测试验证
文件:`tests/unit/test_api_cache.py` / 新增 `tests/unit/test_cache.py`。
- `test_covers_weekend_hit`:缓存末日为周五,请求 end 为周日 → covers_soft True,provider 不打网络(mock ak 断言未调用)。
- `test_covers_gap_miss`:缓存末日在 end 前 30 天 → miss 并增量拉取。
- `test_lru_size_env`:monkeypatch env → DataCache 容量变化,超出逐出。
- `test_universe_ttl`:写入后 mock mtime 为 8 天前 → 读判定过期重拉。

---

## D7. 数据拉取并发

### 问题
`cli/runner.py:399-405`、`cli/sweep.py:183-186`、`factor/engine.py:145-154` 的逐标的串行拉取;指数成分 300+ 首次运行分钟级。

### 修改方案
1. **先决**:E 计划的 DataCache 线程安全(锁 + 原子写)必须先落地 —— 本项依赖该前置,在计划 E 完成后实施。
2. 三处循环统一改为 `concurrent.futures.ThreadPoolExecutor(max_workers=8)`(provider 是 IO 密集,GIL 无害);workers 数经环境变量 `DJINN_FETCH_WORKERS` 可调。
3. provider 限速改线程安全:`_last_request` 加 `threading.Lock`(akshare.py/yahoo.py 的 `_throttle` 内);限速语义变为全局串行化请求节奏(akshare 0.5s/req → 并发收益主要来自 yfinance/多 provider 并行)。
4. sweep 的 API 路径(`api/jobs.py:343` 串行循环 combos)接通 `parallel`:`SweepRequest.parallel` 字段已存在(schemas.py:30)——`run_sweep_job` 读 meta["parallel"],True 时用 ThreadPoolExecutor 跑 `_run_one`(线程而非 joblib 进程:共享 cache、避免进程序列化开销;`_run_one` 纯计算+读缓存,GIL 影响有限,预期 2-4×);进度回调加锁(ProgressCallback 已有锁)。

### 测试验证
- `test_prefetch_concurrent`:mock provider 记录调用,断言全部 symbol 被拉取且结果与串行一致;墙钟 < 串行 1/3(宽松断言防 CI 抖动:只断言完成性,不断言时间;时间用 benchmark 手工验证)。
- sweep parallel:2×2 grid 结果与串行逐值相等。

---

## D8. _bars_at 双重索引查找

### 问题
`event_engine.py:296-297`:每 ts 每 symbol `ts in md.df.index` + `bar_at()`(内部再查一次 + `.loc[ts]` 行抽取 + Bar dataclass 构造)。S2 场景 22.5 万次/回测的双重查找。

### 修改方案
1. 预计算对齐矩阵:run() 启动时为每个 symbol 构建 `ts → iloc` 映射或 `np.searchsorted` 位置数组:

```python
pos_maps: dict[str, dict[pd.Timestamp, int]] = {
    s: {t: i for i, t in enumerate(md.df.index)} for s, md in data.items()
}
```

`_bars_at` 改查 dict(O(1));行抽取用 `md.df.iloc[i]`(比 `.loc[ts]` 快约 2-3 倍)。
2. Bar 构造惰性化:持仓/订单涉及 symbol 才构造 Bar(目前对全部 symbol 构造但只用于撮合+prev_close+prices;可只对 `pending_orders` 涉及 symbol + 持仓 symbol 构造,prices 直接从 df 值取)。简化版:保持全构造但用 iloc(收益已显著),完整惰性化作为可选第二步。

### 测试验证
- `test_bars_at_equivalence`:随机日期抽样对比新旧 `_bars_at` 输出逐字段相等。
- 基准:S2 场景该环节耗时降 ≥50%。

---

## D9. top_n / symbols_on 线性扫 → bisect

### 问题
`screen/scoring.py:109-112` `top_n` 与 `screen/universe_dynamic.py:48-50` `DynamicUniverse.symbols_on` 每次调用线性扫 DatetimeIndex;回测每 bar 调用 → O(T²)。

### 修改方案
两处统一改为 bisect:

```python
from bisect import bisect_right
idx = score_df.index
pos = bisect_right(idx, pd.Timestamp(when))  # idx 单调
row = score_df.iloc[pos - 1] if pos > 0 else None
```

(DatetimeIndex 支持 `<=` 比较;或 `idx.searchsorted(ts, side="right")` 更 idiomatic。)

### 测试验证
- 边界:when 早于首日(返回 None/空)、当日恰有值、between 两天取前值;与原线性扫结果逐一相等(随机 100 日期)。

---

## D10. replace(0, pd.NA) object dtype 退化

### 问题
`factor/library/liquidity.py:29`、`volatility.py:39`:`cap.replace(0.0, pd.NA)` 把 float64 面板转 object dtype,后续除法退化为逐元素 Python。

### 修改方案
两处改为 `.where(cap != 0)`(保留 float64,0→NaN):

```python
turnover = amount_avg / cap.where(cap != 0)
```

### 测试验证
- 因子输出 dtype == float64 断言;输出值与旧实现逐值相等(na 位置一致);微基准断言无 object dtype(`df.dtypes.unique()`)。

---

## D11. /stocks/{symbol} 4 次 Ticker.info → 1 次复用

### 问题
`api/routers/stocks.py:98-100` → `yahoo.py:467,481,506` + snapshot:一次详情请求最多 4 次独立 `yf.Ticker(symbol).info` 网络调用(N+1)。

### 修改方案
1. `yahoo.py` 增加 `_get_info_cached(symbol) -> dict`:进程内 TTL dict({symbol: (ts, info)},TTL 300s + threading.Lock);`get_stock_name`/`get_stock_price`/`get_stock_profile`/`get_fundamentals` 的 info 需求统一走它。
2. `stocks.py` 详情端点改为 orchestrate:先取一次 info,再把 info dict 传给各组装函数(provider 方法加可选 `_info` 内部参数,或路由层直接读 `_get_info_cached` 的结果组装响应)。
3. 顺带修阻塞问题:该端点改 `async def` + `await asyncio.to_thread(...)`(与 E 计划的事件循环卸载同步做)。

### 测试验证
- mock `yf.Ticker` 计数:详情端点一次请求 → info 属性访问 == 1;TTL 内第二次请求 == 1(不增);TTL 过期后 == 2。

---

## D12. winsorize/neutralize 逐日循环优化(可选)

### 问题
`factor/preprocess.py:44,82`:winsorize `df.apply(axis=1)` 逐行 Python 回调;neutralize 逐日 `get_dummies + lstsq`。

### 修改方案
- winsorize(MAD):改为向量化 —— 每日中位数 `df.median(axis=1)`、MAD 用 `(df.sub(med, axis=0)).abs().median(axis=1)`(axis=1 的 median 已是 Cython);clip 用 `df.clip(lo, hi, axis=0)`。全面板无 Python 循环。
- neutralize:行业哑变量跨日复用(行业映射不变时 get_dummies 只做一次,每日只重跑 lstsq);或按行业组向量化 demean(行业中性等价于组内 demean 当无市值项时)。保守做法:保留逐日 lstsq,哑变量预计算。

### 测试验证
- 与旧实现逐值相等(rtol=1e-12);T=1250×N=300 面板 winsorize 耗时降 ≥5×。

---

## 验收清单

1. 全量测试绿;新增等价性测试(D1/D3/D4/D8/D9/D12 的"新旧输出逐值相等"断言是性能改动的安全网,**不可或缺**)。
2. 基准报告:S1/S2/S3 三场景改动前后 `time` 对比写入 PR 描述;S2 目标:总耗时降至基线 30% 以下(D1+D2+D3+D4 叠加)。
3. 内存:S2 场景 tracemalloc 峰值降至基线 60% 以下。

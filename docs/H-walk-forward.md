# 计划 H:Walk-Forward 分析(滚动样本外验证)

> 目标读者:执行模型。本文档自包含,与 A~G 计划同规格:每项给出问题/目标、修改方案(文件/行号/代码)、数据来源(如涉及)、测试验证。
> 总原则:改动后必须 `pytest -n auto -m "not network and not slow and not benchmark"` 全绿;新增行为必须新增测试。
> 项目约定:black line-length 88;mypy --strict;UI 文案/注释中文,标识符英文;Decimal 记账硬性不变量。
> 结论先行:**WFO 不需要新建回测内核。** IS 优化、OOS 评估、多轴参数、后台任务均有现成件直接复用(`cli/sweep.py` / `cli/runner.py` / `api/jobs.py`)。唯一硬缺口在引擎——因子/择时策略需要 lookback 历史才能调仓,引擎现在必须从数据首日开账本,无法"带暖机数据、从窗口起点开仓"。补一个 `EngineConfig.start` 即可,其余是编排。

## 总览

| # | 改进点 | 类型 | 预估工作量 |
|---|---|---|---|
| H1 | 引擎暖机:`EngineConfig.start`(暖机数据可见、账本从指定日开) | 内核 | 0.5 天 |
| H2 | `WalkForwardConfig` 配置模型(窗口几何/网格/目标/门槛) | 配置 | 0.25 天 |
| H3 | IS 优化:复用 sweep 网格 + 暖机透传 | 内核 | 0.5 天 |
| H4 | OOS 评估 + 拼接 walk-forward 净值/指标(编排器) | 内核 | 1 天 |
| H5 | CLI `djinn walk` | CLI | 0.5 天 |
| H6 | API `/walk-forwards` + 后台任务 + 孤儿恢复 | API | 1 天 |
| H7 | 前端 WalkForwardPage | 前端 | 1 天 |
| H8 | 抗过拟合:`min_is_sharpe` 门槛 + `top_k` 部署 | 内核 | 0.5 天 |

**施工顺序**:H1(引擎前置,一切的基础)→ H2(配置)→ H3 → H4(核心编排)→ H5(CLI 壳)→ H9 测试同 H3/H4 一起写。H6/H7(平台接线)可后续单独做;H8 是 H4 的可配置开关,建议落地 H4 时一并接好参数位。

---

## H1. 引擎暖机:`EngineConfig.start`

### 问题

WFO 需要"引擎的数据 dict 覆盖 `[warmup_start, window_end]`(暖机供因子滚动),但账本从窗口起点开、equity 只从窗口起点记录"。现状:

- `engine/event_engine.py:210` 主循环 `for i, ts in enumerate(trading_index)` 从**数据首日**开始建账 → 传暖机数据就会提前开仓,不传暖机数据则窗口首日因子面板为空。
- `factor_portfolio.py:131-140`:`_select_pool` 靠最近 `max_lookback`(≤252,ICIR 时更大)日因子历史打分,数据若从窗口首日给,首个调仓日 `score.dropna()` 后为空 → 空池不交易 → 浪费整个 lookback 期;OOS 更是**开局直接空仓**。
- 择时规则同理(`strategy/timing.py` 的 SMA/ATR 需要 ≥window 个收盘)。

### 修改方案

**1. `engine/event_engine.py:37` 的 `EngineConfig` 加一个字段:**

```python
@dataclass
class EngineConfig:
    ...
    # 账户开账起点:warmup 数据只供因子 lookback,账本/净值从该日起记录(None=数据首日)
    start: date | None = None
```

**2. `engine/event_engine.py:143` 附近,`_aligned_index` 之后过滤交易日索引:**

```python
trading_index = self._aligned_index(data, benchmark)
if cfg.start is not None:
    trading_index = trading_index[trading_index >= pd.Timestamp(cfg.start)]
    if len(trading_index) == 0:
        raise ValueError(f"engine.start={cfg.start} 晚于所有数据区间,无交易日可回测")
```

**关键点:只过滤 `trading_index`,其余不用动。**

- `DataView(data, ts_date)`(`event_engine.py:303`)本就查 `data[sym].df.loc[:ts_date]` **全量历史**,暖机数据自然对策略可见 → 窗口首日即可靠 lookback 调仓。
- `_bars_at` 走预计算的 `pos_maps`(全量 df 的 ts→iloc 映射,`:147-149`),与 `trading_index` 无关,不需要改。
- `_aligned_index`(`:401`)不必包含 warmup 日期——它只负责对齐与循环,不参与取数。
- 过滤后 `equity_hist` / `ts_hist` 长度 == `trading_index` 长度,`:365-370` 的 `pd.Series(equity_hist, index=idx)` 组装自然一致(这正是"只传数据不设 start"会踩的坑:warmup 期多记了账,index 与序列错位)。
- D1 预计算信号路径(`strategy.signals(md.df)`,全量 df)不受影响;信号策略在窗口首日同样能拿到暖机历史。
- 每窗口**新建策略实例**(`build_strategy`),`_bars_seen` 归零 → 窗口首日即首次调仓,用暖机历史算因子,行为正确。

### 测试验证

新增 `tests/unit/test_engine_warmup.py`:

- `EngineConfig(start=X)`:断言 `equity_curve.index[0] == X` 且首值 == `initial_cash`。
- 用一个记录型策略(在 `on_bar` 里 stash `ctx.data[sym]` 的最早日期),断言窗口首日 `ctx.data` 能看到暖机历史(早于 `start`)。
- `start` 早于数据首日(视为 no-op,index 不变)与 `start` 晚于数据末日(`ValueError`)两个边界。

---

## H2. `WalkForwardConfig` 配置模型

### 问题

WFO 需要 IS/OOS 窗口几何 + 参数网格 + 优化目标 + 抗过拟合门槛。`BacktestConfig` 是 `extra="forbid"`(`config/models.py:212`),新增顶层 section 必须建模,否则 YAML 解析直接 400。

### 修改方案

`config/models.py` 新增(放在 `BacktestConfig` 之前):

```python
class WalkForwardConfig(BaseModel):
    """Walk-Forward 分析:滚动样本外验证(period 为全区间,窗口在其内滚动)。"""

    model_config = ConfigDict(extra="forbid")
    is_days: int = Field(default=250, gt=0, description="样本内(训练)窗口,交易日")
    oos_days: int = Field(default=125, gt=0, description="样本外(验证)窗口,交易日")
    step: int | None = Field(default=None, gt=0, description="滚动步长,默认=oos_days(非重叠)")
    n_windows: int | None = Field(default=None, gt=0, description="窗口数上限,默认由区间推导")
    target: str = Field(default="sharpe", description="IS 优化目标(与 sweep 同语义)")
    grid: dict[str, list[Any]] = Field(default_factory=dict, description="参数网格(与 sweep --grid 同格式)")
    top_k: int = Field(default=1, gt=0, description="部署 IS 最优前 k 个组合(1=只部署最优)")
    min_is_sharpe: float | None = Field(default=None, description="IS 目标不达标则该窗口 OOS 空仓(防过拟合)")
    warmup_days: int = Field(default=300, ge=0, description="每窗口前置暖机交易日(≥ max_lookback 最稳)")
```

`BacktestConfig` 挂顶层可选字段:

```python
walk_forward: WalkForwardConfig | None = Field(default=None)
```

**自动兼容**:`loader.py` 的 env 覆盖(`:75` `_apply_env_overrides`)沿 `model_fields` 树遍历,新增 section 自动支持 `DJINN_WALK_FORWARD_IS_DAYS` 等;`dump_config` 自动导出。无需改 loader。

### 测试验证

`tests/unit/test_config.py`:

- YAML 含 `walk_forward` 段 → 正确解析;非法值(`is_days=0` / `step<0` / 未知子键)→ 校验报错。
- `DJINN_WALK_FORWARD_OOS_DAYS=60` env 覆盖生效。
- `dump_config` round-trip 保真。

---

## H3. IS 优化:复用 sweep 网格 + 暖机透传

### 问题

`sweep._run_one`(`cli/sweep.py:128`)是现成的单组合回测:应用轴(`_apply_param`,`:74`)→ 拉数据 → `build_strategy` + `build_engine_config` + `engine.run` → `build_report` → 返回 `{params, target, sharpe/sortino/calmar/...}`。但两处不支持暖机:

- `:147-152` 数据只拉 `[cfg.period.start, cfg.period.end]`,IS 窗口首日因子无 lookback,组合横向可比但**整个 lookback 期不交易**,IS 有效评估期被白白缩短。
- `build_engine_config`(`runner.py:98`)不透传 `start`,引擎从数据首日开账。

### 修改方案

**1. `runner.py:98` 的 `build_engine_config` 加可选 `start`:**

```python
def build_engine_config(cfg: BacktestConfig, *, start: date | None = None) -> EngineConfig:
    ...
    return EngineConfig(
        ...,
        start=start,
    )
```

**2. `runner.py:500` 的 `run_backtest` 加可选 `start`**(OOS 用,见 H4):

```python
def run_backtest(cfg, *, registry=None, csv_dir=None, cache=None,
                 with_attribution=False, should_stop=None,
                 start: date | None = None) -> RunResult:
```

拉数据起点取 `min(cfg.period.start, warmup 起点)`:

```python
fetch_start = start  # start 早于或等于 period.start 时,顺带带上暖机区间
if start is not None:
    fetch_start = min(start, cfg.period.start)
data = {sym: registry.get_ohlcv(sym, fetch_start, cfg.period.end, ...) ...}
```

并把 `start` 透传给 `build_engine_config(cfg, start=start)`。

**3. `sweep.py:128` 的 `_run_one` 加可选 `warmup_start`:**

```python
def _run_one(cfg, registry, params, target="sharpe", *, warmup_start=None):
    ...
    fetch_start = warmup_start or cfg.period.start
    data = {sym: registry.get_ohlcv(sym, fetch_start, cfg.period.end, cfg.adjust, market=market)
            for sym in cfg.universe.symbols}
    engine_cfg = build_engine_config(cfg, start=fetch_start if warmup_start is not None else None)
```

> 注:`DataCache.get` 按 `(provider, symbol, adjust)` 存**全量帧**(`cache.py:139`),`ProviderRegistry.get_ohlcv(sym, start, end, ...)` 拉全量后切片(`provider.py:172`)——WFO 里全区间预拉一次后,各窗口的 `fetch_start` 差异全部是缓存切片,零网络。sweep 的 `run_sweep_job`(`api/jobs.py:420-431`)已示范"预拉全区间 → 各组合缓存命中"。

### 测试验证

`tests/unit/test_sweep_warmup.py`:

- `_run_one(cfg, registry, params, warmup_start=X)`:`report.equity_curve.index[0] == cfg.period.start`(窗口首日即评估,而非暖机日)。
- 断言暖机生效:同一组合,带 warmup 的首笔成交早于不带 warmup 的首笔成交(或前者的 IS 期有效交易日数更多)。

---

## H4. OOS 评估 + 拼接 walk-forward 净值/指标(编排器)

### 问题

无现成编排。需要一个模块把"全区间 → 滚动窗口 → 每窗口 IS 优化 + OOS 评估 → 拼接样本外曲线"串起来。

### 修改方案

新建 `src/djinn/cli/walk_forward.py`:

```python
"""Walk-Forward 分析:滚动样本外验证。"""

@dataclass
class WFWindow:
    no: int
    is_start: date; is_end: date
    oos_start: date; oos_end: date
    best_params: dict[str, Any] | None = None   # IS 最优(未部署为空)
    oos_metrics: dict[str, float] | None = None
    oos_equity: pd.Series | None = None

@dataclass
class WalkForwardReport:
    windows: list[WFWindow]
    equity_curve: pd.Series   # 拼接后的样本外净值(整体归一,index=交易日)
    metrics: Metrics           # 对拼接曲线整体算指标
    target: str
    full_start: date; full_end: date

def _build_windows(trading_days, wf: WalkForwardConfig) -> list[tuple[date,date,date,date]]:
    """在 trading_days 上滚动:IS=[i:i+is],OOS=[i+is:i+is+oos];步长 step(默认 oos)。"""
    step = wf.step or wf.oos_days
    n = (len(trading_days) - wf.is_days - wf.oos_days) // step + 1
    if wf.n_windows: n = min(n, wf.n_windows)
    out = []
    for i in range(n):
        a = i * step
        is_ = trading_days[a : a + wf.is_days]
        oos = trading_days[a + wf.is_days : a + wf.is_days + wf.oos_days]
        out.append((is_[0].date(), is_[-1].date(), oos[0].date(), oos[-1].date()))
    return out

def walk_forward(cfg: BacktestConfig, *, registry=None, grid=None) -> WalkForwardReport:
    wf = cfg.walk_forward
    # 1) 全区间交易日(基准或任一标的日历)+ 预拉全区间数据入缓存
    # 2) combos = _expand_grid(grid or wf.grid)
    segments: list[pd.Series] = []
    for wno, (is_s, is_e, oos_s, oos_e) in enumerate(_build_windows(...)):
        # IS:在 cfg 副本上改 period=[is_s-warmup, is_e],逐个 combo 跑 _run_one(带暖机)
        results = [_run_one(is_cfg, registry, c, wf.target, warmup_start=warmup) for c in combos]
        results.sort(key=_key, reverse=reverse_sort)   # 复用 sweep 的排序/NaN 兜底
        best = results[0]
        # H8:门槛不达标 → 该窗口 OOS 空仓
        if wf.min_is_sharpe is not None and best[wf.target] < wf.min_is_sharpe:
            continue
        # OOS:period=[oos_s, oos_e],run_backtest(start=oos_s) → 净值天然就是 OOS 段
        run = run_backtest(oos_cfg, registry=registry, start=oos_s, with_attribution=False)
        segments.append(run.report.equity_curve)
    equity = _stitch(segments)   # 每段按段首净值归一化后顺序拼接(复利延续)
    metrics = compute_metrics(equity, [], rf=cfg.risk_free_rate, market=...)
    return WalkForwardReport(...)
```

**拼接口径 `_stitch`(核心正确性点):**

- `step == oos_days`(非重叠)时,OOS 段在时间轴上**不相交、连续**。逐段 `seg / seg.iloc[0] * last_value`(首段 `last_value = initial`),把每段收益复利到前段末尾 → 得到整条样本外净值。
- `step < oos_days`(重叠)v1 暂不支持,在 `_build_windows` 里显式拒绝(报错提示"v1 仅支持 step == oos_days")。
- 拼接曲线 index 为各段日期的并集(每窗口重建策略,段内净值自然连续;段与段之间只有边界,无暖机期污染)。

### 测试验证

新增 `tests/unit/test_walk_forward.py`(复用 `test_api_alpha.py` 的确定性 `_StubProvider` 合成序列,不触网):

- 窗口几何:给定全区间 + is/oos/step,`_build_windows` 产出正确个数的窗口,段与段不重叠、相邻段首尾相接。
- IS 最优参数被用于 OOS:用可区分的参数网格(如 fast=10 明显优于 fast=50 的合成序列),断言每个窗口 OOS 段是用 IS top1 的 params 跑的。
- 拼接曲线连续性:`equity_curve` 单调递增/递减规律与合成数据一致;末值与逐段复利手算一致。
- 序列化:`WalkForwardReport` 可 `to_dict()` 且 JSON 友好(每窗口 `oos_metrics` / `best_params` 均为标量)。

---

## H5. CLI `djinn walk`

### 问题

无 CLI 入口。

### 修改方案

`cli/app.py` 注册 `walk_command`(仿 `sweep_command`,`cli/sweep.py:174`):

```bash
djinn walk -c configs/walk.example.yaml --grid '{"fast":[5,10,20],"slow":[20,30,60]}' \
    --is-days 250 --oos-days 125 --target sharpe -o results/walk.json
```

输出:

- 每窗口一行:`# | IS [start~end] | OOS [start~end] | 最优 params | OOS sharpe/mdd/ret`;IS 不达标的窗口标注 `(未部署)`。
- 尾部:拼接后整体 `sharpe / sortino / calmar / annual_return / max_drawdown / n_trades`。
- `-o` 导出 JSON:窗口明细 + 拼接曲线(稀疏化,参考 D5 惯例:日期串 + 值数组)。

新增 `configs/walk.example.yaml`(基于 `configs/sweep.example.yaml` 改造,`strategy.factor_weights` 或裸策略参数二选一示范)。

### 测试验证

`tests/unit/test_walk_cli.py`(或并入 `test_walk_forward.py`):用 stub 数据跑 `walk_forward(...)`,断言打印表格字段齐全、JSON 导出可 `json.loads`。

---

## H6. API `/walk-forwards` + 后台任务 + 孤儿恢复

### 问题

Web 侧无入口。

### 修改方案(完全照抄 sweep 的既有模式)

1. `api/schemas.py`:`WalkForwardRequest`(config + grid + is_days/oos_days/step/target/parallel,字段对齐 H2)。
2. `api/routers/walk_forwards.py`:仿 `routers/sweeps.py` —— `POST /walk-forwards` 校验 grid key(复用 `_validate_grid_keys` 的宽松前缀白名单)→ `registry.create("walk-forward", meta={config, grid, 窗口参数, target, title})` → `background_tasks.add_task(run_walk_forward_job, ...)`;`GET /walk-forwards[/{id}]`。
3. `api/jobs.py`:`run_walk_forward_job` —— **首行从 `result["__meta__"]` 重建输入**(config/grid/窗口参数,CLAUDE.md 的 `__meta__` 约定),任务内部把全区间数据预拉入 `provider_registry`(复用注入的 registry 缓存),每窗口 `walk_forward(...)`;最终 `result={"__meta__": meta, "report": ..., "windows": [...]}`,**必须保留 `__meta__`**。
4. `api/jobs.py:889` 的 `_RUNNERS` 加 `"walk-forward": run_walk_forward_job` → 孤儿恢复自动覆盖。
5. `api/main.py` `include_router(walk_forwards.router)`。
6. `make_title`(`api/jobs.py:34`)加 `kind == "walk-forward"` 分支,标题如"Walk-Forward MACrossover · AAPL · 2020-01-01~2024-12-31 · is=250/oos=125"。

### 测试验证

`tests/unit/test_api_walk.py`:用 `TestClient` + `dependency_overrides` 注入临时 `JobRegistry` + `_StubProvider`(仿 `test_api_alpha.py` 的 `setup_module` 注入模式):

- `POST /walk-forwards` 返回 job;轮询 `GET` 到 `done` 后结果含 `windows` 与拼接曲线,可 JSON。
- 非法 grid key 返回 400。
- 孤儿恢复:造一条 `running` 的 `walk-forward` 记录,`recover_orphaned_jobs` 恢复数 ≥1(注意 `DJINN_TEST=1` 时恢复返回 0,须在非 test 环境或 monkeypatch 下断言)。

---

## H7. 前端 WalkForwardPage

### 问题

无可视化入口。

### 修改方案

- `src/pages/WalkForwardPage.tsx`:表单(全区间沿用回测配置 + is/oos/step + target + grid 文本或图形化轴,可复用 `SweepPage` 的轴行组件)→ `POST /walk-forwards` → 轮询 job。
- 结果展示:每窗口表格(IS 最优 params + OOS 指标 + 未部署标记)+ 拼接净值曲线(复用 `EquityChart`,注意与基准曲线对齐)。
- `api/client.ts` 加 `createWalkForward` / `getWalkForwardJob`;路由在 `router.tsx` 加一页;`src/types/index.ts` **同步**后端 schema(`WalkForwardReport` / `WFWindow`)。

### 测试验证

前端 `tsc -b --noEmit` + `vite build` 过;组件级 `vitest` 若有惯例则补一例(其余计划同规格,按 F 计划惯例)。

---

## H8. 抗过拟合:`min_is_sharpe` 门槛 + `top_k` 部署

### 问题

"IS 网格最优 top1 → OOS"是 WFO 的最小形式,IS 内最高 sharpe 往往过拟合,直接部署风险高。H4 里已留出参数位,本节明确语义:

### 修改方案(并入 H4 实现,不单独改文件)

- `min_is_sharpe`:IS 最优组合的 `target` 值低于阈值 → 该窗口**不部署**(OOS 空仓,等价于"该市场状态策略失效")。避免把噪声里的最优参数硬塞进 OOS。
- `top_k`:`top_k > 1` 时对 IS 前 k 个组合在 OOS 等权平均(或按 IS 指标加权)成一条组合曲线。v1 建议仅支持 `top_k=1`(标注 TODO),门槛开关先行落地。
- 统计对比输出:每窗口同时打印"IS 最优指标 vs OOS 实得指标",供判断过拟合程度;整体报告附 IS 均值 vs OOS 均值落差。

### 测试验证

- `min_is_sharpe` 高于 IS 实际指标时:该窗口 `windows[i].best_params is None`、段数少 1。
- 正常部署窗口:`best_params == IS top1 params`(与 H4 的测试共用)。

---

## 正确性要点(实现时守)

1. **防未来函数天然满足**:IS 用 `[warmup, is_end]`、OOS 用 `[warmup, oos_end]`,暖机只是更早的历史,不含未来;因子 point-in-time(`announce_date`)、ICIR 右移、`t+1` 执行均已由现有链路保证,WFO 不引入新的泄露点。**唯一要盯的是窗口副本的 `period` 别把 OOS 尾部数据带进 IS**(副本只改 start/end,不改数据 dict 的可见范围——见 H1 的 `trading_index` 过滤)。
2. **拼接只能用 OOS 段**:`run_backtest(start=oos_start)` 返回的 `equity_curve` 天然从 OOS 首日起,严禁直接取 IS 段的净值拼进样本外曲线。
3. **每窗口重建策略实例**:`build_strategy` 每窗口新建,`_bars_seen` / 择时状态归零,窗口首日即首次调仓。
4. **stale / 幸存者偏差属已知口径**:窗口内 index 成分以全区间拉取为准(与现有 sweep 一致),成分时变导致的偏差在报告中注明,不做成分重算。
5. **`run_backtest` 改动回归**:`start=None` 时行为与现状完全一致;`build_engine_config` 的 `start` 透传必须走默认 `None`,避免影响普通回测。

## 已知取舍

- **v1 仅支持非重叠窗口(`step == oos_days`)**:重叠窗口的段拼接与统计独立性更复杂,明确拒绝并报错,后续再扩。
- **IS 优化用完整网格全扫描**:与 `djinn sweep` 同语义,不做遗传/贝叶斯优化;需要时接入 `sweep` 的并行路径即可。
- **不做 IS/OOS 段内个股停牌重算、不做成分权重**:维持与主回测一致的行情口径。
- **前端页为独立增强**:内核 + CLI + API 完成后即具备完整 WFO 能力,前端仅是展示层。

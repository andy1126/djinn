# 计划 E:API、数据层与平台化

> 目标读者:执行模型。覆盖:`src/djinn/api/`、`src/djinn/data/`(缓存/provider)、`src/djinn/config/`、部署安全。
> 注意:D6(缓存 covers/LRU/TTL)在性能计划 D 中;本计划的 E1 是 D7(并发拉取)的前置依赖,实施顺序 **E1 → D7**。

## 总览

| # | 改进点 | 类型 | 严重度 | 预估工作量 |
|---|---|---|---|---|
| E1 | DataCache 线程安全(锁 + 原子写 + 单飞) | bug/健壮性 | P0 | 1 天 |
| E2 | 阻塞事件循环的同步调用统一 to_thread | bug | P0 | 1 天 |
| E3 | get_registry 单例化 + provider 限速锁 | 架构 | P1 | 0.5 天 |
| E4 | 任务取消(cancel 端点 + 协作式中断) | 功能 | P1 | 1.5 天 |
| E5 | 并发上限与任务排队 | 功能 | P1 | 1 天 |
| E6 | 结果过期清理(jobs/reports/exports) | 功能 | P1 | 0.5 天 |
| E7 | 孤儿任务恢复修复(limit 漏任务 + 并发上限) | bug | P1 | 0.5 天 |
| E8 | 鉴权 + CORS env 化 + 默认 bind 127.0.0.1 | 安全 | P0(暴露场景) | 1 天 |
| E9 | /data/cache 列表误报 error 修复 | bug | P2 | 0.25 天 |
| E10 | WebSocket 进度推广(sweep/factor-analysis/factor-matrix/screen) | 功能 | P2 | 1 天 |
| E11 | 配置模型修复(currency 联动/loader 严格化/HSI 误判/export 默认) | bug | P1 | 1 天 |
| E12 | JobRegistry 存储优化(写放大/竞态/主键冲突) | 健壮性 | P2 | 0.5 天 |
| E13 | 依赖瘦身(matplotlib/seaborn/plotly → extras) | 工程 | P2 | 0.25 天 |
| E14 | universe 缓存 TTL + yahoo 限速补挂 | 健壮性 | P2 | 0.5 天 |

---

## E1. DataCache 线程安全(锁 + 原子写 + 单飞)

### 问题
多个后台 job 线程共享单例 DataCache(`api/deps.py:27-34` get_cache 是 lru_cache 单例),但:
- `data/cache.py:81-90,111-113,139-141`:内存 LRU(OrderedDict)读写**无锁** —— 并发 `move_to_end`/`popitem` 丢数据或抛 KeyError;
- `cache.py:96`:`df.to_parquet(path)` 非原子 —— 两线程同写一键 → 截断文件(读端靠 try/except 判 miss 兜底,表现为偶发缓存损坏重拉);
- `cache.py:145-160` `merge()`:读-改-写无锁,并发同 symbol 拉取互相覆盖丢区间;
- 无单飞:两个 job 同时拉同一 symbol 重复打网络。

### 修改方案
**文件:`src/djinn/data/cache.py`**

1. **锁**:实例级 `self._mem_lock = threading.RLock()`;`_mem` 的所有读写(move_to_end/popitem/赋值/删除)包在 `with self._mem_lock:` 内。磁盘层用**按键锁**:

```python
self._file_locks: dict[str, threading.Lock] = {}
self._file_locks_guard = threading.Lock()

def _key_lock(self, key: str) -> threading.Lock:
    with self._file_locks_guard:
        return self._file_locks.setdefault(key, threading.Lock())
```

`put/get/merge` 的磁盘读写段包 `with self._key_lock(key):`(merge 的"读旧帧→合并→写回"整体持锁,消除互相覆盖)。

2. **原子写**:

```python
def _atomic_write_parquet(self, df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)  # POSIX 原子 rename
```

3. **单飞(singleflight)**:在 provider 层做更自然(缓存 miss 后的网络拉取段)。给 `ProviderRegistry.get_ohlcv`(`data/provider.py`)加:

```python
self._inflight: dict[tuple, threading.Event] = {}
# miss 时:若是首个拉取线程 → 拉取并 set event;否则 wait(event) 后重读缓存
```

简化实现:把"检查缓存→拉取→写缓存"整体包进按 `(symbol, adjust, freq)` 的键锁(复用 cache 的 `_key_lock` 模式提到 registry),并发同键自然串行、后者直接命中前者写入的缓存 —— 这是最稳妥的"单飞"(牺牲少量并发度,同键请求本就该串行)。

### 测试验证
文件:新建 `tests/unit/test_cache_concurrency.py`。
- `test_concurrent_put_get`:8 线程 × 200 次混合 put/get/merge 随机键 → 无异常;最终 merge 区间完整(同键并发 merge 不丢段:两线程分别 merge [1,5] 与 [6,10],结果应覆盖 [1,10])。
- `test_atomic_write_no_partial`:写入线程 + 读取线程并发,读取端永远不会读到截断 parquet(循环 500 次,读结果要么 miss 要么完整)。
- `test_singleflight`:mock provider 拉取计数 + threading.Barrier 同步两线程同键拉取 → 底层网络函数调用 == 1。
- 回归:test_api_cache.py 全绿。

---

## E2. 阻塞事件循环的同步调用统一 to_thread

### 问题
async handler 里直接跑网络/重计算,阻塞整个事件循环(期间 API 全站无响应):

| 位置 | 问题 |
|---|---|
| `api/routers/data.py:31-34` | `POST /data/fetch` 逐 symbol 同步 get_ohlcv(网络) |
| `api/routers/stocks.py:48-63,76-100` | search/detail 同步 provider 调用(多次 yfinance) |
| `api/routers/universe.py:41-135` | stock-list/index-components/industries 同步网络;行业映射首建分钟级 |
| `api/routers/backtests.py:136` | export 回退路径直接同步 `run_backtest`(对照 :99 report 端点用了 `asyncio.to_thread`,口径不一);:143,151 文件 IO 也在循环里 |
| `api/routers/strategies.py:51-64,93-94`、`indicators.py` 同类 | list/validate 端点在事件循环里 exec 用户代码(compile_user_strategy) |

### 修改方案
统一模式:handler 保持 `async def`,把同步重调用包进 `await asyncio.to_thread(fn, *args, **kwargs)`。逐个文件:

1. `data.py:create_data_fetch`(:18-34)——循环体改 `await asyncio.to_thread(registry.get_ohlcv, sym, start, end, adjust, market)`;进一步可 `asyncio.gather(*[to_thread(...) for ...])` 并发(受 E1 锁与限速约束,安全)。
2. `stocks.py`:`search_stocks`/`get_stock_detail` 整体包 to_thread(与 D11 的 info 复用一起做)。
3. `universe.py`:三个端点的 provider 调用包 to_thread。
4. `backtests.py:export_backtest`(:109-155):回退重跑改 `await asyncio.to_thread(run_backtest, cfg, registry=..., with_attribution=True)`(对齐 :99);`export_csv/export_excel` 调用也包 to_thread。
5. `strategies.py`/`indicators.py`:list 端点中对每个用户策略/指标的编译验证改 `await asyncio.to_thread(compile_user_strategy, ...)`;create/validate 端点同理。

**防回归**:新文件 `tests/unit/test_api_nonblocking.py`——用 `asyncio` 测试客户端并发发两个请求(A=慢 stub provider 的 /stocks/search,B=/health),断言 B 在 A 完成前返回(超时 1s)。TestClient 是同步的,改用 `httpx.ASGITransport` + `pytest.mark.asyncio`(若无 pytest-asyncio 依赖,用 `asyncio.run()` 手写;dev 依赖可加 `pytest-asyncio`)。

### 测试验证
- 上述非阻塞测试;现有 test_api.py/test_api_alpha.py 全绿(接口行为不变,仅调度方式变)。

---

## E3. get_registry 单例化 + provider 限速锁

### 问题
`api/deps.py:32-34`:`get_registry` 无 lru_cache(对照 `get_cache` 有)——每个 HTTP 请求新建 ProviderRegistry 和全部 provider 实例;provider 的 `_last_request` 限速状态随请求销毁 → **跨请求限速失效**(akshare 0.5s/req 形同虚设)。

### 修改方案
1. `deps.py`:

```python
@lru_cache(maxsize=1)
def get_registry(cache: DataCache = Depends(get_cache)) -> ProviderRegistry:
    return default_registry(cache=cache)
```

(lru_cache 缓存的是 Depends 解析后的默认值 —— FastAPI 对带 Depends 参数的 lru_cache 函数:Depends 在请求解析期执行,get_cache 本身单例,故 registry 也单例。验证:`assert get_registry() is get_registry()` 在两个请求上下文。)
注意:`lru_cache` 装饰带参数的函数会以参数值为键;`get_cache` 单例保证键恒定。若担心 FastAPI 依赖交互,改为模块级 `_REGISTRY: ProviderRegistry | None = None` + 惰性初始化 + Lock,语义更显式。
2. `akshare.py`/`yahoo.py` 的 `_throttle`:`self._last_request` 读写包 `self._rate_lock`(实例级 threading.Lock);计算 sleep 时长与更新时间在同一临界区。

### 测试验证
- `test_registry_singleton`:TestClient 连发两请求,dependency_overrides 里断言两次注入同一对象。
- `test_throttle_threadsafe`:4 线程并发调 provider 拉取 20 次,mock 时间断言相邻请求间隔 ≥ rate_limit_sec(误差 10ms)。

---

## E4. 任务取消(cancel 端点 + 协作式中断)

### 修改方案
1. **状态机**:`api/jobs.py` 的 JobRecord.status 增加 `"cancelled"`;`JobRegistry` 增加 `request_cancel(job_id) -> bool`(置 DB 标志位 + 内存 set `self._cancel_flags: set[str]`)。
2. **端点**:`POST /backtests/{job_id}/cancel`、`POST /sweeps/{job_id}/cancel`(其余任务类型同理或统一 `POST /jobs/{job_id}/cancel` —— 推荐统一端点,路由文件新建 `api/routers/jobs.py`)。仅 pending/running 可取消,否则 409。
3. **协作式中断点**(线程不可强杀,只能检查点):
   - `run_backtest_job`:无法中断 `run_backtest` 内部 → 给 `cli/runner.py` 的 `run_backtest(cfg, ..., should_stop: Callable[[], bool] | None = None)`,引擎主循环(event_engine.py:143)每日开头 `if should_stop and should_stop(): raise BacktestCancelled()`;`run_backtest_job` 传 `lambda: registry.is_cancel_requested(job_id)`。
   - `run_sweep_job`(:343 组合循环):每组合前检查。
   - factor-analysis/factor-matrix/screen job:在数据拉取循环与计算循环的检查点。
4. `BacktestCancelled`(utils/exceptions.py 新增)在 runner 层捕获 → job status="cancelled"(非 error);report_store 不落盘;`__meta__` 保留(可重新提交)。
5. 前端:DashboardPage 行内加"取消"按钮(running 时可见);BacktestRunPage 进度区加"停止任务"(F 计划同步)。

### 测试验证
- `test_cancel_backtest`:stub 策略 + 长数据,创建任务后轮询至 running → cancel → 最终 status=="cancelled",无 report 文件;`__meta__` 完好。
- `test_cancel_pending`:pending 任务 cancel → 不进入 running。
- 引擎检查点:should_stop 第 3 日返回 True → 回测在 3 日后抛 BacktestCancelled。

---

## E5. 并发上限与任务排队

### 问题
`BackgroundTasks.add_task` 把任务直接抛进 Starlette 线程池(约 40),无上限无排队;多个大回测并发 → 内存爆炸/OOM。

### 修改方案
1. `api/jobs.py` 新增 `JobScheduler`:

```python
class JobScheduler:
    """进程内 FIFO 调度:最多 N 个任务并发,其余排队。"""
    def __init__(self, registry: JobRegistry, max_concurrent: int | None = None):
        self.max_concurrent = max_concurrent or int(os.environ.get("DJINN_MAX_JOBS", "2"))
        self._sem = threading.Semaphore(self.max_concurrent)
        self._queue: queue.Queue[tuple[str, Callable]] = queue.Queue()
        self._dispatcher_thread: threading.Thread | None = None
```

`submit(kind, job_id, runner)`:入队;dispatcher 线程循环 `queue.get()` → 拿 semaphore → 起线程跑 runner(runner 结束释放 sem)。`max_concurrent` 默认 2(回测吃内存),env 可调。
2. 路由改造:`backtests.py:43`、`sweeps.py:64` 及 alpha 各 create 端点,从 `background_tasks.add_task(run_xxx_job, ...)` 改为 `scheduler.submit("backtest", job.job_id, run_backtest_job)`。scheduler 经 deps 单例注入(`get_scheduler`,lru_cache)。
3. 排队的 job status 保持 pending + stage="排队中";`/backtests` 列表可见。
4. 孤儿恢复(jobs.py:715-747 的 recover_orphaned_jobs)同样走 scheduler.submit(而非直接起线程),顺带获得并发上限(治 E7 的 1000 线程问题)。

### 测试验证
- `test_queue_backpressure`:max_concurrent=1 提交 3 个慢任务 → 任一时刻 running ≤1,其余 pending;全部按提交序完成。
- 取消排队任务(E4 联动):queue 中移除,不占并发额。

---

## E6. 结果过期清理

### 问题
jobs 表、`.cache/djinn_results/*.json`、`.cache/exports/` 只增不减;`report_store.delete`(report_store.py:271)有定义无调用方。

### 修改方案
1. `JobRegistry.purge_older_than(days: int, keep_kinds: tuple = ()) -> int`:删 `updated_at < now - days` 且 status ∈ {done, error, cancelled} 的记录;同步调 `report_store.delete(job_id)` 与 exports 目录对应文件(glob `{job_id}.*`)。
2. 端点:`POST /jobs/purge?days=30`(管理用);并在 lifespan 启动钩子(main.py)每日一次(记录 last_purge 于 meta 表或文件)自动清 `DJINN_JOB_RETENTION_DAYS`(默认 30)之前的任务。
3. 前端 DataManagerPage 或 SettingsPage 加"清理历史任务"按钮(F 计划)。

### 测试验证
- 构造 3 个任务(2 老 1 新)→ purge(days=30) → 老任务记录/report 文件均删,新任务保留;running 老任务**不删**(断言)。

---

## E7. 孤儿任务恢复修复

### 问题
`api/jobs.py:715-747`:① `registry.list(limit=1000)` 只取最新 1000 条,更老的孤儿被静默漏掉;② 恢复无并发上限(N 孤儿=N 线程);③ daemon 线程强杀后再成孤儿,循环恢复无心跳/租约。

### 修改方案
1. `JobRegistry` 增加专用查询 `list_by_status(statuses: list[str]) -> list[JobRecord]`(SQL `WHERE status IN (...)`,无 limit);恢复逻辑改用它。
2. 恢复改走 E5 的 `scheduler.submit`(并发上限自然生效);同 job 幂等:submit 前检查内存 `self._recovered: set` 防重复提交(lifespan 可能重入)。
3. 租约(可选增强):running 任务在 ProgressCallback.update 里刷新 updated_at(现状已是);恢复时只认领 `updated_at < now - 60s` 的 running 任务(防多实例部署误抢)——单机场景直接全认领即可,注释标注。

### 测试验证
- 构造 3 个 running 孤儿(DJINN_TEST 环境外直接调 recover 函数)→ 全部被 scheduler 接管重跑;重复调用 recover 不重复提交;`list_by_status` SQL 单测。

---

## E8. 鉴权 + CORS env 化 + 默认 bind 127.0.0.1

### 问题(安全)
- 全 API 无鉴权;`POST /strategies/user` 是 exec 用户源码入口 —— 任何可达者可 RCE;`DELETE /data/cache` 任何人可清缓存。
- CORS 硬编码 `localhost:5173`(main.py:56-62)。
- `scripts/dev.sh` 与文档的 uvicorn 命令 bind 0.0.0.0。

### 修改方案(最小防线,不做完整用户体系)
1. **可选 Bearer token**:`api/main.py` 加中间件 —— 读 env `DJINN_API_TOKEN`;若设置,所有 `/` 下请求(除 `/health`)要求 `Authorization: Bearer <token>`,不符 401;未设置则放行(单机默认零配置,不破坏现有流程)。
2. **CORS env 化**:`DJINN_CORS_ORIGINS`(逗号分隔,默认 `http://localhost:5173,http://127.0.0.1:5173`)。
3. **bind 默认 127.0.0.1**:`scripts/dev.sh` 的 uvicorn 命令改 `--host 127.0.0.1`(需要局域网访问时显式 `DJINN_HOST=0.0.0.0 ./scripts/dev.sh start`);CLAUDE.md/README 同步。
4. WebSocket:`/backtests/{id}/progress` 同样校验(WS 握手带 `?token=` 或 Authorization header;前端 client.ts subscribeProgress 从 env/配置注入)。
5. 前端:SettingsPage 加"API Token"输入框,存 localStorage,axios 拦截器统一带 Authorization(F 计划)。

### 测试验证
- 设置 token 后:无 header 401、错 token 401、对 token 200、/health 免密;WS 无 token 被拒(连接即关闭)。
- 未设置 token:现有 test_api.py 全绿(零破坏)。

---

## E9. /data/cache 列表误报 error 修复

### 问题
`data/cache.py:179-194` `list_entries`:对所有 parquet 做 `pd.to_datetime(df.index)`;universe 帧(index=symbol 字符串)与基本面帧必抛异常 → 全部显示 `rows=-1, error=True`。

### 修改方案
`list_entries` 按文件命名解析 dtype(键模板 `{provider}::{dtype}::{symbol}::{adjust}`,文件名即键的 sanitize):dtype=="quote" 才做日期索引解析并报告 start/end;universe/fundamental 帧只报 `rows=len(df)` 与 columns 数,start/end 置 None。异常时才标 error。

### 测试验证
- 写入三类帧 → list_entries:quote 有 start/end 日期;universe rows>0 无 error;前端缓存页(DataManagerPage)三类 Tab 均正常显示。

---

## E10. WebSocket 进度推广

### 问题
仅 `/backtests/{id}/progress` 有 WS;sweep/factor-analysis/factor-matrix/screen 只能轮询。

### 修改方案
1. 通用化端点:`api/routers/jobs.py`(E4 新建)加 `WS /jobs/{job_id}/progress`——从 registry.get(job) 读 kind,订阅逻辑与 backtests.py:159-194 相同(抽公共函数 `stream_job_progress(websocket, registry, job_id)` 放 `api/ws.py`,backtests 的 WS 端点改为调用它,旧端点保留兼容)。
2. 前端 `client.ts` 的 `subscribeProgress` 改走 `/jobs/{id}/progress`(新签名带 kind 或自动识别);Sweep/FactorAnalysis/FactorMatrix/Screener 四页把轮询替换为 WS + 断连降级轮询(F 计划统一做)。

### 测试验证
- testclient 的 `websocket_connect("/jobs/{id}/progress")` 对 sweep job 收到进度帧与终态;心跳帧格式一致 `{"type":"heartbeat"}`。

---

## E11. 配置模型修复

### 问题与修改(全部在 `config/models.py` / `config/loader.py`)

1. **currency 联动**(:66):`AccountConfig.currency` 默认 None;`BacktestConfig.model_post_init`(或 runner 构建处):None 时按 `resolved_market()` 映射 `{CN: "CNY", HK: "HKD", US: "USD"}`;显式设置则尊重。
2. **loader 未知顶层键严格化**(loader.py:127-128):当前静默丢弃 → 改为收集后 `raise ConfigError(f"未知顶层配置键: {keys};允许: {sorted(BacktestConfig.model_fields)}")`(env 覆盖键除外)。
3. **resolved_market 的 HSI 误判**(:186-187):纯 index 池默认 CN —— 改为查 `UNIVERSE_INDEX_MAP`(data/universe.py)的指数→市场映射;HSI→HK、SP500/NASDAQ100/DOWJONES→US,查不到再默认 CN。
4. **OutputConfig.export 默认**(:151):默认 `["csv"]` 会在 API 后台任务脏写 ./results —— 默认改 `[]`;CLI 示例 YAML 显式写 `export: ["csv","excel"]`(已有)。
5. **SlippageConfig.type 双值**(:72):"zero"/"none" 并存 → 统一 "zero",loader 加别名迁移("none"→"zero" + DeprecationWarning)。
6. **BacktestConfig 依赖 ScreenCondition**(models.py:14):配置层依赖选股层 —— 把 ScreenCondition 的 pydantic 定义移到 `config/models.py`(或新建 `config/screen_models.py`),screen/screener.py 反向 import,消除分层倒置。
7. **env 覆盖 list 支持**(loader.py:56-87):`DJINN_UNIVERSE_SYMBOLS="AAPL,MSFT"` → 逗号切分;`_coerce` 对 list 字段特判。

### 测试验证
- 每项一个 loader/models 单测(HIS→HK、未知键报错消息含键名、currency 默认 CNY、export 默认空、none→zero 迁移 warning、symbols env 解析);test_config.py 全绿。

---

## E12. JobRegistry 存储优化

### 问题
- 进度写放大:每次 tick 全行 UPDATE(jobs.py:188-200);
- `_notify`(:232-236)锁外遍历 `_subscribers`,与 subscribe/unsubscribe 竞态;
- `create`(:134-160)uuid4 hex[:12] 撞车即 500(无主键冲突重试);
- `JobStatus.result_path`(schemas.py:57)死字段。

### 修改方案
1. 进度节流:ProgressCallback.update 内做 0.5s 最小间隔 + 终态强写(stage 变化必写);result 写入仅终态一次。
2. `_notify` 加锁快照:`with self._lock: cbs = list(self._subscribers.get(rec.job_id, []))`,锁外执行回调。
3. `create` 主键冲突重试 3 次(重新生成 id),仍冲突抛 500。
4. 删 `result_path` 字段及前端镜像(types/index.ts)。

### 测试验证
- 高频 update(1000 次)DB 写入次数显著少于 1000(节流);并发 subscribe/notify 无异常;mock uuid 冲突一次后成功。

---

## E13. 依赖瘦身

### 问题
`matplotlib`/`seaborn`/`plotly` 是主依赖,仅服务 CLI 的 viz/(HTML 报告);Web 用户用 ECharts,pip 安装体积与冷启动受影响。

### 修改方案
`pyproject.toml`:三者移入 `[project.optional-dependencies] viz = [...]`;`viz/plots.py`、`viz/heatmap.py`、`viz/html_report.py` 的 import 全部移到函数内部(延迟加载)+ 友好报错("请 `pip install djinn[viz]`");`io/export.py` 的 openpyxl 同理(可留主依赖,体积小)。`cli/runner.py` 在 `output.report=="html"` 时才触发 viz import。

### 测试验证
- 无 viz extras 环境:`djinn run`(report=none)正常;report=html 报友好错误;装 extras 后 test_viz.py 绿。

---

## E14. universe 缓存 TTL + yahoo 限速补挂

### 问题
- `akshare.py:217,270,398` 的 spot_a/code_name_sina/industry_map 缓存读不传 max_age_days → 永久有效(与 D6-3 同项,此处仅列分工:D 计划改 cache 层,本项改调用点)。
- `yahoo.py:459,473` 的 `get_stock_name`/`get_stock_price` 未调 `_throttle`;rate_limit_sec 默认 0.0(:80)等于不限速。

### 修改方案
1. 三处调用点补 `max_age_days=7`(industry_map 可 30)。
2. yahoo 两个方法入口调 `self._throttle()`;`YahooProvider.__init__` 的 rate_limit_sec 默认改 0.3;重试退避加 jitter:`sleep(base * (0.5 + random.random()))`(random 注入 rng 参数便于测试)。
3. yfiua.github.io 成分 CSV(:372-381):加 1 次重试 + UA header(`urllib.request.Request(url, headers={"User-Agent": "djinn/0.1"})`)。

### 测试验证
- mock urlopen 断言 UA 头与重试;throttle 调用计数;TTL 过期重拉(同 D6 测试,合并写)。

---

## 验收清单

1. 全量测试绿(新增 ~30 用例);`pytest -m "not network"` 与网络套件分开跑。
2. 压测验证:ab/wrk 连续 20 并发 POST /backtests → running 数 ≤ DJINN_MAX_JOBS,其余排队;无 OOM。
3. 安全验证:设置 DJINN_API_TOKEN 后 curl 无 header 全 401;CORS 只允许 env 名单。
4. 文档:CLAUDE.md 的 API 段落同步(单例 registry、scheduler、cancel/purge 端点、token)。

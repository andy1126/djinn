# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What djinn is

djinn 是多市场(A股 / 港股 / 美股)量化回测框架。事件驱动引擎优先(精确撮合 / 滑点 / 费用),向量化引擎为性能补充。
入口有三套,共享同一个回测内核与同一个 `BacktestConfig`:
- **CLI** (`djinn run / sweep / data`) — YAML 配置,可复现,可入版本控制
- **FastAPI 后端** (`src/djinn/api/`) — REST + WebSocket,把内核包成 Web 服务
- **React SPA** (`frontend/`) — 唯一可视化交互入口,Vite + TS + Ant Design + ECharts

## 架构(分层)

```
Data(数据提供器 + 缓存) → Strategy(策略) → Engine(事件驱动引擎)
  → Portfolio(Decimal 账本 / 再平衡 / 风控) → Analytics(指标) → Viz/IO(可视化 + 导出)
```

`cli/runner.py` 的 `build_engine_config()` / `build_strategy()` / `run_backtest()` 是 CLI 与 API 共用的入口;**FastAPI 路由不应另起一套回测逻辑**,必须走 `run_backtest()` 保证一致可复现。

## 精度策略(硬性不变量)

- 技术指标、收益率序列、组合净值曲线 → `float64`(pandas/numpy)
- 现金余额、持仓股数、单笔成交金额、手续费/印花税 → `decimal.Decimal`
- 每个交易日 mark-to-market 时 Decimal→float 入净值序列
- `Account.check_invariant()` 断言 `cash + Σ(market_value) == equity` 始终成立
- **严禁用 float 记现金 / 股数账本**

## 防未来函数

- `DataView` 仅暴露 `<= ctx.now` 的数据切片
- 信号 **t 日生成、t+1 开盘执行**;`engine/event_engine.py` 主循环顺序:解冻 → 撮合昨日挂单 → `strategy.on_bar()` → 再平衡 → 风控过滤 → mark-to-market → 记录

## 策略 API

两种 API,二选一:
- `Strategy` ABC:覆写 `on_bar(ctx)`(复杂 / 组合策略)
- 简单单标的策略覆写 `signals(data) -> pd.Series{1,-1,0}`,由 `SignalAdapter` 转成 `on_bar`(~15 行一个策略)

声明式参数:`fast = param(10, min=2, max=100, description="快速均线")`。`param()` 返回 `_ParamDescriptor`,在 `__init_subclass__` 里收集并校验越界/类型。**关键实现细节**:`_ParamDescriptor._attr()` 必须运行时读 `self.name`(而非 `__init__` 里写死),否则同名参数会互相覆盖——见 `strategy/parameter.py`。
前端 `StrategyParamForm` 由 `/strategies/{name}` 返回的 `param_schema()` 动态生成表单,**新增策略后表单自动出现,无需改前端**。

## 配置

`BacktestConfig`(pydantic v2, `extra="forbid"`)是唯一权威配置模型。
`config/loader.py` 的 `load_config()`:YAML 加载 → `_apply_env_overrides()` 应用 `DJINN_<SECTION>_<FIELD>` env 覆盖(env > yaml > 默认)→ 过滤未知顶层字段 → `model_validate`。
慢复权支持 `forward/backward/none`,默认 **backward**(保证净值连续)。

## FastAPI 后端 (`src/djinn/api/`)

- 长任务(回测/扫描)用 `BackgroundTasks` 后台线程执行,`JobRegistry`(SQLite,`.cache/djinn_jobs.db`)持久化状态 + 进度回调。
- **Job result 的 `__meta__` 约定**:`JobRegistry.create(kind, meta=...)` 把请求元数据(config/grid/target)存入 `result["__meta__"]`。后台任务完成时**必须保留 `__meta__`**(与 summary/results 合并),否则 `/backtests/{id}/report` 与 `/export` 端点拿不到 config 重新生成报告会 400。改 `run_backtest_job` / `run_sweep_job` 的最终 `result={...}` 时务必带上 `meta`。
- WebSocket 进度推送(`/backtests/{id}/progress`):后台线程回调里**只能用 `loop.call_soon_threadsafe(queue.put_nowait, job)`** 投递到事件循环,不能用 `asyncio.create_task`(不在事件循环线程)。
- `data/fetch` 端点:请求体里 `start/end` 是 str,必须在路由内 `date.fromisoformat()` 转成 `date` 再传给 provider,否则底层 `date` vs `str` 比较抛错。
- Dispatcher:`get_job_registry` / `get_cache` 用 `lru_cache` 单例;测试用 `app.dependency_overrides[get_job_registry] = lambda: _test_registry` 注入独立 DB(见 `tests/unit/test_api.py`)。

## React 前端 (`frontend/`)

- TypeScript strict + Vite。`@` → `src/` 路径别名(`vite.config.ts` + `tsconfig.json`)。
- **只能有一套 router**:`App.tsx` 用 `<RouterProvider router={router}>`(`router.tsx` 的 `createBrowserRouter`),`main.tsx` **不要**再包 `<BrowserRouter>`——两个 history 实例冲突会导致白屏。
- Dev server `/api` 代理到 `http://localhost:8000`(`vite.config.ts` proxy);WebSocket 直连后端 8000(不走 vite ws proxy)。
- 状态:全局配置用 `store/configStore.ts`(Zustand);异步数据用 TanStack Query;后端类型镜像在 `src/types/index.ts`——**后端 schema 改动时同步更新该文件**(尤其注意 `metrics` 在内核 Metrics dataclass 里只有 return/sharpe/drawdown 类字段,`alpha/beta/excess_return` 在 `benchmark_stats` 里,不在 `metrics`)。
- 9 个页面见 `src/pages/`,ECharts 组件在 `src/components/charts/`。

## 命令

```bash
# 安装(内核 + 开发工具)
uv pip install -e ".[dev]"

# 后端
ruff check src/djinn tests --fix
black src/djinn tests
mypy --strict src/djinn
pytest -n auto                          # 全部
pytest tests/unit/test_api.py -v        # 单文件
pytest tests/unit/test_config.py::test_load_example_yaml -v  # 单测
python -m uvicorn djinn.api.main:app --host 0.0.0.0 --port 8000  # 启 API

# 前端
cd frontend && npm install
./node_modules/.bin/vite --port 5173        # dev server (HMR, 代理 /api → 8000)
./node_modules/.bin/tsc -b --noEmit        # 类型检查
./node_modules/.bin/vite build             # 生产构建

# CLI
djinn run -c configs/backtest.example.yaml --csv-dir <CSV目录>
djinn sweep -c configs/sweep.example.yaml
djinn data fetch -c configs/backtest.example.yaml
```

## 约定

- **Python 3.13+**,black line-length **88**,ruff,mypy **--strict**
- **语言**:UI 文案 / 注释用中文,代码标识符用英文
- **提交**:`type: description`(feat / fix / chore / refactor / test / docs)
- **数据源**:AkShare 免费免 key 作 A/港股默认,Tushare 需 token 作高质量补充,`yfinance` 作美股,本地 CSV 作离线/测试
- **缓存**:Parquet(`.cache/djinn/`)+ 内存 LRU,key = `(provider, symbol, adjust, freq)`
- **yfinance 易发网络抖动**:失败先查缓存是否已命中,provider 重试逻辑见 `data/providers/yahoo.py`;不要因一次性 yfinance 失败判定代码有 bug。

## 测试

`tests/unit/test_api.py` 用 `fastapi.testclient.TestClient` 直连 `app`,不依赖外部 uvicorn 进程;通过 `dependency_overrides` 注入临时 `JobRegistry`(`.cache/test_jobs.db`)避免污染真实 job 库。新增端点务必在此加测。
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What djinn is

djinn 是多市场(A股 / 港股 / 美股)量化**选股平台**。事件驱动引擎优先(精确撮合 / 滑点 / 费用),向量化引擎为性能补充。横截面 alpha 层(因子引擎 / 因子分析 / 选股策略 / 归因)已全部接入 Web 与 CLI。
入口有三套,共享同一个回测内核与同一个 `BacktestConfig`:
- **CLI** (`djinn run / sweep / data`) — YAML 配置,可复现,可入版本控制
- **FastAPI 后端** (`src/djinn/api/`) — REST + WebSocket,把内核与 alpha 层包成 Web 服务
- **React SPA** (`frontend/`) — 唯一可视化交互入口,Vite + TS + Ant Design + ECharts

## 架构(分层)

```
Data(数据提供器 + 缓存 + 基本面/universe) → Factor(因子引擎/分析)
  → Strategy(策略:择时 + 选股 TopN 组合) → Engine(事件驱动引擎)
  → Portfolio(Decimal 账本 / 再平衡 / 风控) → Analytics(指标 + Brinson/因子归因) → Viz/IO
```

`cli/runner.py` 的 `build_engine_config()` / `build_strategy()` / `run_backtest()` 是 CLI 与 API 共用的入口;**FastAPI 路由不应另起一套回测逻辑**,必须走 `run_backtest()` 保证一致可复现。选股(`strategy.scope == "portfolio"` 或 `FactorPortfolio`)策略由策略自行再平衡,引擎侧用并集日历(`calendar="union"`)且不再注入调仓单。

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

## 横截面 alpha 层(因子 / 选股 / 归因)

因子 → 分析 → 选股 → 归因 闭环,全部 point-in-time、无未来函数。

- **因子引擎**(`src/djinn/factor/`):`Factor` ABC 复用 `strategy/parameter.py` 的 `param()` 声明式参数,`compute(prices, ohlcv, fundamentals) -> pd.DataFrame(date×symbol)`;`FactorEngine` 拼 ohlcv 宽表 + point-in-time 基本面面板;`FACTOR_REGISTRY` 仿 `strategy/library`,因子按类目分文件。`make_factor(name, **params)` 实例化,`param_schema(cls)` 出表单(前端 `FactorAnalysisPage` 用)。新增因子后 `/factors` 自动出现,无需改前端。
- **预处理**(`factor/preprocess.py`):winsorize(mad/sigma)/ standardize(zscore/rank)/ neutralize(行业哑变量 + ln(mktcap),`numpy.linalg.lstsq` 取残差)。
- **因子分析**(`analytics/factor_analysis.py`):`analyze_factor(factor, fwd_returns, ...) -> FactorReport`(`IC_mean/icir/ic_pos_ratio`、分层、衰减);`compute_forward_returns(prices, periods)`。IC 用 `pandas.corr(method='spearman')`,无需 scipy。
- **多因子诊断**(`factor/analysis/matrix.py`):`analyze_factor_matrix(factors: dict[str, DataFrame], prices, periods, ic_method) -> FactorMatrixReport`(相关矩阵 px×px 逐日截面相关跨日均值 + 每因子各期 `ic_summary` + `rank_turnover` 换手)。诊断"因子是否冗余"(ep/sp/bp 常 >0.8),**不是 IC 矩阵**——IC 是"因子 vs 前向收益",两因子间无此概念。配 `/factor-matrix` 端点 + 前端 `FactorMatrixPage`(热力图 `MatrixHeatmap` + IC 汇总表)。
- **选股策略**(`strategy/library/factor_portfolio.py`):多因子打分(`FactorScore(factor, weight)`,负权重 = 越低越好)→ TopN 组合。`FactorPortfolioStrategy` 自行再平衡(走 `SCOPE_PORTFOLIO`),所需 `scores`/`cov` 由策略在调仓时传给 `Allocation`(score/risk_parity/min_variance/mean_variance 不再依赖引擎注入)。
  - **两层择时**(`strategy/library/factor_timing.py` 的 `FactorTiming`,继承 `FactorPortfolioStrategy`):调仓频因子选池 + 日频择时覆盖(市场闸门 / 个股出场 / 入场确认),规则库在 `strategy/timing.py`。配置经 `strategy.selection` / `strategy.timing`(`config/models.py`),`cli/runner.py` 的 `build_strategy()` 按 timing 是否存在选类。`Context.benchmark_close()` 提供基准通道(引擎把 benchmark 注入 ctx,G6)。
  - **ICIR 加权**:`strategy.weighting="icir"`(默认 static)用滚动 ICIR 自动加权(因子负向自动取负权);IC 序列右移 holding_period 防未来函数(`factor/composite.py`)。
- **截面选股**(`src/djinn/screen/`):`Screener.apply(conditions, fundamentals_df)` 条件过滤 + `FactorScore` 打分 + `top_n`,供 `/screens` 端点与选股策略共用。
- **归因**(`src/djinn/analytics/attribution.py`):`brinson_attribution(weights, bench, returns, industry_map) -> BrinsonResult`(三效应恒等式 = 超额收益,等权篮子作基准);`factor_attribution` / `build_exposure_report`(因子暴露 + 行业分布)。
- **归因接线**(`cli/runner.py`):`run_backtest(..., with_attribution=True?)` → `_attach_attribution` 填充 `report.attribution` / `report.factor_exposure`。CLI 默认关闭(避免行业映射网络开销),Web 报告端点(`/backtests/{id}/report`)显式开启。
- **精度**:指标 / 收益率 / 净值曲线 = `float64`;Brinson / IC 同上。基本面因子值用 float(仅账本仍用 Decimal)。
- **防未来函数**:基本面按 `announce_date` 生效(asof 对齐),截面 t 日只用 `announce_date <= t` 的最新一期。

## 配置

`BacktestConfig`(pydantic v2, `extra="forbid"`)是唯一权威配置模型。
`config/loader.py` 的 `load_config()`:YAML 加载 → `_apply_env_overrides()` 应用 `DJINN_<SECTION>_<FIELD>` env 覆盖(env > yaml > 默认)→ 过滤未知顶层字段 → `model_validate`。
慢复权支持 `forward/backward/none`,默认 **backward**(保证净值连续)。

## FastAPI 后端 (`src/djinn/api/`)

- 长任务(回测/扫描)用 `BackgroundTasks` 后台线程执行,`JobRegistry`(SQLite,`.cache/djinn_jobs.db`)持久化状态 + 进度回调。
- **Job result 的 `__meta__` 约定**:`JobRegistry.create(kind, meta=...)` 把请求元数据(config/grid/target)存入 `result["__meta__"]`。后台任务完成时**必须保留 `__meta__`**(与 summary/results 合并),否则 `/backtests/{id}/report` 与 `/export` 端点拿不到 config 重新生成报告会 400。改 `run_backtest_job` / `run_sweep_job` 的最终 `result={...}` 时务必带上 `meta`。
- **孤儿任务恢复**(`api/jobs.py` 的 `recover_orphaned_jobs`):长任务在进程内后台线程执行,进程重启会杀线程,只留 `running`/`pending` 快照。`main.py` 的 `lifespan` 启动钩子扫描并重新提交续跑。**关键依赖 `__meta__` 约定**:每个 runner 首行从 `result["__meta__"]` 重建输入,故恢复只需 `(registry, job_id)`。新增长任务 runner 时**必须**:(1) 从 `__meta__` 自恢复输入,(2) 把 kind 加进 `_RUNNERS` 分发表。测试隔离:`DJINN_TEST=1` 时恢复返回 0,`TestClient` 不用 `with` 不触发 lifespan,双保险。
- WebSocket 进度推送(`/backtests/{id}/progress`):后台线程回调里**只能用 `loop.call_soon_threadsafe(queue.put_nowait, job)`** 投递到事件循环,不能用 `asyncio.create_task`(不在事件循环线程)。
- `data/fetch` 端点:请求体里 `start/end` 是 str,必须在路由内 `date.fromisoformat()` 转成 `date` 再传给 provider,否则底层 `date` vs `str` 比较抛错。
- Dispatcher:`get_job_registry` / `get_cache` 用 `lru_cache` 单例;`get_registry(cache=Depends(get_cache))` 注入 `ProviderRegistry`(universe/factor/screen 端点用)。测试用 `app.dependency_overrides[...]` 注入独立 `JobRegistry` + stub `ProviderRegistry`(见 `tests/unit/test_api.py` / `test_api_alpha.py`)。
- **路由**(`api/routers/`):`backtests` / `data` / `strategies` / `sweeps` + alpha `universe` / `factors` / `factor-analysis` / `factor-matrix` / `screens`。后台任务执行器在 `api/jobs.py`:`run_backtest_job` / `run_sweep_job` / `run_factor_analysis_job` / `run_factor_matrix_job` / `run_screen_job`(共享 `_index_components` / `_resolve_universe` / `_industry_map` / `_build_fundamental_panels`)——**全部保留 `__meta__`**。
- **回测报告缓存**(`api/report_store.py`):`run_backtest_job` 跑完即 `save(job_id, serialize_report(report))` 落盘 `.cache/djinn_results/{job_id}.json`;`/backtests/{id}/report` 与 `/export` **先读缓存**,无缓存才回退重跑 `run_backtest(cfg, registry=provider_registry, with_attribution=True)` 并落盘。`serialize_report` / `rebuild_report` 是一对对称序列化(metrics/trade_stats 等经 `_DictLike` 暴露 `to_dict()`,trades/rejections 用 `_Plain` 供 `export_csv` 按 getattr 读)——`/export` 复用 `export_csv/excel` 无需改写。`create_backtest` / `/report` / `/export` 都注入 `provider_registry = Depends(get_registry)` 并一路传给 job / 回退重跑(否则测试 stub 不生效、会触网)。
- **sweep 多轴**(`cli/sweep.py`):`_run_one` 经 `_apply_param` 写轴——`universe.index`(同时重解析成分股)/ `strategy.factor_weights` / `strategy.weighting`(static/icir)/ `portfolio.allocation` / `strategy.n_stocks` / `strategy.rebalance_freq`,其余裸 key 兜底进 `strategy.params`。白名单 `ALLOWED_SWEEP_AXES`(前后端共享,前端硬编码同表);路由 `create_sweep` 拦明显非法的 `<prefix>.<x>` key(400 + 允许列表,裸策略参数放行)。`run_sweep_job` 预拉 `universe.symbols ∪ 所有扫到的 index 成分`,返回每组合 `config_summary` + `sharpe/sortino/calmar`。排序:`REVERSE_MIN_TARGETS = {volatility, annual_volatility}` 升序,其余降序——**`max_drawdown` 存为 ≤0 负值,值越大(越接近 0)越好,走默认降序**,误放升序会把最深回撤排到最前。
- **归因报告序列化**:`Report.attribution` / `factor_exposure` 是 `dict[str, Any] | None`;`/backtests/{id}/report` 重新跑 `run_backtest(cfg, with_attribution=True)` 生成。Series→`{"index":[str],"values":[float]}`,DataFrame→`{"index":[str],"columns":[str],"data":[[float]]}`;NaN/Inf 在 `_sanitize` / `_safe_float` 转成 None(JSON 不接受 NaN/Inf)。
- `export_backtest` 返回 `dict | FileResponse` 时**必须** `response_model=None` 装饰参数(FastAPI 不支持 Union 作 response_model)。

## React 前端 (`frontend/`)

- TypeScript strict + Vite。`@` → `src/` 路径别名(`vite.config.ts` + `tsconfig.json`)。
- **只能有一套 router**:`App.tsx` 用 `<RouterProvider router={router}>`(`router.tsx` 的 `createBrowserRouter`),`main.tsx` **不要**再包 `<BrowserRouter>`——两个 history 实例冲突会导致白屏。
- Dev server `/api` 代理到 `http://localhost:8000`(`vite.config.ts` proxy);WebSocket 直连后端 8000(不走 vite ws proxy)。
- 状态:全局配置用 `store/configStore.ts`(Zustand);异步数据用 TanStack Query;后端类型镜像在 `src/types/index.ts`——**后端 schema 改动时同步更新该文件**(尤其注意 `metrics` 在内核 Metrics dataclass 里只有 return/sharpe/drawdown 类字段,`alpha/beta/excess_return` 在 `benchmark_stats` 里,不在 `metrics`;归因走 `BacktestReport.attribution` / `factor_exposure`)。
- 13 个页面见 `src/pages/`(回测创建/结果/详情/复盘、扫描、数据缓存、因子分析、多因子诊断、选股、股票池、策略、关于、首页)。ECharts 组件在 `src/components/charts/`:净值/回撤/持仓热力/MRD 饼 + alpha(IC 柱/分层曲线/因子分布/行业饼/Brinson 三效应堆叠柱/`MatrixHeatmap` 泛用矩阵热力)。
- **alpha 页面**:`UniversePage`(股票池/指数成分/行业分布三视图,Segmented 切换)、`FactorAnalysisPage`(单因子 + 动态参数表单由 `param_schema` 生成 + IC/分层/衰减)、`FactorMatrixPage`(2~8 因子 + 权重/方向/参数行 → `MatrixHeatmap` 相关矩阵 + 每因子 IC 汇总表)、`ScreenerPage`(动态条件行 + 因子打分行 + `top_n`)。前端 API wrapper 在 `api/client.ts`(`listFactors`/`getFactor`/`createFactorAnalysis`/`createFactorMatrix`/`getFactorMatrixJob`/`getFactorMatrixReport`/`createScreen`/`getScreenJob`/`getStockList`/`listIndexes`/`getIndexComponents`/`getIndustries`)。
- **`SweepPage` 多轴**:`Segmented` 切"图形化 / 文本";图形化每行一个轴(下拉选自硬编码白名单,与后端 `ALLOWED_SWEEP_AXES` 同步,`portfolio.allocation`/`universe.index` 用多选),文本形兼容旧 `name:v1,v2` 逐行。结果表列出每组合 `config_summary`(权重法/n_stocks/index/标的数)+ `sharpe/sortino/calmar` + 总收益/回撤/交易数,排序方向提示与后端 `REVERSE_MIN_TARGETS` 同步(仅 volatility 类升序,max_drawdown 走降序)。
- **`FactorAnalysisPage` 表单**:与 `StrategyParamForm` API 不同(因子 schema 字段名/结构差异),用本地 paramWidget,不要硬塞 `StrategyParamForm`。

## 命令

```bash
# 安装(内核 + 开发工具)
uv pip install -e ".[dev]"

# 开发环境一键启停前后端(后端 --reload + 前端 Vite)
./scripts/dev.sh start|stop|restart|status

# 后端
ruff check src/djinn tests --fix
black src/djinn tests
mypy --strict src/djinn
pytest -n auto                          # 全部
pytest tests/unit/test_api.py -v        # 单文件
pytest tests/unit/test_config.py::test_load_example_yaml -v  # 单测
python -m uvicorn djinn.api.main:app --host 127.0.0.1 --port 8000 --reload  # 启 API(开发热重载;E8 默认仅本机,需局域网访问显式 --host 0.0.0.0)

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
- **A 股数据源换用新浪**(`data/providers/akshare.py`):**东财(eastmoney)接口在当前网络不可达**(实测 `push2.eastmoney.com` `RemoteDisconnected`),故 A 股**行情**(`stock_zh_a_hist`→`stock_zh_a_daily`)、**全 A 股代码/名称搜索列表**(`stock_zh_a_spot_em`→`stock_info_a_code_name`)已切新浪源。新浪源:行情符号加 `sh/sz/bj` 前缀(`_sina_symbol`),列名已是英文(直接过 `_normalize`),复权 `qfq/hfq/''` 与东财一致。**估值/最新价**(`get_fundamentals`/`get_stock_price`)仍走东财 `_spot_df`(新浪无估值字段),当前不可达则降级为 null——前端详情页 A 股估值/价格显示 —。指数成分(`index_stock_cons`)与财务指标(`stock_financial_analysis_indicator`)本就是新浪源,不受影响。行业映射(东财 `stock_board_industry_*`)无替代源,保持现状。
- **指数成分**(`get_index_components`):A 股宽基走 akshare(`index_stock_cons`);HSI / SP500 / NASDAQ100 / DOWJONES 走 `yahoo.py` 的 `get_index_components`,从 **yfiua.github.io 免费 CSV**(`index-constituents` 仓库,`constituents-{index.lower()}.csv`)读**成分清单**(`Symbol, Name` 两列,**无权重**)。**yfinance 无指数成分接口**(实测 `Ticker.info`/`funds_data`/`.components` 均无,源码零命中),yfiua 是 yfinance 生态对"Yahoo 不提供成分"的社区补全,符号即 Yahoo 原生格式(`0101.HK` / `BRK.B`)。两条链路:yfiua 只提供"有哪些股票"的符号清单,价格数据仍走 `YahooProvider.get_ohlcv` → `yf.Ticker().history()`(即项目一直在用的 yahoo 请求工具)。CSV 无权重,HSI/SP500 成分按等权对待(与 akshare 的 A 股一致)。缓存键 `index_cons_{index.lower()}`,复用 `cache.put_universe/get_universe`。**成分名称**:`get_index_component_names` 从同源数据取 symbol→名称映射(yfiua `Name` 列 / akshare `品种名称` 列),缓存帧同时存 `symbol` + `name` 两列(旧缓存缺 `name` 列自动视为 miss 重拉自愈);`/universe/index-components/{index}` 返回与 `symbols` 位置对齐的 `names`。
- **缓存**:Parquet(`.cache/djinn/`)+ 内存 LRU,key = `(provider, symbol, adjust, freq)`
- **yfinance 易发网络抖动**:失败先查缓存是否已命中,provider 重试逻辑见 `data/providers/yahoo.py`;不要因一次性 yfinance 失败判定代码有 bug。
- **美股带点代码转连字符**:yfinance 对 `BRK.B` / `BF.B` 抛 delisted,需转 `BRK-B`;`YahooProvider._yf_symbol` 处理,`.HK` / A 股后缀守卫不误改写(见 `data/providers/yahoo.py`)。

## 测试

`tests/unit/test_api.py` 用 `fastapi.testclient.TestClient` 直连 `app`,不依赖外部 uvicorn 进程;通过 `dependency_overrides` 注入临时 `JobRegistry`(`.cache/test_jobs.db`)避免污染真实 job 库。新增端点务必在此加测。

`tests/unit/test_api_alpha.py` 覆盖 alpha 路由(`/universe` / `/factors` / `/factor-analysis` / `/factor-matrix` / `/screens` / 归因 / 回测报告缓存 / sweep 多轴),用 **确定性 stub provider**(`_StubProvider(DataProvider)`:`_code_num` 符号 → ord 和、线性上扬斜率合成 OHLCV)在 `setup_module` 注入 `get_registry` + `get_job_registry` override,**不触网**。注意 `test_api.py` 的 `teardown_module` 会清 override,跨文件的 alpha 套件必须在自身 `setup_module` 重新注入。`tests/unit/test_factor_matrix.py` 纯单测 `analyze_factor_matrix`(相关对角=1 / 高相关对 >0.8 / 独立对 ~0 / `to_dict` 可 JSON)。`tests/unit/test_attribution.py` 覆盖 Brinson 恒等式 / 因子归因恒等式 / runner 归因接线(`_ohlcv_from_data` / `_attribution_payloads`)。

跑测试避开外部网络:加 `-m "not network and not slow and not benchmark"`(akshare/yfinance 真实拉取标 `@pytest.mark.network`,网络抖动会让这 2-3 个 flaky,非代码 bug)。
# Djinn

**多市场(A股 / 港股 / 美股)量化选股平台**。

以事件驱动回测内核为核心(精确撮合、滑点、费用),向上提供完整的**横截面 alpha 层**(因子引擎 → 因子分析 → 选股 → 归因),并同时交付三套入口,共享同一个回测内核与同一个 `BacktestConfig`:

- **CLI**(`djinn run / sweep / data`)— YAML 配置,可复现、可入版本控制;
- **FastAPI 后端**(`src/djinn/api/`)— REST + WebSocket,把内核与 alpha 层包成 Web 服务;
- **React SPA**(`frontend/`)— 唯一可视化交互入口(Vite + TypeScript + Ant Design + ECharts)。

## 特性

### 回测内核(事件驱动)

- 单标的与自定义组合(多标的)回测,支持交集 / 并集日历;
- 精确撮合:滑点、佣金、印花税(A 股卖出单边 / 港股双边)、过户费(仅沪市);
- 交易约束:涨跌停、停牌续挂、最小手(A 股买入整手 / 卖出零股)、T+1、成交量上限;
- 退市强制平仓、`adjust=none` 下的分红 / 拆股公司行为处理;
- **精度硬性不变量**:技术指标 / 收益率 / 净值用 `float64`,现金 / 股数 / 费用账本用 `decimal.Decimal`;
- **防未来函数**:信号 `t` 日生成、`t+1` 开盘执行,`DataView` 只暴露 `<= now` 的数据切片。

### 横截面 alpha 层(因子 / 选股 / 归因)

- **因子引擎**:24 个内置因子,`param()` 声明式参数,`FACTOR_REGISTRY` 按类目注册;
- **因子分析**:IC / ICIR / Newey-West t 值 / 分层 / IC 衰减 / 换手,附调仓频率推荐;
- **多因子诊断**:相关矩阵 + 每因子 IC 汇总 + Fama-MacBeth 因子收益;
- **预处理**:winsorize(MAD/σ)、standardize(zscore/rank)、neutralize(行业 + 市值)、Schmidt 正交化;
- **选股**:条件筛选 + 多因子打分 TopN,`FactorPortfolioStrategy` 自行再平衡;
- **归因**:Brinson 行业归因(配置 / 选股 / 交互)+ 因子暴露报告。

### 策略

- 内置 19 个策略(双均线、MACD、RSI 反转、Supertrend、动量、Turtle、网格、DCA、VolTarget、`FactorPortfolio` 选股、`FactorTiming` 择时、`AdaptiveTrendTrail` 等);
- **通用 `SignalStrategy`**:把「OHLCV → 稀疏信号」的指标函数注册后即可接成策略,免每指标写一个策略类;
- **Pine Script 转译**:`pine_to_python` 把 Pine(受支持子集)转成 djinn 策略;
- **用户自定义**:策略 / 指标经 AST 白名单 + 受限内建沙箱动态编译,无需改框架代码。

## 安装

```bash
# Python 3.13+
uv pip install -e ".[dev]"          # 内核 + 开发工具(测试 / lint / mypy / 前端类型)
uv pip install -e ".[web]"          # 额外:FastAPI 后端
uv pip install -e ".[akshare]"      # 额外:A股/港股 AkShare 数据源(免费免 key)
uv pip install -e ".[tushare]"      # 额外:A股 Tushare 数据源(需 token)
uv pip install -e ".[viz]"          # 额外:静态图 / HTML 报告(matplotlib/seaborn/plotly)
```

## 快速开始

### CLI

```bash
# 校验 / 导出配置
djinn show-config -c configs/backtest.example.yaml

# 回测
djinn run -c configs/backtest.example.yaml --csv-dir <CSV目录>

# 参数扫描
djinn sweep -c configs/sweep.example.yaml
```

结果输出到 `results/`:指标摘要、交易 / 持仓 CSV、HTML 报告。

### Web(后端 + 前端)

```bash
# 一键启动前后端(后端 --reload + 前端 Vite)
./scripts/dev.sh start

# 或分别启动
python -m uvicorn djinn.api.main:app --host 127.0.0.1 --port 8000 --reload
cd frontend && npm install && npm run dev        # http://localhost:5173
```

前端 dev server 把 `/api` 代理到 `http://localhost:8000`,WebSocket 直连后端。

## 核心概念

- **`BacktestConfig`**(pydantic v2,`extra="forbid"`)是唯一权威配置模型;YAML 加载 + `DJINN_<SECTION>_<FIELD>` env 覆盖(env > yaml > 默认)。
- **复权**:`forward / backward / none`,默认 `backward`(保证净值连续)。
- **数据缓存**:Parquet(`.cache/djinn/`)+ 内存 LRU,键 = `(provider, symbol, adjust, freq)`。
- **市场推断**:由标的代码或指数(`UNIVERSE_INDEX_MAP`)推断;币种按市场映射(CN→CNY / HK→HKD / US→USD)。

## 数据源与指数

- **A股 / 港股**:AkShare(新浪源,免费免 key);**美股 / 港股**:yfinance;
- **本地 CSV** 作离线 / 测试;Tushare(需 token)作 A股高质量补充。

内置宽基指数:`CSI300` / `CSI500` / `CSI800` / `SSE50` / `STAR50` / `CHINEXT` / `CSI1000` / `HSI` / `HSTECH` / `SP500` / `NASDAQ100` / `DOWJONES`。

## 目录结构

```
src/djinn/
├── data/        数据提供器 / 复权 / 缓存 / 日历 / 基本面 / universe
├── factor/      因子引擎 / 因子库 / 预处理 / 分析(IC·分层·衰减·FMB)
├── screen/      截面选股 / 打分
├── strategy/    策略 ABC / 声明式参数 / 信号适配器 / 内置库 / Pine 转译 / 用户沙箱
├── engine/      事件驱动引擎 / 撮合 / 费用 / 滑点 / 交易约束
├── portfolio/   Decimal 账本 / 持仓 / 分配 / 再平衡 / 风控
├── analytics/   绩效指标 / 基准对比 / 交易统计 / 归因 / 报告
├── api/         FastAPI 后端(路由 / 任务注册 / 报告缓存)
├── viz/         静态图 / 热力图 / HTML 报告
├── io/          CSV / Excel 导出
├── config/      pydantic 配置模型 / YAML 加载 + env 覆盖
├── cli/         typer 入口(run / sweep / data)
└── utils/       日志 / Decimal 辅助 / 日期 / 异常

frontend/        React SPA(Vite + TS + Ant Design + ECharts,13 个页面)
configs/         YAML 示例配置
docs/            计划文档(A~G)
```

## 测试

```bash
# 后端
pytest -m "not network and not slow and not benchmark"   # 离线用例(默认)
pytest -m network                                          # 真实数据拉取(允许 flaky)
mypy --strict src/djinn
ruff check src/djinn tests
black --check src/djinn tests

# 前端
cd frontend && npm run test      # vitest
cd frontend && npm run lint      # eslint
cd frontend && npm run build     # tsc + vite build
```

## License

MIT

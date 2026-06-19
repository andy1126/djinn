# Djinn

**多市场(A股 / 港股 / 美股)量化回测框架**。

事件驱动引擎优先(精确撮合 / 滑点 / 费用),向量化引擎作为性能补充。
YAML 配置 + CLI 优先,Phase 2 并行交付 React SPA + FastAPI Web 前端。

## 特性

- 单标的与自定义组合(多标的)回测
- 统一支持 A 股 / 港股(AkShare / Tushare)与美股(Yahoo Finance)
- 事件驱动引擎:精确撮合、滑点、佣金 / 印花税 / 过户费、涨跌停 / 停牌 / 最小手 / T+1
- 精度策略:指标与收益用 `float64`,现金账本与股数会计用 `Decimal`
- 30+ 绩效指标 + 基准对比
- YAML 配置 + CLI(`djinn run / sweep / data`),可复现、可纳入版本控制

## 安装

```bash
uv pip install -e ".[dev]"            # 内核 + 开发工具
uv pip install -e ".[akshare]"        # 额外:A股/港股 AkShare 数据源
uv pip install -e ".[tushare]"        # 额外:A股 Tushare 数据源(需 token)
pre-commit install
```

## 快速开始

```bash
# 拉取数据(双市场)
djinn data fetch -c configs/backtest.example.yaml

# 跑回测
djinn run -c configs/backtest.example.yaml
```

结果输出到 `results/`:指标、CSV(交易 / 持仓)、HTML 报告。

## 目录结构

```
src/djinn/
├── data/        数据提供器 / 复权 / 缓存 / 日历 / 基准
├── strategy/    策略 ABC / 声明式参数 / 信号适配器 / 内置库
├── engine/      事件驱动引擎 / 撮合 / 费用 / 滑点 / 交易约束
├── portfolio/   Decimal 账本 / 持仓 / 分配 / 再平衡 / 风控
├── analytics/   绩效指标 / 基准对比 / 交易统计 / 报告
├── viz/         净值 / 回撤 / 热力图 / HTML 报告
├── io/          CSV / Excel 导出
├── config/      pydantic 配置模型 / YAML 加载 + env 覆盖
├── cli/         typer 入口(run / sweep / data)
└── utils/       日志 / Decimal 辅助 / 日期 / 异常
```

详见 `CLAUDE.md` 与《实现计划文档》。

## 测试

```bash
pytest -n auto                     # 全部
pytest --benchmark-only            # 性能基准
mypy --strict src/djinn
ruff check src/djinn && black --check src/djinn
```

## License

MIT

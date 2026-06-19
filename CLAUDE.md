# CLAUDE.md — djinn

djinn 是多市场(A股 / 港股 / 美股)量化回测框架。本文件给 Claude Code 提供本子项目的工作约定。

## 架构(分层)

```
Data(数据提供器 + 缓存) → Strategy(策略) → Engine(事件驱动引擎)
  → Portfolio(Decimal 账本 / 再平衡 / 风控) → Analytics(指标) → Viz/IO(可视化 + 导出)
```

- **事件驱动引擎优先**(精确撮合 / 滑点 / 费用),向量化引擎为性能补充(Phase 2)
- **精度策略**:技术指标、收益率序列、组合净值曲线 → `float64`;现金余额、持仓股数、单笔成交金额、手续费 → `Decimal`;每个交易日 mark-to-market 时 Decimal→float 入净值序列
- **防未来函数**:`DataView` 仅暴露 `<= ctx.now` 的数据;信号 `t` 日生成、`t+1` 开盘执行

## 策略 API

两种 API:
- `Strategy` ABC:覆写 `on_bar(ctx)`(复杂 / 组合策略)
- 简单单标的策略可只覆写 `signals(data) -> pd.Series{1,-1,0}`,由 `SignalAdapter` 转成 `on_bar`(~15 行实现一个策略)

声明式参数:`fast = param(10, min=2, max=100, description="快速均线")`,`__init_subclass__` 收集并校验。

## 配置

`BacktestConfig`(pydantic v2)是唯一权威配置模型。CLI 与(Phase 2)FastAPI 都构造同一个 `BacktestConfig` 调用同一内核,结果一致可复现。
YAML 加载 + env 覆盖(env > config.yaml > 默认)。

## 命令

```bash
uv pip install -e ".[dev]"
pytest -n auto
pytest tests/test_xxx.py -v
mypy --strict src/djinn
ruff check src/djinn --fix
black src/djinn
hatch build
```

## 约定

- **Python 3.13+**,black line-length **88**,ruff,mypy **--strict**
- **语言**:UI 文案 / 注释用中文,代码标识符用英文
- **精度**:见上,严禁用 float 记现金 / 股数账本
- **提交**:`type: description`(feat / fix / chore / refactor / test / docs)
- **数据源**:AkShare 免费免 key 作 A/港股默认,Tushare 需 token 作高质量补充,`yfinance` 作美股,本地 CSV 作离线 / 测试
- **缓存**:Parquet(`.cache/djinn/`)+ 内存 LRU,key = `(provider, symbol, adjust, freq)`

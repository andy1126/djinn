# 计划 C:因子与 alpha 研究层

> 目标读者:执行模型。覆盖:因子引擎正确性、估值历史数据源、因子库扩充、统计严谨性、因子合成、策略/沙箱健壮性、Pine 解释器。
> 关键路径:`src/djinn/factor/`、`src/djinn/screen/`、`src/djinn/strategy/`、`src/djinn/indicators/`、`src/djinn/data/providers/`。

## 总览

| # | 改进点 | 类型 | 严重度 | 预估工作量 |
|---|---|---|---|---|
| C1 | net_profit_margin 静默全 NaN 修复 + Factor.required_fields 校验 | bug | P0 | 0.5 天 |
| C2 | A 股估值历史数据源接入(根治 EP/BP/SP/size 前视) | 数据+架构 | P0 | 2~3 天 |
| C3 | 估值快照退化分支 warning + 报告口径标记 | 健壮性 | P1 | 0.5 天 |
| C4 | RSI 全涨窗口返回 100 | bug | P1 | 0.25 天 |
| C5 | neutralize 残差/原值混排修复 + 接入打分流水线 | bug+接线 | P1 | 1 天 |
| C6 | BetaFactor 支持真实基准 | 改进 | P2 | 0.5 天 |
| C7 | 因子库扩充(8 个新因子) | 功能 | P1 | 2~3 天 |
| C8 | IC 统计严谨性(Newey-West t 值 + Fama-MacBeth) | 功能 | P1 | 1.5 天 |
| C9 | 滚动 ICIR 加权合成器(分析→合成闭环) | 功能 | P1 | 2 天 |
| C10 | 因子正交化(Schmidt) | 功能 | P2 | 1 天 |
| C11 | ic_decay 消费:调仓频率推荐 | 功能 | P2 | 0.5 天 |
| C12 | 坏用户指标容错 + 策略继承 MRO + 参数类型校验 | 健壮性 | P1 | 1 天 |
| C13 | Pine:math.* 命名空间 + nz(标量)修复 | bug | P1 | 0.5 天 |
| C14 | score 缺失因子告警(静默重归一化) | 健壮性 | P2 | 0.25 天 |
| C15 | 基本面面板按标的取一次(消除 ×9 重复拉取) | 性能 | P1 | 0.5 天 |

---

## C1. net_profit_margin 静默全 NaN 修复 + Factor.required_fields

### 问题
`factor/library/quality.py:34-45` 的 `NetProfitMarginFactor.compute()` 需要 `COL_NET_PROFIT`/`COL_REVENUE` 基本面面板,但 `factor/engine.py:36-46` 的 `DEFAULT_FUNDAMENTAL_FIELDS` 不含这两个字段 → `fund_panel()`(`library/_util.py:12-14`)返回全 NaN 面板 → 因子已注册进 `FACTOR_REGISTRY`、`/factors` 端点可见,但任何分析结果都是 NaN,**无任何报错**。

### 修改方案
1. **补字段**:`factor/engine.py:36-46` 的 `DEFAULT_FUNDAMENTAL_FIELDS` 末尾加 `"net_profit"`, `"revenue"`(确认 `data/schema.py` 中 COL_NET_PROFIT/COL_REVENUE 常量名与其在 provider normalize 后的列名一致)。
2. **防再犯:声明式字段依赖**。
   - `factor/base.py` 的 `Factor` ABC 增加类属性:`required_fundamentals: tuple[str, ...] = ()`、`required_ohlcv: tuple[str, ...] = ("close",)`。
   - `NetProfitMarginFactor.required_fundamentals = (COL_NET_PROFIT, COL_REVENUE)`;同理给 `turnover`(float_cap)、`ep/bp/sp`、`roe` 等现有因子逐一补上声明(逐个文件核对 compute() 里 `fund_panel(fundamentals, COL_X, ...)` 的调用)。
   - `factor/engine.py` 的 `FactorEngine.compute()`(或 `compute_all`,:97-129)在每个因子计算前校验:
     ```python
     missing = [f for f in factor.required_fundamentals if f not in fundamentals or fundamentals[f].isna().all().all()]
     if missing:
         raise FactorError(f"因子 {factor.name} 所需基本面字段缺失或全空: {missing};"
                           "请检查 DEFAULT_FUNDAMENTAL_FIELDS 或 provider 支持")
     ```
     (启动即 fail-fast,取代静默 NaN。)
3. **provider 字段核查**:`data/providers/akshare.py` 的 `get_fundamentals_history`(`stock_financial_analysis_indicator` 新浪源)是否已 normalize 出 `net_profit`/`revenue` 列;若没有,在该函数的字段映射里补(新浪接口有"主营业务收入"/"净利润"字段);`yahoo.py` 的 `get_fundamentals_history`(:280-310 年度报表)同理(income statement 的 Net Income / Total Revenue)。

### 测试验证
- `test_net_profit_margin_nonzero`:用 `tests/unit/test_api_alpha.py` 的 stub provider 模式合成含 net_profit/revenue 的基本面面板,断言因子值非 NaN 且等于手算比值。
- `test_factor_missing_fields_raises`:人为注册一个声明 `required_fundamentals=("nonexistent",)` 的因子 → `FactorEngine` 抛 FactorError。
- `grep` 全部因子类的 required 声明覆盖率:遍历 `FACTOR_REGISTRY`,凡 compute 内出现 `fund_panel(..., COL_X` 的,X 必须出现在该类 required_fundamentals(写成元测试)。

---

## C2. A 股估值历史数据源接入(根治前视)

### 问题(平台级可信度)
`factor/engine.py:208-216`:`_asof_field_panel` 的退化分支把 `when=end` 的**当日快照常数**填充到全部历史日期。akshare 的 pe/pb/ps/market_cap/float_cap 无历史时序(`providers/akshare.py:446-458` 明确"实时快照")→ **EP/BP/SP/size/turnover 五个因子在 A 股回测中用今天的估值给三年前打分**,IC 与选股回测结果系统性失真(价值因子回看时"已知"哪些股票后来变便宜)。这是选股平台核心卖点的可信度问题。

### 数据来源(三选其二,实现可插拔)

| 源 | 接口 | 字段 | 成本 | 说明 |
|---|---|---|---|---|
| **akshare `stock_a_indicator_lg`**(乐咕) | `ak.stock_a_indicator_lg(symbol="600519", start_date=..., end_date=...)` | 日频 pe/pb/ps(无市值) | 免费免 key | 推荐首选;逐标的拉取,与现有限速/缓存体系兼容 |
| **tushare `daily_basic`** | `pro.daily_basic(ts_code=..., start_date=..., end_date=...)` | 日频 pe/pb/ps/total_mv/circ_mv/turnover_rate | 需 token(已有 extras) | 字段最全(含市值、换手率);有积分门槛 |
| yfinance | `Ticker.info` 历史不可用;`quarterly balance sheet` 的 shares × 价格可近似 market_cap | 季频近似 | 免费 | 仅作美股补充 |

市值推导:若只有 pe/pb/ps 无市值,`market_cap ≈ close × 总股本`;总股本可用 akshare `stock_individual_info_em` 或乐咕接口附带的 total_share。**简化方案:C2 第一阶段只接 pe/pb/ps 日频历史(乐咕 + tushare daily_basic),market_cap/float_cap 的历史用 `close × 总股本(最新)` 近似并在报告口径中注明"股本未做 PIT"**——估值前视(主要矛盾)消除,股本漂移(次要,股本变动不频繁)接受。

### 修改方案

**1. 抽象层(`data/fundamentals.py`)**

`FundamentalsSource` 已有 `get_history(symbol, start, end)` 返回含 `announce_date`/`report_date` 的财报时序。估值是**日频行情衍生序列**,语义不同于财报 —— 增加第三个入口:

```python
def get_daily_valuation(
    self, symbol: str, start: date, end: date
) -> pd.DataFrame:
    """日频估值时序:index=交易日,columns 含 pe/pb/ps(+可选 market_cap/float_cap)。
    基类默认返回空 DataFrame(不支持)。"""
```

**2. akshare provider(`data/providers/akshare.py`)**

- 新增 `get_daily_valuation`:调 `ak.stock_a_indicator_lg(symbol=六位代码, start_date="YYYYMMDD", end_date="YYYYMMDD")`,normalize 列(`pe`→COL_PE 等,沿用 `_normalize` 的命名映射),索引转 DatetimeIndex。
- 走与行情相同的缓存:cache key dtype 新增 `"valuation"`(`data/cache.py` 的键模板 `{provider}::{dtype}::{symbol}::{adjust}`,adjust 对估值固定 "none");复用 `covers()` 语义(注意 D 计划的 covers 修复)。
- 限速复用 `_throttle`;失败抛 ProviderError 由 FactorEngine 决定退化。

**3. tushare provider(如代码库已有 `providers/tushare.py` 骨架则扩展;没有则在 akshare 之后第二阶段做)**

`pro.daily_basic`,字段映射 pe_ttm→pe、pb→pb、ps_ttm→ps、total_mv→market_cap(单位万元 → ×1e4)、circ_mv→float_cap。

**4. FactorEngine 接线(`factor/engine.py:164-237`)**

`_fundamental_panels` 对估值类字段(pe/pb/ps/market_cap/float_cap)优先走日频估值序列:

```python
VALUATION_FIELDS = (COL_PE, COL_PB, COL_PS)
# 逐标的:try provider.get_daily_valuation(sym, start, end)
#   非空 → 直接 reindex 到交易日面板(日频,无需 merge_asof;ffill 缺日)
#   空/异常 → 回退现有 _asof_field_panel 退化分支(快照常数 + warning,见 C3)
```

市值字段:若 provider 无历史市值,用 `close_panel × 总股本` 合成(总股本经 `get_snapshot` 的 market_cap/close 反推最新值),并在引擎返回的 `FactorPanel.meta` 记录 `market_cap_approx=True` 供报告层标注。

**5. 路由层优先级**:`data/providers/router.py`(或 registry 选择逻辑)对 `get_daily_valuation` 按优先级遍历 provider(tushare 有 token 优先,akshare 兜底)。

### 测试验证
- 单测:mock akshare 返回值(构造 DataFrame),断言 normalize 后列名/索引/缓存键;`pytest -m "not network"` 可跑。
- 网络测试(标 `@pytest.mark.network`):`stock_a_indicator_lg` 拉 600519 近 30 天,断言 pe 列非空、日期连续。
- **PIT 验证(关键)**:构造合成估值序列(已知 t0 时刻 pe=10、t1 变为 20),因子引擎在 t0 日期截面只可见 10;EP 因子在 t0 == 0.1。
- 回归:无网络环境下 akshare 不可达时,因子分析任务走退化分支并报 warning(C3),结果仍产出(口径标记)。

---

## C3. 估值快照退化分支 warning + 报告口径标记

### 问题
即使 C2 落地,退化分支(yahoo 美股、akshare 失败)仍存在。当前静默退化,用户不知道因子口径已被污染。

### 修改方案
1. `factor/engine.py:208-216` 退化分支:首次退化时 `_log.warning("字段 %s 无历史时序,使用 %s 快照常数填充全历史(非 PIT)", field, when)`(按字段去重,每字段只警一次:引擎实例上加 `_warned_fields: set`)。
2. `FactorPanel`/`FactorReport`(`factor/analysis/report.py`)增加 `data_caveats: list[str]` 字段,引擎把退化字段名单传入;`/factor-analysis/{id}/report` 响应透出;前端 FactorAnalysisPage 报告区顶部显示 Alert("以下字段为快照口径:pe、pb(非 point-in-time,IC 可能高估)")。
3. 选股策略路径(`FactorPortfolioStrategy` 经 runner 拉基本面时)同样将 caveats 写入回测 `Report`(report.py 的 meta dict),报告页展示。

### 测试验证
- caplog 断言 warning 恰好一次/字段;report dict 含 caveats 且 API 响应可见( stub provider 模式,test_api_alpha.py 加断言)。

---

## C4. RSI 全涨窗口返回 100

### 问题
`indicators/__init__.py:112-114`:

```python
rs = avg_gain / avg_loss.replace(0, np.nan)
out = 100 - 100 / (1 + rs)
return out.fillna(50.0)
```

连续上涨窗口 avg_loss==0 → rs=NaN → RSI=NaN → fillna(50)。连续上涨应 RSI=100;且 fillna(50) 把"数据缺失"与"无下跌"混为一谈。

### 修改方案

```python
rs = avg_gain / avg_loss.replace(0, np.nan)
out = 100 - 100 / (1 + rs)
# 全涨(avg_loss==0 且 avg_gain>0)→ 100;平盘(双 0)→ 50;仅数据缺失段保持 50
both_zero = (avg_loss == 0) & (avg_gain > 0)
out = out.mask(both_zero, 100.0)
flat = (avg_loss == 0) & (avg_gain == 0)
out = out.mask(flat, 50.0)
return out.fillna(50.0)
```

### 测试验证
文件:`tests/unit/test_indicators.py`。
- `test_rsi_all_up`:单调上涨 30 日 close → RSI 末值 == 100。
- `test_rsi_all_down` → 0;`test_rsi_flat`(常数序列)→ 50。
- 对照 TA-Lib 语义(若有):连续 14 涨后 RSI=100。

---

## C5. neutralize 残差/原值混排修复 + 接入打分流水线

### 问题
1. `factor/preprocess.py:113-115`:被 mask 剔除的标的(当日缺市值等)保留**原始因子值**,其余写回**残差** —— 同一行输出混合两种量纲,截面不可比。
2. neutralize 实现存在但**生产路径零调用**:`screen/scoring.py:52-64` 的 `score_cross_section` 只做 winsorize+standardize;`FactorPortfolioStrategy(preprocess=True)` 同样不含 neutralize。

### 修改方案
1. `preprocess.py` neutralize(:100-117):被 mask 标的的输出置 NaN:

```python
resid = pd.Series(np.nan, index=row.index)  # 原为 row.copy()
resid.loc[valid] = y - X @ beta  # 仅有效标的写残差
```

2. 接入打分流水线:`screen/scoring.py` 的 `score_cross_section(factor_values, scores, *, preprocess=True, neutralize: bool = False, industry_map=None, log_mktcap=None)` 增加可选参数;当 `neutralize=True` 且两个面板都给定时,在 winsorize 后、standardize 前调 `preprocess_neutralize`。缺参数时 warning 并跳过(不静默)。
3. `FactorPortfolioStrategy`(`strategy/library/factor_portfolio.py`)增加 `param`:`neutralize = param(False, description="行业/市值中性化")`;调仓打分时若开启,从引擎注入的上下文拿 industry_map 与市值面板 —— 策略当前自行计算 scores(:77-84),所需 `industry_map` 可经 `cli/runner.py` 的 `_industry_map_safe`(归因同款)注入策略构造参数;市值面板由因子引擎的 fundamentals 面板传 `COL_MARKET_CAP` 取 log。接线点:`runner.py` 的 `build_strategy()`。
4. API:`FactorMatrixRequest`/回测请求 schema 暂不动(矩阵诊断页只做诊断);选股端点 `ScreenRequest` 不加(截面选股打分保持轻量)。仅策略与 `score_cross_section` 显式开启。

### 测试验证
- `test_neutralize_masked_nan`:构造含缺市值标的的面板 → 输出中该标的全 NaN,其余标的行业均值为 0(每行业残差均值 ≈0,断言 <1e-10)。
- `test_score_pipeline_neutralize`:`score_cross_section(..., neutralize=True, ...)` 输出经行业分组均值 ≈0。
- 策略级:FactorPortfolioStrategy(neutralize=True) 跑合成数据,断言持仓权重与未开启时不同且行业偏离度下降。

---

## C6. BetaFactor 支持真实基准

### 问题
`factor/library/volatility.py:24-39`:`beta` 的市场代理是截面等权收益(`ret.mean(axis=1)`),池子时变时口径漂移,且非真实基准。

### 修改方案
1. `BetaFactor` 增加 `param`:`benchmark: str | None = param(None, description="基准代码(如 000300.SH/^GSPC);None 时沿用截面等权代理")`。
2. 因子引擎(`factor/engine.py`)在 `compute()` 调用前检测:若因子有 `benchmark` 属性非 None,经 ProviderRegistry 拉取该基准的 close 序列,以 `ohlcv` dict 的特殊键(如 `"__benchmark__"`)传入;`BetaFactor.compute()` 内:

```python
bench = ohlcv.get("__benchmark__")  # Series(日收益)
if bench is not None:
    mkt = bench.reindex(ret.index).ffill()
else:
    mkt = ret.mean(axis=1)  # 现状退化
```

3. `cli/runner.py` 的因子分析/回测路径把 `cfg.universe.benchmark` 透传因子参数(若用户未显式指定)。

### 测试验证
- 合成基准序列与某标的完全相关 → beta==1;反向 → -1;基准缺失时退化为现状(回归现有测试)。

---

## C7. 因子库扩充(8 个新因子)

> 全部遵循现有模式:`factor/library/<category>.py` 加类 + `library/__init__.py` 注册 + `param()` 声明 + 单测。注册后 `/factors` 端点与前端表单自动出现(CLAUDE.md 约定)。

| 因子 | name/category | 公式(向量化) | 数据需求 | 文件 |
|---|---|---|---|---|
| 52 周高点距离 | `high_52w` / momentum | `close / close.rolling(252).max() - 1`(负值,越接近 0 越强) | 仅 close | momentum.py |
| MAX 彩票 | `max_lottery` / volatility | 日收益 `rolling(21).max().rolling(21).mean()` 或月内最大 5 日收益均值 | 仅 close | volatility.py |
| 特质波动率 | `idio_vol` / volatility | 日收益对市场代理(截面等权或 benchmark,C6)滚动 60 日 OLS 残差 std;实现:滚动窗口内 `np.polyfit` 或预计算滚动 beta 后 `ret - beta*mkt` 再 rolling.std | close(+benchmark) | volatility.py |
| 换手率变化率 | `turnover_chg` / liquidity | `turnover(20) / turnover(120) - 1`(复用 turnover 因子逻辑取 amount/float_cap) | amount + float_cap | liquidity.py |
| 应计因子 | `accruals` / quality | `(net_profit - ocf) / total_assets` 的同比变化率(低应计=高质量,因子取负向) | net_profit + ocf + total_assets(基本面历史) | quality.py |
| 资产增长率 | `asset_growth` / quality | `total_assets / total_assets.shift(4期) - 1`(财报期频;强负向异象,权重示例给负) | total_assets | quality.py |
| 现金市值比 | `cfp` / value | `ocf / market_cap`(经营现金流/市值) | ocf + market_cap | value.py |
| 股息率 | `div_yield` / value | `dividend_ttm / market_cap`(schema 已有 COL_DIVIDEND;TTM 用财报时序滚动 4 期求和) | dividend + market_cap | value.py |

**数据需求配套**:
- `ocf`(经营现金流)、`total_assets`:akshare `stock_financial_analysis_indicator`(新浪)含"经营现金流"/"总资产"字段,在 `providers/akshare.py` 的 `get_fundamentals_history` normalize 映射里补列;yahoo `Ticker.cashflow` 的 Operating Cash Flow / balance sheet Total Assets,在 `yahoo.py:280-310` 补。schema.py 加 `COL_OCF`/`COL_TOTAL_ASSETS` 常量。
- `DEFAULT_FUNDAMENTAL_FIELDS`(engine.py:36-46)同步补 `ocf`/`total_assets`(C1 的 required_fundamentals 机制保证缺字段 fail-fast)。
- dividend 字段若 schema 已有 COL_DIVIDEND 且 provider 已填充,直接用;否则参照 ocf 同样补。

**每个因子的测试**(tests/unit/test_factor.py 追加):
- 合成 2 标的 × 300 日面板,手算期望值断言(如 high_52w:构造 close 在末日为 252 日新高 → 因子值 == 0)。
- 边界:窗口不足 → NaN;全 NaN 输入 → 全 NaN 不抛错。
- 注册表元测试:`FACTOR_REGISTRY[name]` 可 `make_factor(name)` 且 `param_schema` 可 JSON 序列化。

---

## C8. IC 统计严谨性(Newey-West t 值 + Fama-MacBeth)

### 修改方案
1. **Newey-West t 值**:`factor/analysis/ic.py` 的 `ic_summary()` 增加输出:

```python
def _newey_west_t(ic: pd.Series, lags: int | None = None) -> float:
    """HAC 标准误下的 t(IC 均值);lags 默认 floor(4*(T/100)^(2/9))。"""
    # e = ic - mean;S = Σ_{j=-L}^{L} w_j * γ_j,w_j = 1-|j|/(L+1)(Bartlett)
    # se = sqrt(S/T);t = mean/se
```

`ic_summary` 返回 dict 增加 `ic_t`、`ic_pvalue`(p 值用 `scipy.stats.t.sf`(已有 scipy 依赖)或正态近似;为避免新依赖用 `math.erfc` 正态近似)。`FactorReport` 与 `/factor-analysis/{id}/report` 透出;前端 IC 汇总卡显示 "t=2.31 (p=0.02)"。

2. **Fama-MacBeth 回归**:新文件 `factor/analysis/fmb.py`:

```python
def fama_macbeth(
    factors: dict[str, pd.DataFrame],   # {name: date×symbol 因子值}
    fwd_returns: pd.DataFrame,          # 前向收益
    *,
    standardize: bool = True,           # 每日截面 zscore 化
) -> FMBResult:
    """逐日截面回归 r_{t+1} = a + Σ λ_k f_k + ε → 时序 {λ_k(t)} 的均值/NW t 值/显著性。

    FMBResult: {factor: {"lambda_mean", "lambda_t", "lambda_pvalue", "pos_ratio"}}, n_days。
    实现:逐日 np.linalg.lstsq(复用 preprocess.neutralize 的模式),λ 时序复用 _newey_west_t。"""
```

3. 接入点:`analyze_factor_matrix`(`matrix.py`)的 `FactorMatrixReport` 增加可选 `fmb` 字段;`/factor-matrix` job(jobs.py 的 run_factor_matrix_job)在因子数 ≥2 时附带计算;前端 FactorMatrixPage 加"Fama-MacBeth 因子收益"表(λ 均值 + t 值)。

### 测试验证
- `test_newey_west_white_noise`:随机 IC 序列(|t| 应 <2 大概率);人为均值 0.05 无自相关序列,t ≈ 经典 t 值(对比 `scipy.stats.ttest_1samp`,容差 10%)。
- `test_fmb_recovers_lambda`:合成 r = 2*f1 + 0*f2 + 噪声 → λ̂₁≈2、λ̂₂≈0,t 值显著性符合;与逐日 statsmodels 对照(若 dev 环境有 statsmodels,仅在测试 dev 依赖加,不进主依赖)。

---

## C9. 滚动 ICIR 加权合成器

### 问题与目标
`ic_summary` 产出了 ICIR 但无下游消费;`FactorScore.weight` 全靠手填,sweep 盲扫。做一个**滚动 ICIR 加权**合成器:每日用过去 N 期各因子 ICIR 归一化作为当日权重,合成打分。这是把"诊断"(factor-analysis)与"选股"(FactorPortfolioStrategy)缝起来的关键一环。

### 修改方案
1. **新文件:`factor/composite.py`**

```python
def rolling_ic_weights(
    factors: dict[str, pd.DataFrame],   # {name: date×symbol}
    fwd_returns: pd.DataFrame,          # 单期前向收益(与调仓频率对齐)
    *,
    window: int = 60,                   # IC 滚动窗口(交易日)
    min_periods: int = 20,
    decay_halflife: float | None = None,  # 可选半衰期衰减权重
) -> pd.DataFrame:
    """返回 date×factor 的权重面板:w_k(t) = ICIR_k(滚动window) / Σ|ICIR|,符号保留。
    ICIR = mean(IC)/std(IC);std=0 或窗口不足 → 权重 0(并记录 coverage)。"""
```

实现:复用 `analysis/ic.py` 的 `compute_ic` 得逐因子 IC 序列 → 每因子 rolling mean/std → 每日归一。向量化:IC 面板 date×factor,一次 rolling 即可,无逐日循环。

2. **合成打分**:同文件 `composite_score(factors, weights_panel) -> pd.DataFrame`:每日截面 `Σ_k w_k(t) * zscore(f_k(t))`(zscore 复用 preprocess)。输出 date×symbol 得分面板。
3. **策略接入**:`FactorPortfolioStrategy` 增加 `weighting = param("static", choices=["static","icir"])` 与 `icir_window = param(60, ...)`;调仓时若 weighting=="icir":用 ≤now 的历史因子值与 1 期前向收益**注意:前向收益在 now 时点只能用到 now−holding_period 的数据**(t 日可用的最近完整前向收益是 t−p 日到 t 日),实现时 `fwd_returns` 面板截断到 `<= now` 即天然 PIT(rolling IC 在 t 日只用 IC(≤t),而 IC(t) 依赖 fwd(t) = ret(t→t+p) —— **这是未来函数!** 必须把 IC 序列右移 p 日:`ic_effective(t) = ic(t−p)`)。在 rolling_ic_weights 内加 `shift_periods` 参数显式处理,docstring 显著标注。
4. **API/CLI**:sweep 的 `strategy.factor_weights` 轴保持现状;新增 sweep 轴 `strategy.weighting`(白名单 `ALLOWED_SWEEP_AXES` 前后端同步加)。

### 测试验证
- `test_icir_weights_pit`(关键):构造因子 A 在 t>t0 后 IC 恒为 +1(完美预测),断言 t0+p 之前 A 的权重 == 0(看不到未来),t0+p 之后权重 →1。
- `test_weights_normalized`:每日 |w| 和 == 1(有有效因子时)。
- 端到端:synthetic 因子(A 有预测力、B 纯噪声)跑 FactorPortfolioStrategy(weighting="icir") vs static 等权 → 前者夏普显著更高。

---

## C10. 因子正交化(Schmidt)

### 修改方案
`factor/preprocess.py` 增加:

```python
def orthogonalize(
    factors: dict[str, pd.DataFrame],
    order: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """逐日截面对因子面板做 Schmidt 正交化:按 order(默认 dict 顺序)依次对
    前序因子回归取残差。返回同构 dict。每日截面实现同 neutralize 的 lstsq 模式,
    自变量为前序因子值 + 常数项。被 mask 标的置 NaN(与 C5 一致)。"""
```

接入:`analyze_factor_matrix` 的 FactorMatrixReport 增加可选 `orthogonalized=True` 分支——诊断页可切换查看正交化后的相关矩阵;`score_cross_section` 增加 `orthogonalize: bool = False`(在 standardize 前)。前端矩阵页加 Switch。

### 测试验证
- 两个完全相关因子(相关系数 1)正交化后:后者全 NaN 或 0 向量(lstsq 残差 ≈0),前者不变;部分相关(0.8)→ 后者与前者截面相关 ≈0(<1e-10);顺序敏感性文档化。

---

## C11. ic_decay 消费:调仓频率推荐

### 修改方案
`factor/analysis/report.py` 的 `FactorReport` 增加 `recommended_rebalance: str | None`(枚举 daily/weekly/monthly/quarterly):规则 = IC 半衰期(ic_decay 输出 {1,5,10,21} 期 IC,找 IC 衰减到峰值 50% 的最短周期)映射到最近调仓档。纯规则函数 `_recommend_freq(decay: dict[int, float]) -> str`。`/factor-analysis/{id}/report` 透出;前端 FactorAnalysisPage 在 IC 衰减图旁显示"建议调仓频率:月度"及 Tooltip 解释。

### 测试验证
- 构造 decay={1:0.08, 5:0.07, 10:0.03, 21:0.01} → 半衰期在 5~10 之间 → 推荐 weekly/monthly(按规则表断言)。

---

## C12. 坏用户指标容错 + 策略继承 MRO + 参数类型校验

### 修改方案
1. **坏指标容错**(`indicators/user.py:59-64`):`get_user_indicator_functions()` 遇编译失败的记录 → `_log.warning("用户指标 %s 编译失败,已跳过: %s", name, e)` 并 continue(不抛 StrategyError 砖掉全部用户策略)。
2. **策略继承 MRO**(`strategy/base.py:235-243`):`__init_subclass__` 的检查改为沿 MRO 查找覆写:

```python
def _overrides(cls, name):
    for klass in cls.__mro__[1:]:  # 跳过自身
        if name in klass.__dict__ and klass.__dict__[name] is not getattr(Strategy, name):
            return True
    return False
# has_on_bar = "on_bar" 在子类自身或中间父类被覆写即可
```

(`class MyMAC(MACrossover): fast = param(5, ...)` 应合法。)

3. **参数类型校验**(`strategy/parameter.py:33-40`):`Parameter.validate` 开头加类型检查:

```python
if self.default is not None and value is not None and not isinstance(value, type(self.default)):
    # bool 是 int 子类,特判;int 参数接受 float 整值自动转 int
    if isinstance(self.default, int) and not isinstance(self.default, bool) and isinstance(value, float) and value.is_integer():
        value = int(value)
    else:
        raise ParameterError(f"参数 {self.name} 类型不符: 期望 {type(self.default).__name__},实际 {type(value).__name__}")
```

min/max 比较前先做类型检查(消除裸 TypeError)。default=None 的参数(如 PairsSpread.symbol_a)在 `__init__` 阶段跳过校验但在 param_schema 输出 `required: true`(`strategy/parameter.py` 的 schema 生成处加),前端表单据此标必填。

### 测试验证
- `test_bad_indicator_skipped`:store 注入一个语法错误指标 → 其他正常用户策略仍可编译,caplog 有 warning。
- `test_strategy_inheritance`:`class Sub(MACrossover): fast = param(5)` 可实例化且 fast==5。
- `test_param_type_check`:`MACrossover(fast="abc")` 抛 ParameterError(消息含参数名);`fast=10.0` 自动转 10;bool/int 边界。

---

## C13. Pine:math.* 命名空间 + nz(标量)修复

### 问题
1. `strategy/pine/__init__.py:97-107` `_MATH_MAP` 把 `math.sqrt/log/exp/sign` 映射到裸名 `sqrt/log/exp/sign`,但执行命名空间(`strategy/user.py:28-38` = SAFE_BUILTINS + pd/np + indicators)没有这四个函数 → 编译通过、运行 NameError。
2. 同文件 :356-359:`nz(标量)` 生成 `0.fillna(1)` 语法错误。

### 修改方案
1. `strategy/user.py` 的 `_build_namespace()`(:28-38)注入:

```python
ns.update({
    "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
    "sign": np.sign, "pow": np.power, "abs_": np.abs,  # abs 用内建
})
```

(与 `_MATH_MAP` 的目标名一一对应;"abs"/"min"/"max"/"round" 走 SAFE_BUILTINS 内建,无需注入 —— 逐个核对 _MATH_MAP 九个映射目标在命名空间的可达性,写成元测试。)

2. `pine/__init__.py` 的 `nz()` 生成逻辑(:356-359 附近):判断参数 AST 节点类型——若为序列表达式(变量/调用/下标)生成 `<expr>.fillna(1)`;若为字面量/标量,直接生成 `<expr>`(nz 对标量无操作)。补单测:`nz(0)`、`nz(close)`、`nz(close - open)` 三种输入的转译输出可编译可运行。

### 测试验证
- `test_pine_math_funcs`:含 `math.sqrt(close)` 的 Pine 脚本端到端(pine_to_python → compile_user_strategy → 跑 signals)无 NameError。
- 元测试:遍历 `_MATH_MAP.values()`,断言每个名字在 `_build_namespace()` 结果中可达。

---

## C14. score 缺失因子告警

### 问题
`screen/scoring.py:56-57`:`score_cross_section` 对 scores 里缺失的因子静默 continue;`score_universe`(:93-96)对某日期缺失的因子整列跳过 → 权重静默重归一化,因子面板起始不齐时早期得分口径漂移。

### 修改方案
两处静默点改 `_log.warning` + 首次告警(实例级 seen set 去重);`score_cross_section` 返回值附带 `meta: dict`(或模块级 `LAST_SCORE_META`)记录实际参与因子名单,供调用方/测试断言。

### 测试验证
- caplog 断言 warning 一次;打分结果中缺失因子的权重确实按剩余因子归一(手算断言)。

---

## C15. 基本面面板按标的取一次(消除 ×9 重复拉取)

### 问题
`factor/engine.py:174-179`:`_fundamental_panels` 按**字段**循环,每字段对全部标的调 `get_history` → 同一标的财报时序被拉 9 遍(provider 有 `finhist_` 落盘缓存兜底网络层,但反序列化+normalize 开销 ×9)。

### 修改方案
重构为按**标的**循环:

```python
def _fundamental_panels(self, symbols, start, end, source, fields):
    histories: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        histories[sym] = source.get_history(sym, start - timedelta(days=_HISTORY_LOOKBACK_DAYS), end, ...)
    for field in fields:
        panels[field] = _asof_field_panel_from_histories(histories, field, ...)
```

(`_asof_field_panel` 拆成"取历史"与"asof 对齐"两步,后者接收已取的 histories dict。)

### 测试验证
- mock source 记录 `get_history` 调用次数:9 字段 × 10 标的 → 断言调用 == 10(原 90);输出面板与重构前逐值相等(快照对比)。

---

## 验收清单

1. 静态检查 + 全量 pytest 绿(新增 ~40 用例);网络测试单独跑 `pytest -m network` 确认乐咕/tushare 接口连通(允许 flaky,CLAUDE.md 约定)。
2. C2 上线前后对比:同一 EP 因子在 CSI300 的 IC/ICIR 应**显著下降**(前视消除);在报告中注明口径变化(CHANGELOG 或 docs)。
3. 前端:`/factors` 自动出现 8 个新因子;FactorMatrixPage 的 FMB 表与 FactorAnalysisPage 的 t 值/建议频率展示正常;`tsc -b --noEmit` 过。

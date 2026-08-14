# 计划 B:指标、交易统计与归因口径修正

> 目标读者:执行模型。本文档自包含。所有改动位于 `src/djinn/analytics/`、`src/djinn/portfolio/`、`src/djinn/cli/sweep.py`。
> 口径总原则:指标数值必须与主流量化平台(backtrader/qs/聚宽)可对比;口径非主流时必须文档化。

## 总览

| # | 改进点 | 严重度 | 预估工作量 |
|---|---|---|---|
| B1 | round-trip 交易统计(胜率/盈亏比/avg_holding_days + 佣金摊入成本) | P0 | 2 天 |
| B2 | metrics.n_trades 口径修复 + sweep 引用修正 | P0 | 0.25 天 |
| B3 | 索提诺标准口径(MAR + 下行偏差) | P2 | 0.25 天 |
| B4 | Calmar 在 mdd=0 时返回 NaN + sweep 排序 NaN 处理 | P1 | 0.25 天 |
| B5 | monthly_returns 补首月收益 | P2 | 0.25 天 |
| B6 | alpha → Jensen alpha + 新增 upside_capture | P1 | 0.5 天 |
| B7 | 因子归因暴露滞后一日(前视修复) | P1 | 0.5 天 |
| B8 | VaR 口径与注释一致化 + 换手率口径文档化 | P2 | 0.25 天 |
| B9 | 两套 Fill 命名消除(可读性) | P2 | 0.5 天 |

---

## B1. round-trip 交易统计

### 问题(三个互相叠加)
1. `analytics/trades.py:60-65`:胜率/盈亏比的数据源是 `realized_pnls: dict[str, Decimal]`(各标的**累计**已实现盈亏),同一标的多次买卖被压成一笔 → "胜率"实为"盈利标的占比"。
2. `portfolio/account.py:160-161`:`realized = q_money((price - pos.avg_cost) * qty)` —— **不含佣金摊销**(注释自承"简化"),高换手策略盈利虚高。
3. `trades.py:75`:`avg_holding_days` 硬编码 0.0,未实现。

### 修改方案(新建 round-trip 配对器,替换数据源)

**1. 新文件:`src/djinn/analytics/roundtrip.py`**

按 FIFO 开平仓配对,从 fills 序列重建交易回合:

```python
"""Round-trip(回合)交易配对:FIFO 开仓→平仓,含双边佣金摊销。"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

@dataclass
class RoundTrip:
    symbol: str
    open_date: date
    close_date: date
    qty: float                 # 本回合股数
    open_price: float          # 加权平均开仓价
    close_price: float         # 加权平均平仓价
    pnl: float                 # 净盈亏(含双边佣金)
    holding_days: int          # (close_date - open_date).days

def pair_round_trips(fills: list[Any]) -> list[RoundTrip]:
    """按 FIFO 把 fills 配对为回合。

    规则:
    - 每个标的维护一个 FIFO 开仓队列 [(date, qty, price, commission), ...];
    - 买单:入队;
    - 卖单:从队首依次冲销;每次冲销生成一个 RoundTrip
      (被部分冲销的开仓批拆成两段);
    - pnl = (close_price - open_price) * qty - 开仓佣金摊派 - 平仓佣金摊派;
      佣金按股数比例摊派到被冲销的股数上;
    - 回测结束仍未平仓的开仓批不生成回合(浮盈不进胜率)。
    """
```

实现要点:
- 输入是 `engine.events.Fill`(float 口径,字段 `timestamp/symbol/side/qty/price/commission`);统计层用 float 即可(与 metrics float64 口径一致,不违反账本 Decimal 不变量)。
- 佣金摊派:开仓批 `(date, qty0, price, comm0)`,被冲销 `q` 股时摊 `comm0 * q/qty0`;平仓侧同理。
- 部分冲销拆段:开仓批剩 `qty0 - q` 继续排队。
- 卖空不存在(平台仅做多),side 只有 buy/sell,无需处理反向开仓。

**2. 改 `trades.py:compute_trade_stats`**

签名改为 `compute_trade_stats(fills, positions=None, *, realized_pnls=None)` 保留旧参数兼容,但内部改为:

```python
rts = pair_round_trips(fills)
pnls = [rt.pnl for rt in rts]
avg_hold = float(np.mean([rt.holding_days for rt in rts])) if rts else 0.0
# wins/losses/avg_win/avg_loss/win_rate/pl_ratio 逻辑不变,数据源换成 pnls
```

`TradeStats` 增加字段 `n_round_trips: int = 0`,`to_dict()` 同步输出;`per_trade_pnl` 语义改为"每回合 pnl"(docstring 更新)。

**3. `account.py` 的佣金摊销(可选增强,非必须)**

若 B1-2 落地,`Position.realized_pnl` 仅用于展示时可保持现状;若希望账本层的 realized_pnl 也含费用,在 `account.py:160` 改为 `realized = q_money((price - pos.avg_cost) * qty - commission - 卖出股数摊派的买入佣金)` —— 需要 Position 记录累计买入佣金(`pos._cum_buy_cost` 已含,可摊派:`pos._cum_buy_cost * qty / pos.qty - qty*avg_cost 部分`)。**建议:本项只做展示层(B1-2),账本层 realized_pnl 口径在 docstring 注明"不含费用",避免双重维护。**

**4. 调用点更新**
- `analytics/report.py:85` 附近:`compute_trade_stats(result.trades, realized_pnls=...)` 调用处改为只传 fills(round-trip 自给自足),`realized_pnls` 参数标记 deprecated。
- 前端 `types/index.ts` 的 TradeStats 镜像类型加 `n_round_trips`(CLAUDE.md 约定:后端 schema 改动同步该文件)。

### 测试验证
文件:`tests/unit/test_metrics.py`(已存在)追加;新建 `tests/unit/test_roundtrip.py`。

- `test_fifo_pairing_basic`:买 100@10(佣金 5)→ 卖 100@12(佣金 5)→ 1 个回合,pnl = 200 − 10 = 190(双边佣金摊销),holding_days 正确。
- `test_fifo_partial_close`:买 100@10 → 买 100@11 → 卖 150@12 → 2 个回合(100 股 @10→12、50 股 @11→12),剩余 50 股不生成回合。
- `test_win_rate_multi_trades_same_symbol`:同一标的 3 次完整买卖(2 盈 1 亏)→ win_rate == 2/3(改动前会聚合成 1 笔)。
- `test_avg_holding_days`:两回合 holding_days 3 与 7 → avg == 5.0。
- `test_open_position_excluded`:末段有未平仓 → 不计回合。
- 回归:`report.build_report` 集成用例(test_attribution.py / test_api.py 的报告断言)数值更新为合理新值。

---

## B2. metrics.n_trades 口径修复 + sweep 引用修正

### 问题
`analytics/metrics.py:175`:`n_trades=len(pnls)`,而 `pnls` 来自 `compute_metrics(..., trades=...)` 的 `_pnls_from_trades`(trades.py:186-193 当前返回空列表,实际 pnls 由 report.py:85 传入 trade_stats.per_trade_pnl)→ `metrics.n_trades` 实为"有盈亏的标的数"(B1 落地后变为"回合数",仍非成交笔数)。`cli/sweep.py:150` 直接用 `m.n_trades` 填结果表 trades 列 → sweep 表交易数错误。

### 修改方案
1. `metrics.py`:`n_trades=len(list(trades))`(trades 参数即 fills 列表)。注意 `compute_metrics` 的 trades 参数是 Iterable,可能已被消费 → 在函数开头 `trades = list(trades)` 物化。
2. `cli/sweep.py:139-151`:`_run_one` 里改用 `report.trade_stats.n_trades`(=len(fills))或保留 `m.n_trades`(修复后语义已正确);顺手修 :202 的 `r.get(target, 0.0) or 0.0` → `r.get(target) if r.get(target) is not None else 0.0`(避免合法 0.0 与缺失混同);删除 :139-151 中 `{target: ...}` 与 `"sharpe": m.sharpe` 的同键重复(当 target="sharpe" 时),统一为固定键集 + target 值。

### 测试验证
- `test_n_trades_counts_fills`:3 买 2 卖 → metrics.n_trades == 5(不是回合数也不是标的数)。
- `test_sweep_trades_column`:sweep 结果 dict 的 trades 字段 == 实际 fills 数。
- 排序回归:sweep --target calmar 与 B4 联测。

---

## B3. 索提诺标准口径

### 问题
`metrics.py:140-141`:下行阈值用 0 而非 MAR(最低可接受收益,通常取 rf 日化),半方差只取负样本的 `std()` 而非全样本下行偏差 → 与夏普的 excess 口径不自洽,数值不可跨平台对比。

### 修改方案
`metrics.py` sortino 计算改为:

```python
mar = rf / af  # 日化最低可接受收益(与 sharpe 的 rf 口径一致)
excess = rets - mar
downside = np.minimum(excess, 0.0)
downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(af))
sortino = float(excess.mean() * af / downside_dev) if downside_dev > 0 else 0.0
```

(年化:`mean*af` 为年化超额,`downside_dev` 年化;与 sharpe 同结构。)

### 测试验证
- `test_sortino_standard`:手工构造 5 日收益序列(含正负),用 numpy 按上式独立计算期望值断言;全正收益序列 → downside_dev==0 → sortino==0(文档注明:无下行风险时定义为 0 而非 inf,避免 sweep 排序爆炸)。
- 与 sharpe 关系:rf=0 且收益对称分布时 sortino ≈ sharpe × √2(负半部分样本约半)——断言同号即可,不做强等式。

---

## B4. Calmar 在 mdd=0 时返回 NaN + sweep 排序 NaN 处理

### 问题
`metrics.py:146`:`calmar = ann_return/abs(mdd) if mdd < 0 else 0` —— 零回撤正收益组合得 0,sweep `--target calmar` 时最优组合排最后。

### 修改方案
1. `metrics.py`:mdd==0 时 `calmar = float("nan")`(NaN 语义="未定义",优于 0 或 inf)。
2. `cli/sweep.py:202` 附近排序键:`key=lambda r: (r.get(target) is not None and not math.isnan(r.get(target, nan)), r.get(target) or 0.0)` —— NaN 排最后,其余正常升降序。注意 `api/jobs.py:352` 的排序同样改。
3. `report_store.py` 的 `_safe_float`/`_sanitize` 已把 NaN→None,报告 JSON 不受影响。

### 测试验证
- `test_calmar_nan_when_no_drawdown`:单调上涨净值 → `math.isnan(m.calmar)`。
- `test_sweep_nan_last`:两组结果(calmar=2.0 与 NaN)→ 排序后 2.0 在前。

---

## B5. monthly_returns 补首月收益

### 问题
`metrics.py:243-244`:`monthly = equity.resample("ME").last().pct_change().dropna()` —— 首月(期初→第一个月末)收益被 dropna 丢掉。

### 修改方案
```python
month_end = equity.resample("ME").last()
# 期初净值作为第一个"上月末"前置,使首月收益 = 首月末/期初 − 1
month_end = pd.concat([equity.iloc[[0]], month_end])
month_end = month_end[~month_end.index.duplicated(keep="last")]
monthly = month_end.pct_change().dropna()
```
(期初值索引用回测首日日期,重采样后重复标签去重保 last。)

### 测试验证
- `test_monthly_first_month_present`:2020-01-01 起、净值首日 100000、1 月末 105000 → monthly 首行(2020-01)== 0.05。
- 月度收益热力图(前端 ReturnsHeatmap)首月不再空白:e2e 目测。

---

## B6. alpha → Jensen alpha + 新增 upside_capture

### 问题
`analytics/trades.py:157-162`:`alpha = ((Rs/Rb) − 1)/n_years` —— 与同行算出的 beta 完全无关,也非 CAGR 差;标注 alpha 误导。另 `downside_capture`(:163-168)无 upside 对称。

### 修改方案
`compare_benchmark()` 内:

```python
# Jensen alpha:α = (R_s − rf) − β(R_b − rf),年化(日化均值的 af 倍)
rf_daily = rf / af
jensen = float((sr.mean() - rf_daily) - beta * (br.mean() - rf_daily)) * af
up_mask = br > 0
up_capture = (
    float(sr[up_mask].sum() / br[up_mask].sum())
    if up_mask.any() and br[up_mask].sum() != 0 else 0.0
)
```

`BenchmarkStats` 字段 `alpha` 语义改为 Jensen alpha(docstring 注明),新增 `upside_capture: float = 0.0`,`to_dict()` 同步。前端 `types/index.ts` 的 BenchmarkStats 镜像同步加字段;前端指标卡片如展示 alpha,文案改"Jensen α"。

### 测试验证
- `test_jensen_alpha`:构造 sr = 1.2×br + 0.0005 的日收益(无噪声)→ beta≈1.2,jensen ≈ 0.0005×af(±1e-6);旧公式值不同(断言已替换)。
- `test_upside_capture`:对称数据 upside/downside capture 均 ≈1;策略只在下跌日减半 → downside≈0.5、upside≈1。

---

## B7. 因子归因暴露滞后一日

### 问题
`analytics/attribution.py:186-193` 的 `factor_attribution`:`expo(t) × fret(t)` 同期相乘;暴露来自 `build_exposure_report` 的**当日收盘**权重(:232-234)。t 日收盘才知道的暴露解释 t 日收益 = 前视。Brinson 侧有 `shift(1)` 保护(:113-116),因子侧没有。

### 修改方案
`attribution.py` `factor_attribution(exposure, factor_returns)` 内部:

```python
expo_lag = exposure.shift(1)  # t 日收益由 t-1 日收盘暴露解释(与 Brinson 一致)
contrib = expo_lag.mul(factor_returns, axis=0)  # 视现有实现形式调整
```

具体改法取决于现有 :186-193 的循环/矩阵形态:若是逐日循环,把当日的 expo 换成 `exposure.iloc[i-1]`;首行(t=0)无暴露,剔除(与 Brinson 首日剔除一致)。`build_exposure_report`(:220-240)无需改(暴露序列本身仍按当日收盘记录,滞后在使用处)。

### 测试验证
文件:`tests/unit/test_attribution.py`(已存在)。
- `test_factor_attribution_no_lookahead`:构造 exposure 在 T 日突变(0→1),factor_returns 全期常数 → T 日贡献应为 0(暴露 T-1 为 0),T+1 日起非零。
- 恒等式回归:现有 Brinson 恒等式测试不受影响(改动仅因子归因);因子归因总贡献 vs 实际超额的方向性 sanity(容许残差项,现有测试若有恒等断言按新口径更新)。

---

## B8. VaR 口径与注释一致化 + 换手率文档化

### 问题
- `metrics.py:149`:`var_95 = -quantile(rets, 0.05)`;收益整体为正时 5% 分位为正 → var_95 为负,与注释"日损失,正"矛盾。
- `metrics.py:208-219` `_turnover`:双边成交额合计/平均净值、未年化;主流通行口径为"单边年化换手"。

### 修改方案
1. VaR:`var_95 = max(0.0, -float(rets.quantile(0.05)))`(CVaR 同理 `max(0, ...)`),docstring 注明"历史法、日度、非负"。
2. 换手率:不改数值(避免破坏历史对比),在 `Metrics` dataclass 字段注释与 `docs/`(或 README 指标表)注明口径:"turnover = Σ|买卖成交额| / 平均净值(双边、区间合计,未年化)";**另**新增字段 `turnover_annual: float`(= turnover × af / n_days,单边口径 = 双边/2)进 `to_dict()`,前端指标卡片标注。

### 测试验证
- `test_var_nonnegative`:全正收益 → var_95 == 0.0。
- `test_turnover_annual`:已知成交额序列手算两个口径断言。

---

## B9. 两套 Fill 命名消除

### 问题
`engine/events.py:50-61` 的 `Fill`(float)与 `portfolio/account.py:26-46` 的 `Fill`(Decimal)同名不同口径,import 时极易混淆(`metrics.py:213-217` 用的是 float 版)。

### 修改方案
- `account.py` 的 Fill 重命名为 `LedgerEntry`(账本分录,Decimal);`Account.fills: list[LedgerEntry]`。
- 引用点:`account.py` 内部(buy/sell/:134,170)、`broker.py`(`self.account.buy/sell` 返回值类型注解)、`io/export.py`(getattr 读字段,不受影响)、`report_store.py`(同上)、`engine/broker.py:175` 用的是 events.Fill 不动。
- `events.py` 的 Fill 保持原名(执行回报,float)。

### 测试验证
- 全量 grep `from djinn.portfolio.account import.*Fill` 为零;`mypy --strict` 通过(类型注解会揪出所有引用点);pytest 全绿。

---

## 验收清单

1. 四个静态检查全过;pytest 全绿(新增 ~20 用例)。
2. 手工对照:同一回测的胜率/盈亏比在改动后应**下降或持平**(佣金摊销 + round-trip 拆分),若上升必有 bug。
3. 前端:`frontend/src/types/index.ts` 的 TradeStats/BenchmarkStats 镜像已同步;`tsc -b --noEmit` 过。

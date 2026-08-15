# 计划 G:选股流水线增强 + 两层择时(因子池月频 + 指标日频)

> 目标读者:执行模型。本文档自包含,与 A~F 计划同规格:每项给出问题/目标、修改方案(文件/代码)、数据来源(如涉及)、测试验证。
> 与 C 计划的关系:C 计划修"因子本身"(数据 PIT、因子库、合成权重、统计检验);本计划修"因子怎么变成持仓"——选股流水线(过滤/行业中性/换手惩罚)与择时覆盖层。**两项共用一次 `FactorPortfolioStrategy` 重构(G0),必须按顺序施工。**
> 总原则:改动后 `pytest -n auto -m "not network and not slow and not benchmark"` 全绿;G0 重构必须附"默认参数下输出与旧实现逐一致"的等价性测试。

## 总览

| # | 改进点 | 类型 | 预估工作量 |
|---|---|---|---|
| G0 | `FactorPortfolioStrategy` 拆 `_select_pool()` 重构(等价性) | 重构 | 0.5 天 |
| G1 | 资格过滤:流动性下限 + 次新 + ST + 停牌 | 功能 | 1 天 |
| G2 | 行业中性选股(每行业 TopK + 补位) | 功能 | 1 天 |
| G3 | 行业暴露上限(策略层权重缩放) | 功能 | 0.5 天 |
| G4 | 换手惩罚(min_score_diff,新旧持仓 diff) | 功能 | 0.5 天 |
| G5 | 择时规则库 `strategy/timing.py`(闸门/出场/入场) | 功能 | 1 天 |
| G6 | Context/引擎 benchmark 通道 | 架构 | 0.5 天 |
| G7 | `FactorTimingStrategy`(继承 + 择时覆盖层) | 功能 | 1.5 天 |
| G8 | 配置 schema + runner 接线 + 前端 types + sweep 轴 | 接线 | 0.5 天 |
| G9 | 端到端验证 + 成交 tag 归因 + 调仓快照透出 | 验证 | 1 天 |

目标流水线(完成后):

```
因子面板 → 截面打分(C 计划负责权重来源)
  → G1 资格过滤(流动性/次新/ST/停牌)
  → G2 行业中性 TopK(或全池 TopN)
  → G4 换手惩罚(得分优势不足不换仓)
  → G3 权重分配(allocation + 行业上限)
  ── 调仓日执行以上;每日叠加 G5~G7 择时层 ──
  → 市场闸门(指数 SMA200 → 仓位上限)
  → 池内出场(SMA20 跌破 / ATR 吊灯)
  → 入场确认(站上 SMA20 + 冷却期)
```

---

## G0. `FactorPortfolioStrategy` 拆 `_select_pool()` 重构

### 目标
现有 `on_bar`(`strategy/library/factor_portfolio.py:64-109`)把"调仓节拍 + 因子打分 + 选股 + 权重 + 下单"写在一个方法里。G1~G4 要在打分与下单之间插入四段处理,G7 要覆写节拍逻辑——先把"选池"抽成独立方法。

### 修改方案
**文件:`src/djinn/strategy/library/factor_portfolio.py`**

1. 把 `on_bar` 的 :70-101(取面板 → 逐因子截面 → 打分 → nlargest → 权重)原样移入新方法:

```python
def _select_pool(self, ctx: Context) -> tuple[list[str], dict[str, float]]:
    """因子打分 → (TopN 名单, 名义权重 dict)。

    防未来函数:只吃 ctx.data <= now 截面(经 _visible_panels)。
    返回空名单表示本日无法选股(数据不足),调用方跳过。
    """
    prices, ohlcv = self._visible_panels(ctx)
    if prices.empty:
        return [], {}
    fundamentals = self._visible_fundamentals(ctx)
    cross: dict[str, pd.Series] = {}
    for f in self._factors:
        try:
            panel = f.compute(prices, ohlcv, fundamentals)
        except Exception as e:
            _log.warning("因子 %s 计算失败 @%s: %s", f.name, ctx.now, e)
            continue
        if len(panel) == 0:
            continue
        cross[f.name] = panel.iloc[-1]
    if not cross:
        return [], {}
    cross_df = pd.DataFrame(cross)
    score = score_cross_section(cross_df, self._scores, self.preprocess)
    selected = score.dropna().nlargest(self.n_stocks).index.tolist()
    if not selected:
        return [], {}
    last_close = prices.iloc[-1]
    price_map = {s: float(last_close[s]) for s in selected if pd.notna(last_close.get(s))}
    scores_map = {s: float(score[s]) for s in selected}
    cov = self._selected_cov(prices, selected)
    weights = self.allocation.target_weights(
        selected, prices=price_map, scores=scores_map, cov=cov
    )
    return selected, weights
```

2. `on_bar` 改为调用方(保留原调出/调入下单语义,并给订单打 tag,供 G9 归因):

```python
def on_bar(self, ctx: Context) -> None:
    n = self._bars_seen
    self._bars_seen += 1
    if n % self.rebalance_freq != 0:
        return
    selected, weights = self._select_pool(ctx)
    if not selected:
        return
    selected_set = set(selected)
    for s, pos in ctx.portfolio.positions.items():
        if pos.qty > 0 and s not in selected_set:
            ctx.order_target_percent(s, 0.0)
            ctx.orders[-1].tag = "rebalance:out"
    for s, w in weights.items():
        ctx.order_target_percent(s, w)
        ctx.orders[-1].tag = "rebalance:in"
```

(`OrderIntent.tag` 已存在并经 `orders_from_intents`(broker.py:283)透传到 `Fill.tag`,前端交易明细可见。)

### 测试验证
- **等价性(安全网)**:`tests/unit/test_strategy_portfolio.py` 现有用例全部不改断言必须仍绿;另加 `test_select_pool_equiv`:固定种子合成面板,重构前后同一 `ctx` 下 `on_bar` 产生的 orders 序列逐一致(symbol/target_percent 顺序与数值)。
- `test_select_pool_returns`:单测 `_select_pool` 返回值的名单与权重键集一致、权重和 ≤1+1e-9。

---

## G1. 资格过滤(流动性 / 次新 / ST / 停牌)

### 目标
打分 TopN 现状不过滤"买不了/不该买"的票:日均成交额过低的僵尸股(滑点巨大)、上市未满 N 天的次新(无锚定价)、ST 股(涨跌停 5% 且风险特异)、当日停牌股。在 `_select_pool` 的打分后插入过滤。

### 修改方案
**文件:`src/djinn/strategy/library/factor_portfolio.py`**

1. 构造函数新增参数(全部可选,默认关闭,保持 G0 等价性):

```python
def __init__(self, ..., 
             min_amount: float | None = None,      # 20 日平均成交额下限(元)
             min_list_days: int | None = None,     # 上市最少交易日数(以数据首行近似)
             exclude_st: bool = False,
             names: dict[str, str] | None = None,  # symbol → 名称(判 ST)
             ) -> None:
```

2. 新增方法,在 `_select_pool` 内 `score = score_cross_section(...)` 之后、`nlargest` 之前调用:

```python
def _tradable(self, ctx: Context, candidates: list[str], prices: Panel, ohlcv: PanelDict) -> list[str]:
    """资格过滤:流动性/次新/ST/当日停牌(无行情)。"""
    out = []
    amount = ohlcv.get(COL_AMOUNT)
    for s in candidates:
        if s not in prices.columns:
            continue
        ser = prices[s].dropna()
        # 当日无行情(union 日历下停牌/未上市)→ 排除
        if len(ser) == 0 or ser.index[-1] < prices.index[-1]:
            continue
        # 次新:数据首行距今日不足 min_list_days 个交易日
        if self.min_list_days is not None and len(ser) < self.min_list_days:
            continue
        # 流动性:近 20 日平均成交额
        if self.min_amount is not None and amount is not None and s in amount.columns:
            amt = amount[s].dropna().iloc[-20:]
            if len(amt) == 0 or float(amt.mean()) < self.min_amount:
                continue
        # ST:名称含 "ST"
        if self.exclude_st and self.names and "ST" in self.names.get(s, "").upper():
            continue
        out.append(s)
    return out
```

`_select_pool` 中:`selected_raw = score.dropna()` → `eligible = self._tradable(ctx, list(selected_raw.index), prices, ohlcv)` → `selected = score[eligible].nlargest(self.n_stocks)...`(注:过滤发生在取 TopN **之前**,保证过滤后仍取满 N 只)。

3. runner 注入(见 G8):`names` 从 universe 解析路径尽量带上(指数成分缓存帧已含 name 列,见 CLAUDE.md"成分名称"约定);拿不到时 `exclude_st=True` 降级为 warning + 跳过。

### 数据来源
- 成交额:`ohlcv[COL_AMOUNT]` 面板(akshare/yahoo 日线均含 amount,已流入 `_visible_panels`,无需新数据);
- 上市天数:**近似** = 数据首行(数据窗口覆盖上市日起时准确;`csv_dir`/长窗口拉取下成立),docstring 注明近似口径;
- ST:股票名称(指数成分缓存 name 列 / `get_stock_name`,见 G8 接线),不新增 provider 调用。

### 测试验证
文件:`tests/unit/test_strategy_portfolio.py` 追加。
- `test_tradable_amount`:两只票成交额分别 1e8/1e6,min_amount=5e7 → 低额票被滤;
- `test_tradable_list_days`:一只票数据仅 60 行,min_list_days=120 → 被滤;
- `test_tradable_st`:names={"600001":"ST  XX"},exclude_st=True → 被滤;names=None → warning 且不滤(降级);
- `test_tradable_suspended`:union 日历下一只票当日无 bar → 被滤;
- 默认值等价性:全部参数 None/False 时与 G0 输出逐一致。

---

## G2. 行业中性选股(每行业 TopK + 补位)

### 目标
现状全池 `nlargest(n)` 在行业极端行情下会全仓同一赛道(如 2020 年白酒)。行业中性:每行业取前 k 名,余额全局补位,保证行业分散同时不遗漏绝对高分票。

### 修改方案
**文件:`src/djinn/strategy/library/factor_portfolio.py`**

1. 构造函数新增:`industry_neutral: bool = False`、`industry_map: dict[str, str] | None = None`(runner 注入,与归因同款 `_industry_map_safe`)。

2. 新增方法,替换 `_select_pool` 中的 `nlargest`:

```python
@staticmethod
def _pick_neutral(score: pd.Series, industry_map: dict[str, str], n: int) -> list[str]:
    """行业中性 TopK:每行业 ⌈n/行业数⌉ 名,超额砍尾、欠额全局补位。"""
    groups: dict[str, list[str]] = {}
    for s in score.index:
        groups.setdefault(industry_map.get(s, "未知"), []).append(s)
    # groups 内成员保持 score 降序(score 已排序则按 index 顺序)
    k = max(1, -(-n // max(1, len(groups))))  # ceil(n/行业数)
    picked: list[str] = []
    ranked = score.sort_values(ascending=False)
    for _ind, members in groups.items():
        ordered = [s for s in ranked.index if s in members]
        picked.extend(ordered[:k])
    if len(picked) > n:                      # 超额:全局按分砍尾
        picked = [s for s in ranked.index if s in picked][:n]
    elif len(picked) < n:                    # 欠额:全局剩余按分补足
        rest = [s for s in ranked.index if s not in picked]
        picked.extend(rest[: n - len(picked)])
    return picked
```

`_select_pool` 中:

```python
if self.industry_neutral and self.industry_map:
    selected = self._pick_neutral(score[eligible].dropna(), self.industry_map, self.n_stocks)
else:
    if self.industry_neutral and not self.industry_map:
        _log.warning("industry_neutral=True 但无行业映射,退化为全池 TopN")
    selected = score[eligible].dropna().nlargest(self.n_stocks).index.tolist()
```

### 数据来源
行业映射:复用 `cli/runner.py` 的 `_industry_map_safe`(akshare 东财行业,归因同款;港股/美股 provider 若无行业则退化为全池 TopN + warning)。**无新增数据源**。

### 测试验证
- `test_pick_neutral_basic`:3 行业(A/B/C)各 4 只、score 单调 → n=6,k=2,每行业取前 2;
- `test_pick_neutral_topup`:C 行业仅 1 只 → 取 1 只后从全局剩余按分补足 6 只,且绝对最高分不被行业配额挤掉;
- `test_pick_neutral_trim`:2 行业、n=3 → k=2 第一遍取 4 只,砍尾到 3,保留全局前 3;
- `test_pick_neutral_missing_map`:industry_map 缺某票 → 归"未知"组参与;
- 默认参数等价性(同 G1)。

---

## G3. 行业暴露上限(策略层权重缩放)

### 目标
引擎层 `risk.max_sector_weight` 只对 target_percent 订单做事后截断(且 B16:对 size 单无效)。策略层在权重分配后、下单前做行业缩放,语义更直接。

### 修改方案
**文件:`src/djinn/strategy/library/factor_portfolio.py`**

1. 构造函数新增:`max_sector_weight: float | None = None`(如 0.3)。

2. `_select_pool` 内 `weights = self.allocation.target_weights(...)` 之后追加:

```python
def _apply_sector_cap(self, weights: dict[str, float]) -> dict[str, float]:
    """行业权重超过上限时等比缩放到上限;腾出的权重留现金(不再分配,与 G7 闸门语义一致)。"""
    if self.max_sector_weight is None or not self.industry_map:
        return weights
    cap = self.max_sector_weight
    by_ind: dict[str, float] = {}
    for s, w in weights.items():
        by_ind[self.industry_map.get(s, "未知")] = by_ind.get(self.industry_map.get(s, "未知"), 0.0) + w
    scale = {ind: min(1.0, cap / total) if total > cap else 1.0 for ind, total in by_ind.items()}
    if all(v == 1.0 for v in scale.values()):
        return weights
    return {s: w * scale.get(self.industry_map.get(s, "未知"), 1.0) for s, w in weights.items()}
```

(腾出部分**留现金不二次分配**——二次分配会反复触发同一问题且把风险集中到次优行业;docstring 注明。)

### 测试验证
- `test_sector_cap_scales`:3 只同行业票各占 0.2(行业合计 0.6),cap=0.3 → 每只缩到 0.1,总仓位 0.7(0.3 留现金);
- 无 industry_map / cap=None → 原样返回(等价性)。

---

## G4. 换手惩罚(min_score_diff)

### 目标
现状调仓日新旧池全量 diff,得分优势微乎其微也换仓,贡献大量无效换手(`rank_turnover` 只展示不消费)。规则:**新票的得分优势必须 ≥ min_score_diff(zscore σ 单位)才换入**,否则继续持有对应老票(即使它已跌出 TopN)。实测可砍 30~50% 换手且收益影响很小。

### 修改方案
**文件:`src/djinn/strategy/library/factor_portfolio.py`**

1. 构造函数新增:`min_score_diff: float = 0.0`(建议 0.3~1.0,0=现状)。

2. `_select_pool` 内,`selected` 确定后、权重分配前:

```python
def _apply_turnover_penalty(
    self, ctx: Context, selected: list[str], score: pd.Series
) -> list[str]:
    """换手惩罚:新入选票相对被替换持仓票的得分优势不足 min_score_diff 时保留老票。"""
    if self.min_score_diff <= 0:
        return selected
    holdings = [s for s, p in ctx.portfolio.positions.items() if p.qty > 0]
    keep = [s for s in holdings if s not in selected and s in score.index]
    if not keep:
        return selected
    new_in = [s for s in selected if s not in holdings]
    out = list(selected)
    # 候选替换对:按得分升序的新票 ↔ 按得分降序的老票,逐对比较
    for s_old in sorted(keep, key=lambda s: -score[s]):
        if not new_in:
            break
        s_new = min(new_in, key=lambda s: score[s])
        if score[s_new] - score[s_old] < self.min_score_diff:
            out.remove(s_new)
            out.append(s_old)
            new_in.remove(s_new)
    return out
```

注意语义:保留的老票继续参与后续权重分配(按其 score,与入选票同一通道),权重自然反映其当前得分水平。

### 测试验证
- `test_turnover_penalty_keeps_old`:持仓 X(score=0.8)、新票 Y(score=1.0),min_score_diff=0.5 → 不换,Y 被拦,X 留在 selected;
- `test_turnover_penalty_swaps_when_big_gap`:Y score=1.5 → 换入;
- 老票不在 score.index(数据缺失)→ 不保留(安全);
- `min_score_diff=0` → 与 G0 逐一致(等价性)。

---

## G5. 择时规则库 `strategy/timing.py`

### 目标
为组合策略提供可插拔的日频择时组件:**市场闸门**(指数趋势过滤总仓位)、**个股出场**(趋势破坏提前卖)、**入场确认**(不追跌)。规则对象自带最小状态,全部只吃 ≤now 数据,增量 O(1) 更新(不引入 O(T²) 全历史 rolling)。

### 修改方案
**新文件:`src/djinn/strategy/timing.py`**(完整代码骨架见下,可直接照抄落地)

```python
"""择时规则库:市场闸门 / 个股出场 / 入场确认(供组合策略叠加)。

所有规则以增量 deque 缓冲维护状态(O(1)/标的/日),只吃历史 append 数据,
天然无未来函数。t 日判断、t+1 成交由引擎撮合保证。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MarketRegimeFilter:
    """指数 SMA 闸门:收盘 < SMA(window) → 仓位上限 floor,否则 1.0。"""

    window: int = 200
    floor: float = 0.0
    _closes: deque = field(default_factory=lambda: deque(maxlen=210))

    def update(self, close: float | None) -> None:
        if close is not None and close > 0:
            self._closes.append(close)

    def exposure_cap(self) -> float:
        if len(self._closes) < self.window:
            return 1.0  # 暖机期放行
        sma = float(np.mean(list(self._closes)[-self.window:]))
        return 1.0 if self._closes[-1] > sma else self.floor


class ExitRule:
    """个股出场规则基类。"""

    def update(self, sym: str, o: float, h: float, l: float, c: float) -> None: ...
    def should_exit(self, sym: str) -> bool: return False
    def arm(self, sym: str, price: float) -> None: ...
    def disarm(self, sym: str) -> None: ...


@dataclass
class SMABreakExit(ExitRule):
    """收盘跌破 SMA(window) → 出场。无 arm 状态。"""

    window: int = 20
    _closes: dict[str, deque] = field(default_factory=dict)

    def update(self, sym, o, h, l, c):
        buf = self._closes.setdefault(sym, deque(maxlen=self.window + 5))
        buf.append(c)

    def should_exit(self, sym) -> bool:
        buf = self._closes.get(sym)
        if not buf or len(buf) < self.window:
            return False
        return buf[-1] < float(np.mean(list(buf)[-self.window:]))


@dataclass
class ATRTrailingExit(ExitRule):
    """吊灯止损:收盘 < peak − mult × ATR(window);peak 自 arm 起跟踪最高价。"""

    mult: float = 3.0
    window: int = 14
    _bars: dict[str, deque] = field(default_factory=dict)
    _peak: dict[str, float] = field(default_factory=dict)

    def update(self, sym, o, h, l, c):
        buf = self._bars.setdefault(sym, deque(maxlen=self.window + 10))
        buf.append((h, l, c))
        if sym in self._peak:
            self._peak[sym] = max(self._peak[sym], h)

    def arm(self, sym: str, price: float) -> None:
        self._peak[sym] = price

    def disarm(self, sym: str) -> None:
        self._peak.pop(sym, None)

    def should_exit(self, sym) -> bool:
        if sym not in self._peak:
            return False
        buf = self._bars.get(sym)
        if not buf or len(buf) < self.window + 1:
            return False
        rows = list(buf)
        trs = [
            max(h - l, abs(h - pc), abs(l - pc))
            for (h, l, _c), (_, _, pc) in zip(rows[1:], rows[:-1])
        ]
        atr = float(np.mean(trs[-self.window:]))
        return rows[-1][2] < self._peak[sym] - self.mult * atr


@dataclass
class AboveSMAConfirm:
    """入场确认:收盘站上 SMA(window) 才允许买入;数据不足不拦截。"""

    window: int = 20

    def entry_ok(self, closes: pd.Series) -> bool:
        if closes is None or len(closes) < self.window:
            return True
        return float(closes.iloc[-1]) > float(closes.iloc[-self.window:].mean())
```

### 测试验证
文件:`tests/unit/test_factor_timing.py`(新建)。
- `test_regime_filter`:合成指数序列,SMA200 上→cap=1.0;下穿→cap=floor;不足 200 日→1.0;
- `test_sma_break_exit`:close 第 21 日跌破 SMA20 → 当日 True,前一日 False;
- `test_atr_trailing`:arm(100) 后 peak 只升不降;构造 ATR=2、mult=3,收盘 < peak−6 → True;
- `test_above_sma_confirm`:站上/跌破两态;数据不足 → True(不拦截)。

---

## G6. Context/引擎 benchmark 通道

### 目标
现状 benchmark 仅用于净值对比曲线(`cli/runner.py:429` 传 `engine.run(benchmark=...)`),策略访问不到;市场闸门(G5)需要它。

### 修改方案
1. **`src/djinn/strategy/base.py`**:
   - `Context.__init__` 增加可选参数 `benchmark: tuple[str, DataView] | None = None`(symbol + 单标的 DataView);
   - `Context` 增加便捷方法:

```python
def benchmark_close(self) -> float | None:
    """基准最近收盘价(无基准/无数据 → None)。"""
    if self.benchmark is None:
        return None
    sym, view = self.benchmark
    try:
        return view.latest(sym, "close")
    except Exception:
        return None
```

2. **`src/djinn/engine/event_engine.py`**(:175-177 构造 ctx 处):

```python
bench_tuple = None
if benchmark is not None:
    bench_tuple = (benchmark.symbol, DataView({benchmark.symbol: benchmark}, ts_date))
ctx = Context(now=ts_date, data=data_view, portfolio=portfolio_view, benchmark=bench_tuple)
```

(防未来函数不变量保持:benchmark DataView 与主 DataView 同一 `now` 切片。)

### 测试验证
- `test_benchmark_in_ctx`:带 benchmark 跑最小回测,策略内 `ctx.benchmark_close()` 等于基准当日收盘、且不等于次日值(抽查 3 日);无 benchmark 时返回 None 不抛错。

---

## G7. `FactorTimingStrategy`(继承 + 择时覆盖层)

### 目标与语义
继承 `FactorPortfolioStrategy`(G0 重构后自动获得 G1~G4 增强),覆写 `on_bar` 叠加日频择时。三条语义(实现与测试都围绕它们):

1. **因子判决优先**:掉出池 → 立即清零,不择时;
2. **被择时挡住的权重份额留现金**,不摊给其他票;
3. **冷却期 + 调仓日重置**:指标卖出的票冷却 `cooldown_days`(按交易日),调仓日清空冷却名单。

### 修改方案
**新文件:`src/djinn/strategy/library/factor_timing.py`**

```python
"""因子选股(调仓频)+ 指标择时(日频)两层组合策略。"""

from __future__ import annotations

from djinn.strategy.base import SCOPE_PORTFOLIO, Context
from djinn.strategy.library.factor_portfolio import FactorPortfolioStrategy
from djinn.strategy.timing import (
    ATRTrailingExit,
    AboveSMAConfirm,
    ExitRule,
    MarketRegimeFilter,
)


class FactorTimingStrategy(FactorPortfolioStrategy):
    scope = SCOPE_PORTFOLIO

    def __init__(
        self,
        *args,
        regime: MarketRegimeFilter | None = None,
        exit_rule: ExitRule | None = None,
        entry_confirm: AboveSMAConfirm | None = None,
        cooldown_days: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._regime = regime
        self._exit = exit_rule
        self._confirm = entry_confirm
        self._cooldown_days = max(0, int(cooldown_days))
        self._pool: list[str] = []
        self._base_w: dict[str, float] = {}
        self._inactive: dict[str, int] = {}  # sym -> 冷却起始 bars_seen

    def on_bar(self, ctx: Context) -> None:
        n = self._bars_seen
        self._bars_seen += 1

        # 0. 更新规则缓冲(基准 + 池内 ∪ 持仓)
        if self._regime is not None:
            self._regime.update(ctx.benchmark_close())
        if self._exit is not None:
            for sym in set(self._pool) | {s for s, p in ctx.portfolio.positions.items() if p.qty > 0}:
                if sym not in ctx.data:
                    continue
                try:
                    o = ctx.data.latest(sym, "open")
                    h = ctx.data.latest(sym, "high")
                    l = ctx.data.latest(sym, "low")
                    c = ctx.data.latest(sym, "close")
                except Exception:
                    continue
                self._exit.update(sym, o, h, l, c)

        # A. 市场闸门
        cap = self._regime.exposure_cap() if self._regime is not None else 1.0

        # B. 调仓日:因子重选池(出池即卖,因子判决优先)
        if n % self.rebalance_freq == 0:
            selected, weights = self._select_pool(ctx)
            for s, pos in ctx.portfolio.positions.items():
                if pos.qty > 0 and s not in selected:
                    ctx.order_target_percent(s, 0.0)
                    ctx.orders[-1].tag = "rebalance:out"
                    if self._exit is not None:
                        self._exit.disarm(s)
            self._pool, self._base_w = selected, weights
            self._inactive.clear()  # 冷却重置

        # C. 每日:池内出场检查
        if self._exit is not None:
            for s, pos in list(ctx.portfolio.positions.items()):
                if pos.qty <= 0 or s not in self._pool:
                    continue
                if self._exit.should_exit(s):
                    ctx.order_target_percent(s, 0.0)
                    ctx.orders[-1].tag = f"exit:{type(self._exit).__name__}"
                    self._inactive[s] = n
                    self._exit.disarm(s)

        # D. 每日:入场确认(池内未持仓、过冷却、指标确认)
        for s in self._pool:
            if ctx.portfolio.has_position(s):
                continue
            if s in self._inactive and n - self._inactive[s] < self._cooldown_days:
                continue
            if s not in ctx.data:
                continue
            if self._confirm is not None:
                hist = ctx.data.history(s, "close", self._confirm.window + 5)
                if not self._confirm.entry_ok(hist):
                    continue
            w = self._base_w.get(s, 0.0) * cap  # 名义权重 × 闸门;被挡份额留现金
            if w > 0:
                ctx.order_target_percent(s, w)
                ctx.orders[-1].tag = f"entry:cap={cap:.2f}"
                if isinstance(self._exit, ATRTrailingExit):
                    try:
                        self._exit.arm(s, ctx.data.latest(s, "close"))
                    except Exception:
                        pass
```

要点说明(实现者注意):
- **ATR 吊灯的 arm 时机**:下单即以 t 日收盘 arm(t+1 才成交,峰值跟踪从下单日起算,误差 1 日可接受;docstring 注明)。成交失败(拒单)时 arm 状态无害——无持仓则 `should_exit` 段不会命中该票(`pos.qty <=0` 跳过),下次下单会重 arm。
- **`_select_pool` 的 G1~G4 增强对本策略自动生效**(继承)。
- **注册**:加入 `strategy/library/__init__.py` 的 STRATEGY_REGISTRY(名称 `FactorTiming`),前端策略页/param_schema 自动出现。

### 测试验证
文件:`tests/unit/test_factor_timing.py`。
- `test_exit_and_cooldown`:合成 3 票面板,因子恒定选 A/B/C;B 在调仓后第 5 日跌破 SMA20 → 当日出现 target=0 订单且 tag 含 `exit`;冷却期内(5 交易日)即便涨回不买;冷却结束且 close>SMA20 → 出现 `entry` 单;
- `test_regime_scales_weights`:基准跌破 SMA200(floor=0.3)后,新 entry 单 target == base_w×0.3;
- `test_out_of_pool_immediate`:掉出池的票**当日**清零(不择时);
- `test_cash_left_for_blocked`:1 只票被出场确认拦截时,总目标权重和 < 1(差额留现金,不摊给其他票);
- **无未来函数**:`_select_pool` 截面与 FactorEngine 单独对当日计算一致(复用 test_api_alpha 的 stub provider 模式);on_bar 内断言 `ctx.data[s].index[-1] <= ctx.now`;
- `test_factor_timing_equiv_when_no_timing`:regime/exit/confirm 全 None 时,行为与父类逐一致(订单序列相等)。

---

## G8. 配置 schema + runner 接线 + 前端 types + sweep 轴

### 修改方案

**1. `src/djinn/config/models.py`** — `StrategyConfig` 增加两块(均为可选,`extra="forbid"` 保持):

```python
class SelectionConfig(BaseModel):
    """选股流水线增强(组合策略)。"""
    min_amount: float | None = None          # 20 日平均成交额下限
    min_list_days: int | None = None         # 上市最少交易日数
    exclude_st: bool = False
    industry_neutral: bool = False
    max_sector_weight: float | None = None   # 行业暴露上限(0,1]
    min_score_diff: float = 0.0              # 换手惩罚阈值(zscore σ)

class TimingConfig(BaseModel):
    """择时覆盖层(组合策略)。"""
    market_filter: dict[str, Any] | None = None   # {type:"sma", window:200, floor:0.3}
    exit_rule: dict[str, Any] | None = None       # {type:"sma_break",window:20} | {type:"atr_trail",mult:3.0,window:14}
    entry_confirm: dict[str, Any] | None = None   # {type:"above_sma",window:20}
    cooldown_days: int = 5
```

`StrategyConfig` 加 `selection: SelectionConfig | None = None`、`timing: TimingConfig | None = None`。

**2. `src/djinn/cli/runner.py`** — `_build_factor_portfolio`(:178-202)扩展:

```python
def _build_factor_portfolio(cfg, *, fundamentals=None, registry=None, market=None):
    ...现有因子/scores/n_stocks/rebalance_freq 解析...
    sel = cfg.strategy.selection
    timing = cfg.strategy.timing
    kwargs: dict[str, Any] = {}
    if sel is not None:
        kwargs.update(
            min_amount=sel.min_amount,
            min_list_days=sel.min_list_days,
            exclude_st=sel.exclude_st,
            min_score_diff=sel.min_score_diff,
        )
        if sel.industry_neutral or sel.max_sector_weight is not None:
            kwargs["industry_map"] = _industry_map_safe(registry, symbols) if registry else None
            kwargs["industry_neutral"] = sel.industry_neutral
            kwargs["max_sector_weight"] = sel.max_sector_weight
        if sel.exclude_st:
            kwargs["names"] = _resolve_names(cfg, registry)   # 见下;失败 → warning + None
    cls = FactorTimingStrategy if timing is not None else FactorPortfolioStrategy
    if timing is not None:
        kwargs.update(_build_timing(timing))   # 见下小工厂
    return cls(factors=..., scores=..., **kwargs)
```

- `_resolve_names(cfg, registry)`:优先用 universe.index 成分缓存帧的 name 列(复用 `_index_components` 同源的带名称版本;`get_index_component_names` 已存在,见 CLAUDE.md"成分名称"约定);显式 symbols 走 `get_stock_name`(逐个,带 try/except);全部失败 → `None` + warning。
- `_build_timing(t)` 小工厂:`market_filter.type=="sma"` → `MarketRegimeFilter(window, floor)`;`exit_rule.type` → `SMABreakExit(window)` / `ATRTrailingExit(mult, window)`;`entry_confirm.type=="above_sma"` → `AboveSMAConfirm(window)`;未知 type → `ConfigError`(列出允许值)。
- `build_strategy` 签名加 `registry=None, market=None` 透传;`run_backtest` 调用处补上这两个参数。
- `_industry_map_safe` 现已存在于 runner(归因用),直接复用。

**3. 配置示例**(`configs/portfolio.example.yaml` 追加注释块):

```yaml
strategy:
  name: FactorPortfolio
  factor_weights: {momentum: 1.0, ep: 0.5, roe: 0.5}
  n_stocks: 20
  rebalance_freq: 20
  selection:
    min_amount: 50000000        # 20日平均成交额 ≥ 5000万
    min_list_days: 120
    exclude_st: true
    industry_neutral: true
    max_sector_weight: 0.3
    min_score_diff: 0.5
  timing:
    market_filter: {type: sma, window: 200, floor: 0.3}
    exit_rule: {type: sma_break, window: 20}
    entry_confirm: {type: above_sma, window: 20}
    cooldown_days: 5
```

**4. sweep 轴**:`cli/sweep.py` 的 `ALLOWED_SWEEP_AXES` 追加 `strategy.min_score_diff`(`_apply_param` 写 `cfg.strategy.selection.min_score_diff`,selection 为 None 时先建默认实例);择时参数暂不进轴(dict 嵌套,文本模式可后续支持)。前端 SweepPage 白名单硬编码同步加一行。

**5. 前端 types**:`frontend/src/types/index.ts` 的 `StrategyConfig` 镜像加:

```ts
selection?: {
  min_amount?: number | null
  min_list_days?: number | null
  exclude_st?: boolean
  industry_neutral?: boolean
  max_sector_weight?: number | null
  min_score_diff?: number
} | null
timing?: {
  market_filter?: Record<string, unknown> | null
  exit_rule?: Record<string, unknown> | null
  entry_confirm?: Record<string, unknown> | null
  cooldown_days?: number
} | null
```

(BacktestRunPage 的组合策略表单块如已有 factor_weights/n_stocks 输入,新增字段按同模式追加;API 请求体即 BacktestConfig,新字段自动透传,无需改路由。)

### 测试验证
- `tests/unit/test_config.py`:`load_config` 解析含 selection/timing 的 YAML;未知 timing.type → ConfigError;selection 默认值全 None 时行为等价于缺省;
- `tests/unit/test_api_alpha.py` 模式:`build_strategy` 返回类型(timing 非空 → FactorTimingStrategy);`_build_timing` 各 type 映射;
- sweep:`strategy.min_score_diff: [0, 0.5]` 两组合跑通且 config_summary 含该值。

---

## G9. 端到端验证 + 成交 tag 归因 + 调仓快照透出

### 目标
1. **tag 归因**(G0 已埋点):导出交易明细含 `rebalance:in/out`、`exit:*`、`entry:cap=*`;验证前端 TradesTable 与 `io/export.py` 的 CSV 输出 tag 列可见。
2. **调仓快照**(可选增强,供前端"每次调仓的池子与得分"展示):策略在 `_select_pool` 成功后把 `(date, selected, top scores)` append 到 `self.selection_log: list[dict]`;`cli/runner.py` 在 `build_report` 后把 `getattr(strategy, "selection_log", None)` 写入 `report.meta["selection_log"]`(report_store 序列化时经 `_sanitize` 自动转 JSON)。前端 ReportDetail 加折叠面板展示。
3. **端到端对比**:同一 factor_weights/universe 下三档跑同一区间:
   - `FactorPortfolioStrategy`(基线)
   - +selection(行业中性 + 换手惩罚)
   - +timing(两层)
   断言方向:行业集中度(HHI)下降、换手下降、熊市段最大回撤下降;写入基准测试报告。

### 测试验证
- `tests/unit/test_factor_timing.py::test_end_to_end_synthetic`:合成含一段熊市的 10 票面板 + 指数,三档对比断言上述方向(容差内);
- `test_tag_in_trades`:回测结果 trades 的 tag 集合 ⊇ {"rebalance:in","rebalance:out"};开启择时后出现 `exit:`/`entry:` 前缀 tag;
- `test_selection_log_serialized`:`/backtests/{id}/report` 响应含 `meta.selection_log`(stub 模式);
- 手工:`configs/portfolio.example.yaml` 开全量 selection+timing 跑 CSI300 一年,检查交易 CSV 的 tag 分布与调仓快照。

---

## 实施顺序与验收

**顺序**:G0(重构+等价性)→ G1~G4(可并行,但都改 `_select_pool`,建议串行避免冲突)→ G5(规则库,独立)→ G6(benchmark 通道,独立)→ G7(依赖 G0/G5/G6)→ G8(接线)→ G9(验证)。

**验收清单**:
1. `ruff check src/djinn tests` / `black --check` / `mypy --strict src/djinn` 全过;
2. `pytest -n auto -m "not network and not slow and not benchmark"` 全绿(含 G0 等价性 + 新增 ~35 用例);
3. 默认配置(无 selection/timing)回测结果与改动前**逐值一致**(回归硬指标);
4. 前端 `tsc -b --noEmit` 过;策略页出现 FactorTiming,SweepPage 白名单含 `strategy.min_score_diff`;
5. CLAUDE.md 回写:策略 API 段(两层策略 + benchmark 通道)、sweep 轴白名单段。

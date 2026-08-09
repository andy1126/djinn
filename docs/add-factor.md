# 添加新因子

本文档说明如何向 djinn 添加一个新的横截面因子。因子系统采用**手动注册**（无自动发现），添加一个因子通常只需要 **2 步**：创建因子类 + 注册。前端（因子下拉、参数表单、因子分析 / 多因子诊断 / 选股打分）会自动出现，无需修改前端代码。

## 整体流程

```
Step 1: 在 factor/library/ 新建因子类文件(继承 Factor,覆写 compute)
Step 2: 在 factor/library/__init__.py 注册进 FACTOR_REGISTRY
        ↓
完成:前端 /factors、因子分析、多因子诊断、选股自动支持新因子
```

## Step 1：创建因子类文件

在 `src/djinn/factor/library/` 下按类目新建文件，继承 `Factor`，覆写 `compute()`。

```python
"""新因子:一句话说明本文件涵盖的因子。"""

from __future__ import annotations

from djinn.factor.base import Factor, Panel, PanelDict, param


class MyFactor(Factor):
    """一句话描述(会显示在前端下拉与详情页)。"""

    name = "my_factor"          # 因子名(全库唯一,前端下拉用)
    category = "my_category"    # 类目(分组展示用)

    # 可选:声明式参数(自动生成前端参数表单)
    period = param(20, min=2, max=250, description="回看窗口(交易日)")

    def compute(
        self, prices: Panel, ohlcv: PanelDict, fundamentals: PanelDict
    ) -> Panel:
        # prices:       date × symbol 收盘价宽表
        # ohlcv:        {open/high/low/volume/amount} → date × symbol 宽表
        # fundamentals: {pe/pb/market_cap/...} → date × symbol(point-in-time 对齐)
        # 返回:同形状 date × symbol 的因子值面板
        return prices.pct_change(int(self.period))
```

### `compute()` 的契约

- **输入**：`prices` / `ohlcv` / `fundamentals` 三个面板，均为 `date × symbol` 宽表（`index=date`, `columns=symbol`）
- **输出**：与 `prices` 同形状的因子值宽表（`date × symbol`）
- **防未来函数**：`date t` 只能使用 `≤ t` 的数据（pandas `shift` / `rolling` 天然满足）；基本面已按 `announce_date` point-in-time 对齐，无需自行处理
- **参数读取**：声明式参数在 `compute()` 里通过 `self.<param名>` 读取（如 `int(self.period)`）

## Step 2：注册进 FACTOR_REGISTRY

编辑 `src/djinn/factor/library/__init__.py`，两处改动：

```python
from djinn.factor.library.my_factor import MyFactor   # ① 新增 import

FACTOR_REGISTRY: dict[str, type[Factor]] = {
    ...
    "my_factor": MyFactor,   # ② 新增注册(键 = 因子名)
}
```

（可选）把类名加进 `__all__` 列表。

## 三种数据输入模式

根据因子依赖的数据类型，选择对应的取数方式：

| 输入 | 拿到什么 | 适用场景 | 参考实现 |
|------|---------|---------|---------|
| `prices` | 收盘价宽表 | 动量 / 反转 / 波动 | `library/momentum.py` |
| `ohlcv` | `{open/high/low/volume/amount}` 宽表 | 换手率 / 量价因子 | `library/liquidity.py` |
| `fundamentals` | `{pe/pb/market_cap/roe/...}` 宽表 | 估值 / 质量 / 规模 | `library/value.py`、`size.py` |

### 用 `ohlcv` / `fundamentals` 时

`ohlcv` / `fundamentals` 是 `dict[str, Panel]`，按字段名取宽表；缺字段时要兜底：

```python
from djinn.data.schema import COL_AMOUNT

amount = ohlcv.get(COL_AMOUNT)
if amount is None:
    return pd.DataFrame(float("nan"), index=prices.index, columns=prices.columns)
```

基本面字段有便捷工具 `fund_panel`（对齐 + 缺失兜底 NaN），参考 `library/value.py`：

```python
from djinn.data.schema import COL_PE
from djinn.factor.library._util import fund_panel

pe = fund_panel(fundamentals, COL_PE, prices)   # 缺失全 NaN,已对齐 prices
pos = pe.where(pe > 0)                           # 负 / 零估值无意义 → NaN
return 1.0 / pos
```

## 基本面字段白名单

`fundamentals` 只能取 `engine.py` 的 `DEFAULT_FUNDAMENTAL_FIELDS` 列出的字段：

```
market_cap, float_cap, pe, pb, ps, roe, gross_margin, revenue_yoy, profit_yoy
```

**如果新因子需要白名单之外的字段**（如 `debt_ratio`），需要额外改动：
1. `src/djinn/factor/engine.py` — `DEFAULT_FUNDAMENTAL_FIELDS` 加字段名
2. `src/djinn/data/providers/fundamentals_router.py` + `src/djinn/data/schema.py` — 确认该字段能由数据源（akshare / yahoo）提供并规范化
3. 在对应 provider 的 `get_fundamentals` 补充取值

## 可参考的现有因子模板

- **纯价格（最简单）**：`library/momentum.py`
- **带参数 + OHLCV**：`library/liquidity.py`（换手率 = `ohlcv[COL_AMOUNT]` / `fundamentals[COL_FLOAT_CAP]`）
- **纯基本面**：`library/value.py`（`fund_panel` + 倒数）
- **复杂计算**：`library/volatility.py`（rolling 标准差 / beta）、`library/size.py`（`np.log`）

## 验证

```bash
# ① 注册检查:新因子应出现在列表
curl -s http://localhost:8000/factors | python3 -m json.tool

# ② 全量测试确保不破坏现有因子
pytest -m "not network"

# ③ 端到端:前端因子分析页下拉应出现新因子,可提交任务
```

## 注意事项

- **因子名全库唯一**：`name` 作为 `FACTOR_REGISTRY` 的键，重复会覆盖
- **类目自由字符串**：`category` 用于前端分组，无枚举约束
- **参数约束**：`param()` 声明 `min/max/choices`，越界由参数系统自动校验
- **新增 `param` 描述必须写**：`description` 会显示在前端表单上，缺省会误导用户
- **勿忘 `__meta__` 约定**：因子分析任务靠 `__meta__` 恢复输入（见 CLAUDE.md），新增因子不影响此机制

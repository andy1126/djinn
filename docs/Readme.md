# djinn 改进计划索引

本目录 6 份计划文件由全库代码审查(252 项单测全绿基线 + 4 路深度审查 + 关键结论逐条人工验证)产出,供执行模型按序施工。每份计划自包含:问题(文件:行号)、修改方案(含代码)、数据来源(如涉及)、测试验证(测试文件/用例/断言/运行命令)。

## 计划清单与依赖

| 文件 | 主题 | 改进点数 | 依赖关系 |
|---|---|---|---|
| [A-engine-correctness.md](A-engine-correctness.md) | 回测内核:撮合、费用与交易约束正确性 | 12 | 无前置;A6(停牌结构化)是 A5(限价单)前置,按文件内编号顺序做 |
| [B-analytics-metrics.md](B-analytics-metrics.md) | 指标、交易统计与归因口径 | 9 | 无前置;与 A 可并行 |
| [C-factor-alpha.md](C-factor-alpha.md) | 因子与 alpha 研究层(含估值数据源、8 新因子、FMB、ICIR 合成) | 15 | 无前置;C1 先行(它是 C7 质量类因子的字段基础) |
| [D-performance.md](D-performance.md) | 性能优化(含基准场景 S1/S2/S3) | 12 | **D7(并发拉取)必须先完成 E1(缓存线程安全)**;其余独立 |
| [E-api-platform.md](E-api-platform.md) | API、数据层与平台化(缓存锁/取消/排队/鉴权) | 14 | E1 是 D7 前置;E4/E5 联动;E10 后接 F 的 WS 改造 |
| [F-frontend.md](F-frontend.md) | 前端(WS/选股表格化/persist/bundle/暗色等) | 20 | F17(停止按钮)依赖 E4;F5 部分依赖 E6/E8;F1 的 WS 通用化配合 E10;其余独立 |

## 推荐施工顺序

**阶段 0(止血,1~2 天,全部并行可做)**:
- A1(HK 印花税)、A2(卖出零股)、B2(n_trades)、C1(因子字段校验)、C4(RSI)、E1(缓存锁)、D6(covers 修复)、A7(基准 NaN)
- 这批是"已验证的正确性 bug",改动小、测试明确、直接提升结果可信度。

**阶段 1(核心闭环,~2 周)**:
- C2(A 股估值历史数据源 —— 最重要的单项,消除估值因子前视)
- A3(先卖后买)+ A4(口径统一)+ A5(死字段)
- B1(round-trip 交易统计)
- F2(选股表格化 + 一键回测)+ F1(WS 修复)+ F3(config persist)
- E2(阻塞调用 to_thread)+ E3(registry 单例)

**阶段 2(研究深化,~2 周)**:
- C8(Newey-West + FMB)、C9(ICIR 合成器)、C7(因子扩充)、C5(neutralize 接线)
- D1(signals 预计算)+ D2(equity 缓存)+ D3(lookback 截断)
- E4(取消)+ E5(排队)+ E8(鉴权)

**阶段 3(平台完善)**:
- A9(退市)、A10(公司行为)、A11(成交量上限)
- C10(正交化)、D5(稀疏序列化)、D7(并发,在 E1 后)
- F8(bundle)、F14(深链)、F16(通知中心)、F18(暗色)

## 跨计划公共约定

1. **等价性测试是性能改动的安全网**:D1/D3/D4/D8/D9/D12 必须附"新旧输出逐值相等"断言,不可省略。
2. **防未来函数三处高危点**(改错即污染所有回测):
   - C2/C3:估值序列 PIT 对齐(announce_date 生效、快照退化告警);
   - C9:ICIR 权重的 IC 序列必须右移 holding_period(`ic_effective(t) = ic(t−p)`);
   - B7:因子归因暴露 shift(1)。
3. **后端 schema 改动同步前端** `frontend/src/types/index.ts`(B1 的 n_round_trips、B6 的 upside_capture、E4 的 cancelled 状态等,各计划内已标注)。
4. **Decimal 不变量**:账本层(现金/股数/费用)只用 `decimal.Decimal`;统计/指标层 float64;B1 的 round-trip 在统计层用 float 属合规。
5. 每个阶段完成后:`ruff check src/djinn tests` + `black src/djinn tests` + `mypy --strict src/djinn` + `pytest -n auto -m "not network and not slow and not benchmark"` 全绿;前端 `tsc -b --noEmit` + `vite build` 过。
6. CLAUDE.md 是活文档:架构行为变化(如 scheduler、cancel 端点、新因子自动注册、预计算信号)完成后回写 CLAUDE.md 对应段落。

## 基线数据(供对比)

- 测试基线:`tests/unit` 252 项全绿(2026-08-14,pandas 3.0.5 / Python 3.13)。
- 已知不修的取舍:
  - 做空/期货/期权/分钟级:超出当前平台定位,不做;
  - 多用户/多租户:单机工具定位,E8 只做 token 级最小防线;
  - Pine 全量兼容:投入产出比低,仅修 C13 的两个 bug,`strategy.exit`/`var` 支持暂不做(UI 改明示支持子集)。

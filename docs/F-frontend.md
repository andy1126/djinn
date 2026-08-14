# 计划 F:前端改善

> 目标读者:执行模型。覆盖 `frontend/`(React 18 + TS strict + Vite 6 + antd 5 + TanStack Query 5 + Zustand 5 + ECharts)。
> 验证约定:每项完成后 `./node_modules/.bin/tsc -b --noEmit` 过 + `./node_modules/.bin/vite build` 过;涉及行为的给出手动验证步骤(页面→操作→预期)。前端当前无测试框架,本计划引入 vitest 做关键逻辑单测(见 F13)。

## 总览

| # | 改进点 | 类型 | 严重度 | 预估工作量 |
|---|---|---|---|---|
| F1 | WebSocket:卸载清理 + 断连降级 + 地址 env 化 | bug | P0 | 0.5 天 |
| F2 | ScreenerPage 结果表格化 + 导出 + 工作流动作 | 功能补完 | P0 | 1 天 |
| F3 | configStore persist + 跨页状态一致性 | bug | P0 | 0.5 天 |
| F4 | DashboardPage 条件轮询 | bug | P1 | 0.1 天 |
| F5 | SettingsPage 错误节流 + 实质化 | bug+功能 | P1 | 0.5 天 |
| F6 | queryFn 副作用消除(双份状态) | bug | P1 | 0.25 天 |
| F7 | 全局错误处理统一(axios 拦截器 + 列表 error 态) | 健壮性 | P1 | 0.5 天 |
| F8 | Bundle 优化(路由 lazy + manualChunks + echarts 按需) | 性能 | P1 | 0.5 天 |
| F9 | 共享组件抽取(paramWidget/任务历史列/格式化函数) | 代码质量 | P1 | 0.5 天 |
| F10 | ECharts notMerge + dayjs 中文 locale + 杂项修复 | bug | P1 | 0.25 天 |
| F11 | 搜索防抖 + abort(UniversePage) | 性能 | P2 | 0.25 天 |
| F12 | 类型安全收敛(78 处 any + eslint 配置补全) | 代码质量 | P2 | 1 天 |
| F13 | vitest 引入 + 关键逻辑单测 | 工程 | P2 | 0.5 天 |
| F14 | 分享链接/深链(URL 状态) | 功能 | P2 | 1 天 |
| F15 | 报告对比增强(命名/归一化/高亮) | UX | P2 | 0.5 天 |
| F16 | 全局通知中心 + 任务完成提醒 | UX | P2 | 0.5 天 |
| F17 | 回测停止按钮(依赖 E4 cancel 端点) | 功能 | P2 | 0.25 天 |
| F18 | 暗色模式 | UX | P3 | 1 天 |
| F19 | 响应式/移动端基础适配 | UX | P3 | 1 天 |
| F20 | 指数成分大列表虚拟化 | 性能 | P3 | 0.5 天 |

---

## F1. WebSocket:卸载清理 + 断连降级 + 地址 env 化

### 问题
`frontend/src/pages/BacktestRunPage.tsx:79-88` + `frontend/src/api/client.ts:257-278`:
1. `wsRef.current` 只在下次订阅前 close,**组件卸载时从不关闭**(无 useEffect cleanup)→ 回测中离开页面后 WS 继续推送,setState 打到已卸载组件;
2. `subscribeProgress` 支持 `onClose` 回调但调用方不传 → WS 断连后进度条永远停住,无任何降级;
3. `client.ts:265` WS 地址硬编码 `ws://localhost:8000`(注释自认"开发环境")——生产构建必坏;`client.ts:45` `baseURL: '/api'` 依赖 vite dev proxy,无 env 入口。

### 修改方案

**1. env 化(`client.ts`)**

```ts
const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'
const WS_BASE =
  import.meta.env.VITE_WS_BASE ??
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`
const http = axios.create({ baseURL: API_BASE, timeout: 60000 })

export const subscribeProgress = (jobId, onUpdate, onClose?) => {
  const url = `${WS_BASE}/api/backtests/${jobId}/progress`  // 与 vite proxy 前缀一致;E10 后改 /api/jobs/{id}/progress
  ...
}
```

新增 `frontend/.env.development`(可选,默认推导即可):`VITE_WS_BASE=ws://localhost:8000`。`vite.config.ts` 无需变(dev 下走推导:host=localhost:5173 → 但后端在 8000 —— 故 dev 环境**必须**提供 .env.development 或保留 localhost:8000 默认;决定:默认值 = `ws://<hostname>:8000`,即 `ws://${window.location.hostname}:8000`,dev/生产同域部署均可,跨域部署用 env 覆盖)。

**2. 自定义 hook 封装(新文件 `frontend/src/hooks/useJobProgress.ts`)**

```ts
export function useJobProgress(jobId: string | null) {
  const [job, setJob] = useState<JobStatus | null>(null)
  const [wsDead, setWsDead] = useState(false)
  useEffect(() => {
    if (!jobId) return
    setWsDead(false)
    const ws = subscribeProgress(jobId, setJob, () => setWsDead(true))  // onClose → 降级
    return () => ws.close()  // 卸载清理
  }, [jobId])
  // WS 断开 → TanStack Query 轮询降级(条件轮询模式,参照 SweepPage.tsx:71-74)
  const poll = useQuery({
    queryKey: ['job-poll', jobId],
    queryFn: () => getBacktest(jobId),
    enabled: !!jobId && wsDead,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'error' ? false : 2000,
  })
  return { job: wsDead ? (poll.data ?? job) : job, via: wsDead ? 'poll' : 'ws' }
}
```

**3. BacktestRunPage 改用 hook**,删除手写 wsRef 逻辑;进度区加来源指示(WS 断开时显示"实时连接已断开,已切换轮询"小字提示)。

### 验证
- 手动:启动回测 → 进度推送正常 → 中断后端进程 → 2s 内出现降级提示且进度继续以轮询更新 → 恢复后端;回测中切换路由离开再回来,无重复 WS 连接(DevTools Network WS 面板确认旧连接已关闭)。
- `tsc -b --noEmit` 过。

---

## F2. ScreenerPage 结果表格化 + 导出 + 工作流动作

### 问题
`ScreenerPage.tsx:257-259`:选股结果直接 `JSON.stringify` 塞 `<pre>` —— 核心功能出口是裸 JSON。后端 `run_screen_job`(`api/jobs.py:626-694`)返回的行结构已是结构化数据:`{symbol, ...基本面字段(动态列), score?}`(由 `_screen_row` 生成,字段集 = 基本面快照的列)。

### 修改方案(`ScreenerPage.tsx` 结果区重写)

1. **动态列表格**:

```tsx
const rows: Record<string, any>[] = job?.result?.results ?? []
const columns = useMemo(() => {
  if (!rows.length) return []
  const keys = Object.keys(rows[0]).filter(k => k !== 'symbol')
  return [
    { title: '代码', dataIndex: 'symbol', key: 'symbol', fixed: 'left' as const, width: 110 },
    ...keys.map(k => ({
      title: FIELD_LABELS[k] ?? k,  // 字段中文名映射(见下),未知字段原样显示
      dataIndex: k, key: k,
      render: (v: any) => (typeof v === 'number' ? formatCompact(v) : v ?? '—'),
      sorter: (a: any, b: any) => (a[k] ?? -Infinity) - (b[k] ?? -Infinity),
    })),
  ]
}, [rows])
<Table dataSource={rows} columns={columns} rowKey="symbol" size="small"
       pagination={{ pageSize: 50 }} scroll={{ x: true }} />
```

`FIELD_LABELS`:在 `types/index.ts` 或新 `utils/fieldLabels.ts` 建 `{pe:'PE',pb:'PB',roe:'ROE',market_cap:'总市值',score:'综合得分',...}`(与 `/screens/fields` 端点返回的字段清单对齐 —— 更优:该端点若已返回 label 直接用,没有则前端补映射表)。`formatCompact`:市值类 ≥1e8 显示 "x.xx 亿",≥1e4 "x.xx 万"(注意 A 股用户习惯)。

2. **动作条**(结果区顶部):
   - **导出 CSV**:前端生成 Blob 下载(`new Blob([toCsv(rows)])`,不依赖后端;toCsv 工具函数);
   - **加入 Profile**:`Modal` 选择已有 Profile 或新建 → 调 `updateProfile`/`createProfile`(`client.ts:241-254` 已有)把 symbols 写入;成功后 message 提示;
   - **用这组股票发起回测**:`useConfigStore.getState().updateConfig('universe', {...cfg.universe, symbols: rows.map(r=>r.symbol), index: undefined})` → `navigate('/backtest')`(configStore 已有,F3 persist 后跨页保留)。
3. 空态保留现有优秀 Alert(:247-252)不动;加载态接现有任务轮询。

### 验证
- 手动:跑一次美股筛选 → 表格渲染、列可排序、导出 CSV 打开正确;加入 Profile 后 UniversePage 的 Profile 管理可见;发起回测跳转后 BacktestRunPage 的标的已填。
- vitest:`toCsv`/`formatCompact` 单测。

---

## F3. configStore persist + 跨页状态一致性

### 问题
1. `store/configStore.ts:35`:无 persist —— 刷新页面全部配置丢回 NVDA 默认;
2. `PortfolioConfigPage.tsx:12-14`:useState 初始化器从 config 拷贝 symbols → 其他页改 config 后回本页显示**过期副本**;`apply()` 在 `allocation !== 'custom'` 时**静默丢弃**手调权重(:32-34);
3. `BacktestRunPage.tsx:44-53`:mount-only `form.setFieldsValue` 同理,跨页变更不同步。

### 修改方案
1. **persist**:

```ts
import { persist, createJSONStorage } from 'zustand/middleware'
export const useConfigStore = create<ConfigStore>()(
  persist(
    (set) => ({ /* 现状不变 */ }),
    { name: 'djinn-config', storage: createJSONStorage(() => localStorage), version: 1 },
  ),
)
```

2. **PortfolioConfigPage 去本地副本**:symbols/weights 直接读写 store(`useConfigStore(s => s.config)`),行编辑先改本地 draft state 但 **draft 从 store 派生并用 `useEffect(() => setDraft(storeSymbols), [storeSymbols])` 同步**;`apply()` 在非 custom 时若用户手调过权重 → `Modal.confirm`("当前分配方式为等权/市值,手调权重将不生效,是否切换为自定义权重?")而不是静默丢弃。
3. **BacktestRunPage 同步**:`form.setFieldsValue` 放 `useEffect(..., [config])`(依赖 store 配置),store 变化时表单跟随;提交时 `syncConfig` 逻辑不变。
4. 加"重置为默认"按钮(调 store.reset(),confirm 后执行)。

### 验证
- 手动:改策略参数 → 刷新页面 → 参数保留;Portfolio 页改资金 → 进回测页表单一致;非 custom 下手调权重弹确认。
- vitest:store persist 序列化/反序列化 round-trip(localStorage mock)。

---

## F4. DashboardPage 条件轮询

### 问题
`DashboardPage.tsx:43-47` `refetchInterval: 3000` 常量 —— 永久轮询(其他四页都做了条件轮询)。

### 修改方案
改为函数形(参照 `SweepPage.tsx:71-74` 现成模式):

```ts
refetchInterval: (q) =>
  (q.state.data ?? []).some(j => j.status === 'pending' || j.status === 'running') ? 3000 : false,
```

### 验证
- 手动:无进行中任务时 DevTools Network 无周期请求;启动回测后恢复 3s 轮询,完成后停止。

---

## F5. SettingsPage 错误节流 + 实质化

### 问题
`SettingsPage.tsx:7-17`:健康检查 5s 轮询 + `useEffect([error])` 里 `message.error` —— error 对象引用每次变,**每 5 秒弹一条**;且该页除红绿灯外无任何设置项,名不副实。

### 修改方案
1. 错误提示节流:用 `useRef<string>` 记录已提示的错误消息,相同消息不重复弹;或干脆删除 message 弹窗 —— 页内红灯+错误文本已足够(推荐:删弹窗,页面内联 Alert 展示 `error.message`)。
2. 实质化(与 E6/E8 联动):
   - **后端健康**区(现状保留);
   - **API Token** 输入(存 localStorage,axios 拦截器读取,见 F7);
   - **外观**:明/暗切换(F18);
   - **数据维护**:显示缓存大小(调 `/data/cache` 求和)、"清理历史任务"按钮(调 `POST /jobs/purge`,E6)、缓存清理按钮(已有 `DELETE /data/cache`,加 confirm)。

### 验证
- 手动:断网 → 页面红灯 + 单条内联错误,无刷屏;填 token 后请求头带 Authorization(F7 联动)。

---

## F6. queryFn 副作用消除

### 问题
`FactorAnalysisPage.tsx:143-151`、`FactorMatrixPage.tsx:145-153`:在 `queryFn` 里 `setReport(r)` —— queryFn 必须无副作用(StrictMode 双调用重复 setState);数据同时存 Query 缓存和 useState 双份。

### 修改方案
删除本地 `report` state 与 queryFn 内 setReport,组件直接消费:

```tsx
const { data: report, isFetching } = useQuery({ queryKey: ['factor-report', jobId], queryFn: () => getFactorAnalysisReport(jobId!), enabled: job?.status === 'done' })
```

渲染处全部改 `report?.xxx`(两处页面的图表/表格 props)。

### 验证
- 手动:任务完成后报告正常渲染;React DevTools 无多余重渲染告警;StrictMode 下无双倍请求(Query 缓存去重)。

---

## F7. 全局错误处理统一

### 问题
- 列表查询(strategies/factors/indexes/profiles/cache/indicators)失败时**静默渲染空表** —— 用户分不清"无数据"与"后端挂了";全站仅 UniversePage:411 与 ReportDetail:51-53 有 isError 分支;
- axios 无全局拦截器;
- `ReportCompare.tsx:22-36` 单报告失败被 `catch → null` 静默吞掉,失败列整列空白。

### 修改方案
1. **axios 拦截器**(`client.ts`):

```ts
http.interceptors.response.use(undefined, (err) => {
  // 统一在错误对象上挂友好消息;不全局弹窗(列表页内联展示,见 2)
  err.friendlyMessage = err?.response?.data?.detail ?? err.message ?? '请求失败'
  return Promise.reject(err)
})
```

2. **共享组件 `QueryErrorAlert`**(新 `components/QueryErrorAlert.tsx`):`{error, retry?}` → antd Alert(error 文案 + 重试按钮)。在上述 6+ 个列表查询的渲染处统一:`if (isError) return <QueryErrorAlert error={error} retry={refetch} />`(表格上方或替换表格)。
3. `ReportCompare`:失败列显示 ErrorTag("加载失败:原因")而非空白;`Promise.allSettled` 替换 catch-null(:22-36),列渲染处按 settled 状态分支。

### 验证
- 手动:停后端 → 各列表页显示内联错误+重试(非空白表);恢复后重试成功;对比页单报告失败显示失败标记。

---

## F8. Bundle 优化

### 问题
`vite.config.ts` 零构建配置:echarts 全量 + CodeMirror + antd 全进首屏单 chunk(gzip 估 1MB+);无路由 lazy。

### 修改方案
1. **路由级 lazy**(`router.tsx`):全部页面 `const HomePage = lazy(() => import('./pages/HomePage'))`,children 的 element 包 `<Suspense fallback={<Spin/>}>`(LayoutShell 的 Outlet 处统一包一次即可)。
2. **manualChunks**(`vite.config.ts`):

```ts
build: {
  rollupOptions: {
    output: {
      manualChunks: {
        echarts: ['echarts', 'echarts-for-react'],
        editor: ['@uiw/react-codemirror', '@codemirror/lang-python'],
        antd: ['antd', '@ant-design/icons'],
        vendor: ['react', 'react-dom', 'react-router-dom', 'axios', 'zustand', '@tanstack/react-query', 'dayjs'],
      },
    },
  },
}
```

3. **echarts 按需**(可选进阶):新 `components/charts/echartsCore.ts` 用 `echarts/core` + 按需注册 LineChart/BarChart/PieChart/HeatmapChart/GridComponent/TooltipComponent/LegendComponent/DataZoomComponent + CanvasRenderer;各图表组件改 `import ReactEChartsCore from 'echarts-for-react/lib/core'`。收益约 300KB gzip,工作量集中在逐一核对图表用到的组件类型 —— 若时间紧,1+2 已够,本步列为可选。
4. **CodeMirror 页隔离**:IndicatorLibraryPage/StrategyPage 已随路由 lazy 自动隔离,确认 editor chunk 不进首屏。

### 验证
- `vite build` 后 `dist/assets` 列表确认分 chunk;首屏 JS(gzip)对比:目标 < 500KB(原 ~1MB+);手动冒烟全部 14 页路由懒加载正常(每页首次进入有 loading 态)。

---

## F9. 共享组件抽取

### 问题
- `paramWidget` 在 `FactorAnalysisPage.tsx:30-74` 与 `FactorMatrixPage.tsx:33-77` **逐字重复 45 行**;与 `StrategyParamForm.tsx` 职责三份重叠;
- 历史任务表格列(任务/状态/进度/阶段/错误)在 SweepPage:167-182、FactorAnalysisPage:300-321、FactorMatrixPage:376-397、ScreenerPage:265-285 重复 4 份;
- `fmtPct/fmtNum` 在 MetricsCards:9-10 与 ReportCompare:16-17 重复。

### 修改方案
1. 新 `components/ParamFields.tsx`:输入 `(schema: ParamSchema[], value, onChange)`,统一渲染 int/float/str/bool/select 控件;FactorAnalysisPage/FactorMatrixPage 删本地 paramWidget 改引用;`StrategyParamForm` 保持其 Form 集成形态但内部控件层与 ParamFields 共用子组件(或文档注明分工:ParamFields 用于非 antd-Form 场景)。
2. 新 `components/JobHistoryTable.tsx`:props `{jobs, kind, onOpen(jobId), extraColumns?}`;内置 任务标题/状态 Tag/进度条/阶段/错误/时间 列;四页替换,页面只传 extraColumns(如 Sweep 的"最优值"列)。
3. 新 `utils/format.ts`:`fmtPct/fmtNum/formatCompact` 一处定义,MetricsCards/ReportCompare/其他页面改 import。

### 验证
- `tsc` 过;四页手动冒烟(任务列表渲染/排序/点击行为不变);`grep -c "paramWidget" frontend/src/pages/` 输出 0。

---

## F10. ECharts notMerge + dayjs 中文 locale + 杂项修复

### 修改方案(均为小点,一次做完)
1. **notMerge**:`components/charts/` 全部图表的 `<ReactECharts option={...}>` 加 `notMerge`(换报告/换标的时旧 series 不残留);`PositionAreaChart`/`FactorDistChart` 是重灾区,逐一检查。
2. **dayjs locale**:`main.tsx` 顶部加 `import 'dayjs/locale/zh-cn'` + `dayjs.locale('zh-cn')`(修 antd zhCN 下 DatePicker 月份/星期英文)。
3. **模块级可变计数器**:`SweepPage.tsx:34`、`FactorMatrixPage.tsx:79` 的 `let _uid = 0` 改为组件内 `useRef(0)` 或 `crypto.randomUUID()`(HMR 下与 state 冲突)。
4. **死代码**:`DataManagerPage.tsx:25` 未使用的 `symbols/setSymbols` 删除。
5. **index 作为 key**:`UniversePage.tsx:420`、`PortfolioConfigPage.tsx:47`、`TradesTable.tsx:30` 改稳定 key(symbol/trade id)。
6. **ReportDetail 导出提示**(:29):"已导出到 {服务器路径}" 对 Web 用户无意义 → 改提示"CSV 已生成",或配合后端 E 计划改文件下载后走 Blob 保存。
7. **DashboardPage 80ms setTimeout 滚动**(:29-32):报告异步拉取,定时不可靠 —— 改为 ReportDetail 挂载后 `useEffect` 内 `scrollIntoView`(经 ref 回调或 MutationObserver;简化:ReportDetail 组件 `useEffect(() => ref.current?.scrollIntoView({behavior:'smooth'}), [jobId])`)。

### 验证
- 手动逐项:切换两份报告图表无残影;DatePicker 显示中文月份;HMR 改动 SweepPage 后行 key 不重复;导出提示合理;点"查看结果"滚动到位。

---

## F11. 搜索防抖 + abort(UniversePage)

### 问题
`UniversePage.tsx:266-270`:`queryKey` 含 `query`,每个字符一次请求,无 abort → 快速输入并发乱序。

### 修改方案
输入值 `const [raw, setRaw] = useState('')`,防抖值 `const q = useDebouncedValue(raw, 300)`(新 `hooks/useDebouncedValue.ts`:setTimeout+cleanup 10 行);`queryKey: ['stock-search', q]`,`enabled: q.length >= 1`;TanStack Query 自动 abort 旧请求(v5 的 queryFn 接收 `signal`,axios get 传 `signal` —— client.ts 的 searchStocks 加可选 `signal?: AbortSignal` 参数透传)。

### 验证
- 手动:快速输入 "nvda" → Network 面板仅 1~2 次请求且先前请求被 cancel;结果与最终输入对应。

---

## F12. 类型安全收敛 + eslint 配置

### 问题
- 全项目 78 处 `any`(典型:`FactorAnalysisPage.tsx:93` useState\<any|null>(已有 FactorReport 类型)、各页 `poll.data as any`、`SweepPage.tsx:136-164` 整段 `(r as any)`、`BacktestRunPage.tsx:55,72` `syncConfig(v: any)`);
- `tsconfig.json:15-16` 关闭 `noUnusedLocals/noUnusedParameters`;
- `package.json:10` 有 lint 脚本但**仓库无 eslint 配置文件**(lint 命令直接失败)。

### 修改方案
1. **补 eslint 配置**:`eslint.config.js`(flat config)用 `typescript-eslint`  recommended + react-hooks 插件;`npm i -D eslint @eslint/js typescript-eslint eslint-plugin-react-hooks eslint-plugin-react-refresh`;lint 脚本改 `eslint src`。规则从严但 `no-explicit-any` 先设 warn。
2. **types/index.ts 补全**:`SweepResultRow`(字段与 jobs.py:139-151 的结果 dict 对齐:config_summary/sharpe/sortino/calmar/total_return/max_drawdown/n_trades 等)、`ScreenResultRow`(symbol + 动态字段用 `Record<string, unknown>` + symbol/score 已知键)、`FactorReport`(若已有则用之)。逐文件替换 any:
   - `SweepPage.tsx:136-164` 列定义 `ColumnsType<SweepResultRow>`;
   - `poll.data as any` → `JobStatus`(client.ts 已有类型)直接消费;
   - `FactorAnalysisPage.tsx:93` → `useState<FactorReport | null>`(F6 重构后此 state 删除,顺势解决)。
3. `tsconfig` 打开 `noUnusedLocals/noUnusedParameters`,逐一修复报错(预期 10-20 处,多为未用 import)。

### 验证
- `npm run lint` 可运行且 0 error(warn 列表输出);`tsc -b --noEmit` 在 strict 新开关下过;`grep -c "as any\|: any" frontend/src/**/*.tsx` 显著下降(目标 < 20,残留逐个注明原因)。

---

## F13. vitest 引入 + 关键逻辑单测

### 修改方案
1. `npm i -D vitest @testing-library/react @testing-library/user-event jsdom`;`vite.config.ts` 加 `test: { environment: 'jsdom', globals: true }`(或独立 `vitest.config.ts`);package.json 加 `"test": "vitest run"`。
2. 首批测试(纯逻辑优先,不碰 echarts):
   - `utils/format.ts` 的 fmtPct/fmtNum/formatCompact;
   - F2 的 toCsv;
   - `hooks/useDebouncedValue`(fake timers);
   - `store/configStore` 的 updateConfig/reset/persist round-trip;
   - F1 的 WS URL 推导逻辑(抽纯函数 `wsUrl(jobId)` 后单测)。

### 验证
- `npm test` 绿;CI/命令文档(CLAUDE.md 前端命令段)补 `npm test`。

---

## F14. 分享链接/深链(URL 状态)

### 问题
对比页勾选、因子分析/矩阵/选股任务均无深链(仅回测有 `/results/:jobId`)—— 无法把视图分享给他人(或自己跨设备)。

### 修改方案(react-router `useSearchParams`)
- `/results?compare=<jobId1>,<jobId2>`:DashboardPage 勾选状态 ↔ URL 双向同步(初始读、变更写 `setSearchParams(..., {replace:true})`);
- `/factors?job=<jobId>`、`/factor-matrix?job=<jobId>`、`/screener?job=<jobId>`:各页从 URL 恢复当前查看的任务(有 job 参数直接加载报告);
- 各报告区加"复制链接"按钮(`navigator.clipboard.writeText(location.href)` + message 成功)。

### 验证
- 手动:勾选两个报告 → URL 出现 compare → 复制到新标签页 → 对比视图原样恢复;三个 alpha 页同验。

---

## F15. 报告对比增强

### 问题
`ReportCompare.tsx`:legend 用裸 jobId(不可读);无净值起点归一化;无最优高亮/差异列。

### 修改方案
1. **可读名**:列标题用 `title` 元数据(JobStatus.to_dict 已含 title,jobs.py:78)+ jobId 短码 tooltip。
2. **归一化**:净值曲线图各序列 `v / v[0]` 起点对齐(图表层处理,不动数据)。
3. **指标表**:每指标行标"最优者"底色(sharpe/sortino/calmar 越大越好,max_drawdown 越接近 0 越好 —— 方向表与后端 `REVERSE_MIN_TARGETS` 同步硬编码于 `utils/metricDirections.ts`);增加"差异"行(各列 vs 首列的差值,可选 Switch 开启)。

### 验证
- 手动:两份报告对比,legend 显示策略名+参数;起点对齐;最优格高亮正确(抽查 3 个指标方向)。

---

## F16. 全局通知中心 + 任务完成提醒

### 修改方案
1. 新 `store/notifyStore.ts`(zustand):`{items: {id, kind, title, status, ts}[], push, markRead}`。
2. `LayoutShell.tsx` 头部加 Bell 图标 + Badge(未读数) + Dropdown 列表(点击跳对应页,配合 F14 深链)。
3. 任务完成感知:各任务列表页的条件轮询(F4 模式)检测 `running → done/error` 迁移时 push 通知 + `message.success`;跨页也能收到(轮询在各页,通知入全局 store)。
4. 可选:浏览器 Notification API 授权后发系统通知(回测完成时标签页不可见场景)。

### 验证
- 手动:跑回测后切到别的页 → 完成时铃铛 Badge +1 → 点击跳转 `/results/:jobId`。

---

## F17. 回测停止按钮(依赖 E4)

### 修改方案
- `client.ts` 加 `cancelBacktest = (jobId) => http.post(\`/backtests/\${jobId}/cancel\`)`(或 E4 统一端点 `/jobs/{id}/cancel`);
- DashboardPage 行操作列:running 时显示"取消"(Popconfirm);BacktestRunPage 进度区加"停止任务"按钮(F1 的 hook 返回 cancel 方法);
- 取消成功后列表刷新,状态 Tag 显示"已取消"(E4 的 cancelled 状态;types/index.ts 的 JobStatus.status 联合类型加 `'cancelled'`)。

### 验证
- 手动:长回测跑到 30% 取消 → 状态变 cancelled,无报告产出;列表 Tag 正确。

---

## F18. 暗色模式

### 修改方案
1. `store/uiStore.ts`(新):`{dark: boolean, toggle}`,persist 到 localStorage。
2. `LayoutShell.tsx` 的 ConfigProvider(:112):`theme={{ algorithm: dark ? theme.darkAlgorithm : theme.defaultAlgorithm, token: {...现状} }}`;Menu/Layout 背景由 algorithm 自动处理,自定义样式(若有硬编码颜色)改用 token。
3. ECharts:各图表 option 的 `backgroundColor: 'transparent'`,文本/轴颜色读 antd token(`theme.useToken()` 传给图表或 CSS 变量);最省做法:图表包一层 `data-theme` 容器,option 里用 `color` 调色板自适应 + textStyle 用 token.colorText。
4. CodeMirror:`theme={dark ? 'dark' : 'light'}`(@uiw/react-codemirror 支持字符串主题)。
5. SettingsPage(F5)加切换开关。

### 验证
- 手动:切换后全站 14 页无刺眼的白色块;图表文字可读;刷新保持。

---

## F19. 响应式/移动端基础适配

### 修改方案(达到"平板可用、手机可读"即可,不追求完整移动体验)
1. `LayoutShell.tsx:114`:Sider 加 `breakpoint="lg" collapsedWidth="0"`(窄屏自动折叠成抽屉);
2. 栅格断点:`Col span={12}` → `xs={24} md={12}`;`span={8}` → `xs={24} sm={12} lg={8}`;`span={6}` → `xs={12} sm={12} lg={6}` —— 逐页过(BacktestRunPage:96、StrategyPage:174、MetricsCards:15-38、UniversePage:168-187、FactorAnalysisPage/FactorMatrixPage/ScreenerPage 的表单栅格);
3. 表格:全部 `scroll={{ x: true }}`(窄屏横向滚动而非挤压);
4. 首页管线卡片(HomePage:130 的 Card onClick)补 `role="button" tabIndex={0} onKeyDown(Enter)`(顺手修 a11y)。

### 验证
- 手动:Chrome DevTools 切 iPad/iPhone 尺寸,14 页无内容溢出、表格可横滚、菜单折叠可用。

---

## F20. 指数成分大列表虚拟化

### 问题
`UniversePage.tsx:418-427`:指数成分(CSI800 → 800 个 Col)全量渲染。

### 修改方案
该区域(成分股的 Col 网格)改为 antd `List` + `virtual`(antd v5 支持 `list` 的虚拟滚动)或 `react-window` 的 FixedSizeGrid(若不想加依赖:分页 100/页已足够 —— **推荐最简:改分页卡片网格**,`Pagination` 控制 slice,不引入新依赖)。

### 验证
- 手动:CSI800 成分页首屏渲染时间明显改善(React DevTools Profiler 对比);翻页正常。

---

## 验收清单

1. `npm run lint` 0 error、`tsc -b --noEmit` 过、`npm test` 绿、`vite build` 分 chunk 成功且首屏 < 500KB gzip。
2. 手动冒烟清单(每页一条):14 页全部可达;F1/F2/F3/F4/F7/F14/F16/F17 的手动验证步骤逐项通过。
3. 与后端联动:E4(cancel)/E6(purge)/E8(token)/E10(WS 通用端点)就绪后,F5/F10.6/F16/F17 联调。

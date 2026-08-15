import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Button, Card, Checkbox, Form, Input, message, Progress, Select, Segmented, Space, Table, Tag, Typography,
} from 'antd'
import { createSweep, errDetail, listSweeps } from '@/api/client'
import { useConfigStore } from '@/store/configStore'
import JobHistoryTable from '@/components/JobHistoryTable'
import type { JobStatus, SweepResultRow } from '@/types'

/**
 * sweep 可扫轴(与后端 cli/sweep.py:ALLOWED_SWEEP_AXES 同步)。
 * 裸策略参数(无前缀)也可扫——下拉"策略参数"对应文本框手填 key。
 */
const AXIS_OPTIONS = [
  { value: '__param__', label: '策略参数(自定义 key)' },
  { value: 'universe.index', label: 'universe.index 成分池' },
  { value: 'strategy.factor_weights', label: 'strategy.factor_weights 因子组合' },
  { value: 'portfolio.allocation', label: 'portfolio.allocation 权重法' },
  { value: 'strategy.n_stocks', label: 'strategy.n_stocks 选股数' },
  { value: 'strategy.rebalance_freq', label: 'strategy.rebalance_freq 调仓频率' },
]

const ALLOCATION_OPTIONS = ['equal', 'market_cap', 'custom', 'score', 'risk_parity', 'min_variance', 'mean_variance']
const INDEX_OPTIONS = ['CSI300', 'CSI500', 'CSI800', 'HSI', 'SP500', 'NASDAQ100', 'DOWJONES']

interface AxisDraft {
  uid: number
  axis: string
  /** 自定义策略参数名(仅 axis=__param__ 时用)。 */
  paramKey?: string
  valuesRaw: string
}

// F10:随机种子避免 HMR 重载后计数器归零与既有 state 的 uid 冲突
let _uid = Math.floor(Math.random() * 1_000_000)

/** 把单行草案解析成 grid entry。 */
function draftToGridEntry(d: AxisDraft): [string, (number | string)[]] | null {
  const vals = d.valuesRaw
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      const n = Number(s)
      return Number.isNaN(n) ? s : n
    })
  if (!vals.length) return null
  let key: string
  if (d.axis === '__param__') {
    if (!d.paramKey?.trim()) return null
    key = d.paramKey.trim()
  } else {
    key = d.axis
  }
  return [key, vals]
}

export default function SweepPage() {
  const qc = useQueryClient()
  const { config } = useConfigStore()
  const [form] = Form.useForm()
  const [jobId, setJobId] = useState<string | null>(null)
  const [mode, setMode] = useState<'graphic' | 'text'>('graphic')
  const [drafts, setDrafts] = useState<AxisDraft[]>([
    { uid: ++_uid, axis: '__param__', paramKey: 'fast', valuesRaw: '5,10,20' },
    { uid: ++_uid, axis: '__param__', paramKey: 'slow', valuesRaw: '20,30,50' },
  ])

  const { data: jobs } = useQuery({
    queryKey: ['sweeps'],
    queryFn: () => listSweeps(20),
    refetchInterval: (query) => {
      const data = query.state.data as JobStatus[] | undefined
      return data?.some((j) => j.status === 'pending' || j.status === 'running') ? 3000 : false
    },
  })

  const sweepMut = useMutation({
    mutationFn: createSweep,
    onSuccess: (resp) => {
      setJobId(resp.job_id)
      message.success(`扫描任务已创建: ${resp.job_id}`)
      qc.invalidateQueries({ queryKey: ['sweeps'] })
    },
    onError: (e) => message.error(errDetail(e)),
  })

  const addDraft = () =>
    setDrafts((p) => [...p, { uid: ++_uid, axis: '__param__', paramKey: '', valuesRaw: '' }])
  const removeDraft = (uid: number) => setDrafts((p) => p.filter((d) => d.uid !== uid))
  const update = (uid: number, patch: Partial<AxisDraft>) =>
    setDrafts((p) => p.map((d) => (d.uid === uid ? { ...d, ...patch } : d)))

  const buildGrid = (gridText: string | undefined): Record<string, (number | string)[]> => {
    if (mode === 'text') {
      const grid: Record<string, (number | string)[]> = {}
      ;(gridText || '').split('\n').forEach((line) => {
        const [name, vals] = line.split(':')
        if (name && vals) {
          grid[name.trim()] = vals.split(',').map((s) => {
            const n = Number(s.trim())
            return Number.isNaN(n) ? s.trim() : n
          })
        }
      })
      return grid
    }
    const grid: Record<string, (number | string)[]> = {}
    for (const d of drafts) {
      const entry = draftToGridEntry(d)
      if (entry) grid[entry[0]] = entry[1]
    }
    return grid
  }

  const onSubmit = (v: { gridText?: string; target: string; parallel: boolean }) => {
    const grid = buildGrid(v.gridText ?? '')
    if (Object.keys(grid).length === 0) {
      message.error('请至少配置一个扫轴')
      return
    }
    sweepMut.mutate({ config, grid, target: v.target, parallel: v.parallel })
  }

  const running = jobs?.find((j) => j.job_id === jobId)

  // 结果行(完成任务 result.results)
  const resultRows: SweepResultRow[] = (() => {
    const r = running
    if (!r || r.status !== 'done') return []
    const results = (r.result as { results?: unknown } | undefined)?.results
    return Array.isArray(results) ? (results as SweepResultRow[]) : []
  })()
  const target = ((running?.result as { target?: string } | undefined)?.target) || 'sharpe'

  const summaryCols = [
    { title: '组合', key: 'params', render: (_: unknown, r: SweepResultRow) => (
      <Space direction="vertical" size={0}>
        {Object.entries(r.params).map(([k, v]) => (
          <Typography.Text key={k} code>{k}={String(v)}</Typography.Text>
        ))}
      </Space>
    )},
    { title: '权重法', key: 'alloc', render: (_: unknown, r: SweepResultRow) =>
      r.config_summary['portfolio.allocation'] },
    { title: 'n_stocks', key: 'nstocks', render: (_: unknown, r: SweepResultRow) =>
      r.config_summary['strategy.n_stocks'] ?? '—' },
    { title: 'index', key: 'idx', render: (_: unknown, r: SweepResultRow) =>
      r.config_summary['universe.index'] ?? '—' },
    { title: '标的数', key: 'nsym', render: (_: unknown, r: SweepResultRow) =>
      r.config_summary.n_symbols },
    { title: 'target', key: 'target', render: (_: unknown, r: SweepResultRow) =>
      <Tag color="blue">{Number(r[target] ?? 0).toFixed(3)}</Tag> },
    { title: 'sharpe', key: 'sharpe', render: (_: unknown, r: SweepResultRow) =>
      Number(r.sharpe ?? 0).toFixed(3) },
    { title: 'sortino', key: 'sortino', render: (_: unknown, r: SweepResultRow) =>
      Number(r.sortino ?? 0).toFixed(3) },
    { title: 'calmar', key: 'calmar', render: (_: unknown, r: SweepResultRow) =>
      Number(r.calmar ?? 0).toFixed(3) },
    { title: '总收益', key: 'ret', render: (_: unknown, r: SweepResultRow) =>
      `${(Number(r.total_return ?? 0) * 100).toFixed(2)}%` },
    { title: '最大回撤', key: 'mdd', render: (_: unknown, r: SweepResultRow) =>
      `${(Number(r.max_drawdown ?? 0) * 100).toFixed(2)}%` },
    { title: '交易数', key: 'trades', render: (_: unknown, r: SweepResultRow) =>
      r.n_trades ?? '—' },
  ]

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>参数扫描</Typography.Title>

      <Card title="扫描配置">
        <Segmented
          value={mode}
          onChange={(m) => setMode(m as 'graphic' | 'text')}
          options={[
            { label: '图形化', value: 'graphic' },
            { label: '文本', value: 'text' },
          ]}
          style={{ marginBottom: 16 }}
        />
        <Form
          form={form}
          layout="vertical"
          onFinish={onSubmit}
          initialValues={{ target: 'sharpe', parallel: false, gridText: 'fast:5,10,20\nslow:20,30,50' }}
        >
          {mode === 'graphic' ? (
            <Card size="small" title="扫轴(每行一个轴,值用逗号分隔)">
              <Space direction="vertical" style={{ width: '100%' }} size="small">
                {drafts.map((d) => (
                  <Space key={d.uid} wrap>
                    <Select
                      value={d.axis}
                      onChange={(v) => update(d.uid, { axis: v as string })}
                      options={AXIS_OPTIONS}
                      style={{ width: 280 }}
                    />
                    {d.axis === '__param__' && (
                      <Input
                        placeholder="参数名(如 fast)"
                        value={d.paramKey}
                        onChange={(e) => update(d.uid, { paramKey: e.target.value })}
                        style={{ width: 140 }}
                      />
                    )}
                    {d.axis === 'portfolio.allocation' ? (
                      <Select
                        mode="multiple"
                        placeholder="选权重法"
                        value={d.valuesRaw ? d.valuesRaw.split(',').map((s) => s.trim()) : []}
                        onChange={(vs) => update(d.uid, { valuesRaw: vs.join(',') })}
                        options={ALLOCATION_OPTIONS.map((a) => ({ value: a, label: a }))}
                        style={{ minWidth: 240 }}
                      />
                    ) : d.axis === 'universe.index' ? (
                      <Select
                        mode="multiple"
                        placeholder="选指数"
                        value={d.valuesRaw ? d.valuesRaw.split(',').map((s) => s.trim()) : []}
                        onChange={(vs) => update(d.uid, { valuesRaw: vs.join(',') })}
                        options={INDEX_OPTIONS.map((a) => ({ value: a, label: a }))}
                        style={{ minWidth: 220 }}
                      />
                    ) : (
                      <Input.TextArea
                        rows={1}
                        placeholder="值:5,10,20"
                        value={d.valuesRaw}
                        onChange={(e) => update(d.uid, { valuesRaw: e.target.value })}
                        style={{ minWidth: 260 }}
                      />
                    )}
                    <Button danger onClick={() => removeDraft(d.uid)}>删除</Button>
                  </Space>
                ))}
                <Button type="dashed" onClick={addDraft}>+ 添加扫轴</Button>
              </Space>
            </Card>
          ) : (
            <Form.Item name="gridText" label="参数网格(每行:name:value1,value2)" rules={[{ required: true }]}>
              <Input.TextArea rows={4} placeholder="fast:5,10,20&#10;slow:20,30,50&#10;portfolio.allocation:equal,score" />
            </Form.Item>
          )}
          <Form.Item name="target" label="优化目标" style={{ marginTop: 16 }}>
            <Select
              style={{ width: 220 }}
              options={[
                'sharpe', 'sortino', 'calmar', 'total_return', 'annual_return',
                'max_drawdown', 'volatility', 'annual_volatility', 'n_trades',
              ].map((t) => ({ value: t, label: t }))}
            />
          </Form.Item>
          <Form.Item name="parallel" valuePropName="checked" label="并行执行">
            <Checkbox>joblib 多进程</Checkbox>
          </Form.Item>
          <Form.Item label="基础配置">
            <span>策略: {config.strategy.name} · 标的: {config.universe.symbols.join(',')}</span>
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={sweepMut.isPending}>开始扫描</Button>
          </Form.Item>
        </Form>
      </Card>

      {jobId && running && (
        <Card title={`当前扫描 ${running.title || running.job_id}`}>
          <Progress percent={Math.round(running.progress * 100)} status={running.status === 'done' ? 'success' : running.status === 'error' ? 'exception' : 'active'} />
          <div>阶段: {running.stage}</div>
          {running.error && <Typography.Text type="danger">{running.error}</Typography.Text>}
        </Card>
      )}

      {resultRows.length > 0 && (
        <Card title={`扫描结果(按 ${target} ${['volatility', 'annual_volatility'].includes(target) ? '升序' : '降序'} 排序)`}>
          <Table
            size="small"
            dataSource={resultRows}
            rowKey={(r) => JSON.stringify(r.params)}
            columns={summaryCols}
            pagination={{ pageSize: 20 }}
            scroll={{ x: 'max-content' }}
          />
        </Card>
      )}

      <Card title="历史扫描任务">
        <JobHistoryTable
          jobs={(jobs || []) as JobStatus[]}
          onOpen={(id) => setJobId(id)}
        />
      </Card>
    </Space>
  )
}
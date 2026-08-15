import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Alert, Button, Card, DatePicker, Form, Input, InputNumber, message, Progress, Select, Space, Table, Tag, Typography,
} from 'antd'
import {
  listFactors,
  listIndexes,
  listScreenFields,
  listScreenMarkets,
  listScreenJobs,
  createScreen,
  getScreenJob,
  errDetail,
} from '@/api/client'
import { useConfigStore } from '@/store/configStore'
import JobHistoryTable from '@/components/JobHistoryTable'
import type { FactorInfo, IndexInfo, JobStatus, ScreenField, ScreenMarket, ScreenOp, ScreenResultRow } from '@/types'

const OPS = ['gt', 'lt', 'ge', 'le', 'eq', 'between', 'in']

// 数值格式化:市值类 ≥1e8 显示"x.xx 亿",≥1e4 "x.xx 万"
function formatCompact(v: number): string {
  if (v == null || Number.isNaN(v)) return '—'
  if (Math.abs(v) >= 1e8) return `${(v / 1e8).toFixed(2)} 亿`
  if (Math.abs(v) >= 1e4) return `${(v / 1e4).toFixed(2)} 万`
  if (Number.isInteger(v)) return String(v)
  return v.toFixed(4)
}

// 结果行 → CSV 字符串(前端 Blob 下载)
function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return ''
  const keys = Object.keys(rows[0])
  const esc = (v: unknown): string => {
    const s = v == null ? '' : String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const header = keys.map(esc).join(',')
  const body = rows.map((r) => keys.map((k) => esc(r[k])).join(',')).join('\n')
  return `${header}\n${body}`
}

/**
 * 选股页:条件过滤 + 可选多因子打分排序 → 后台任务 → 股票列表 + 得分。
 * 市场由所选宽基指数推断(不再单独选);历史任务可回看(刷新不丢失)。
 */
export default function ScreenerPage() {
  const qc = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()
  const [jobId, setJobId] = useState<string | null>(null)
  const [conditions, setConditions] = useState<Array<{ field: string; op: string; value: string }>>([
    { field: 'pe', op: 'lt', value: '30' },
  ])
  const [scores, setScores] = useState<Array<{ factor: string; weight: number; direction: 1 | -1 }>>([])
  const [topN, setTopN] = useState<number | null>(10)

  // F14:深链 —— 从 URL ?job=<id> 恢复查看的任务;切换任务时写回 URL
  useEffect(() => {
    const j = searchParams.get('job')
    if (j) setJobId(j)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const selectJob = (id: string) => {
    setJobId(id)
    setSearchParams({ job: id }, { replace: true })
  }

  const { data: factorsResp } = useQuery({ queryKey: ['factors'], queryFn: listFactors })
  const factors: FactorInfo[] = factorsResp?.factors || []
  const { data: indexesResp } = useQuery({ queryKey: ['indexes'], queryFn: listIndexes })
  const indexes: IndexInfo[] = useMemo(() => indexesResp?.indexes || [], [indexesResp])

  const { data: fieldsResp } = useQuery({ queryKey: ['screen-fields'], queryFn: listScreenFields })
  const screenFields: ScreenField[] = fieldsResp?.fields || []
  const fieldGroups = [
    {
      label: '估值字段',
      options: screenFields.filter((f) => f.group === 'valuation').map((f) => ({ value: f.name, label: f.label })),
    },
    {
      label: '财务字段',
      options: screenFields.filter((f) => f.group === 'financial').map((f) => ({ value: f.name, label: f.label })),
    },
  ]

  const { data: marketsResp } = useQuery({ queryKey: ['screen-markets'], queryFn: listScreenMarkets })
  const screenMarkets: ScreenMarket[] = marketsResp?.markets || []
  const marketAvailable = useMemo(() => {
    const m: Record<string, boolean> = {}
    for (const x of screenMarkets) m[x.market] = x.available
    return m
  }, [screenMarkets])
  const indexOptions = indexes.map((i) => {
    const avail = marketAvailable[i.market] ?? true
    return {
      value: i.key,
      label: avail ? `${i.key} · ${i.name}` : `${i.key} · ${i.name}(暂不可用)`,
      disabled: !avail,
    }
  })

  const [form] = Form.useForm()

  const poll = useQuery({
    queryKey: ['screen-job', jobId],
    queryFn: () => getScreenJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const s = q.state.data?.status
      return s === 'pending' || s === 'running' ? 2000 : false
    },
  })
  const job = poll.data
  const navigate = useNavigate()
  const config = useConfigStore((s) => s.config)
  const updateConfig = useConfigStore((s) => s.updateConfig)

  const rows: ScreenResultRow[] = (job?.result?.results as ScreenResultRow[] | undefined) ?? []
  const fieldLabel = useMemo(() => {
    const m: Record<string, string> = { score: '综合得分' }
    for (const f of screenFields) m[f.name] = f.label
    return m
  }, [screenFields])
  const columns = useMemo(() => {
    if (!rows.length) return []
    const keys = Object.keys(rows[0]).filter((k) => k !== 'symbol')
    return [
      { title: '代码', dataIndex: 'symbol', key: 'symbol', fixed: 'left' as const, width: 120 },
      ...keys.map((k) => ({
        title: fieldLabel[k] ?? k,
        dataIndex: k,
        key: k,
        render: (v: unknown) => (typeof v === 'number' ? formatCompact(v) : (v ?? '—')),
        sorter: (a: ScreenResultRow, b: ScreenResultRow) => ((a[k] as number) ?? -Infinity) - ((b[k] as number) ?? -Infinity),
      })),
    ]
  }, [rows, fieldLabel])

  const exportCsv = () => {
    const blob = new Blob(['﻿' + toCsv(rows)], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `screener-${jobId}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const startBacktest = () => {
    updateConfig('universe', {
      ...config.universe,
      symbols: rows.map((r) => r.symbol),
      benchmark: null,
    })
    navigate('/backtest')
  }

  const { data: historyJobs } = useQuery({
    queryKey: ['screen-jobs'],
    queryFn: () => listScreenJobs(20),
    refetchInterval: (query) => {
      const data = query.state.data as JobStatus[] | undefined
      return data?.some((j) => j.status === 'pending' || j.status === 'running') ? 3000 : false
    },
  })

  const mut = useMutation({
    mutationFn: createScreen,
    onSuccess: (resp) => {
      selectJob(resp.job_id)
      message.success(`选股任务已创建: ${resp.job_id}`)
      qc.invalidateQueries({ queryKey: ['screen-job', resp.job_id] })
      qc.invalidateQueries({ queryKey: ['screen-jobs'] })
    },
    onError: (e) => message.error(errDetail(e)),
  })

  const onSubmit = (v: { index?: string; when?: Date }) => {
    const when = v.when
    const idx = indexes.find((i) => i.key === v.index)
    mut.mutate({
      conditions: conditions
        .filter((c) => c.field.trim() && c.value.trim())
        .map((c) => {
          const num = Number(c.value)
          const value =
            c.op === 'between'
              ? c.value.split(',').map((s) => (Number.isNaN(Number(s)) ? s.trim() : Number(s.trim())))
              : c.op === 'in'
                ? c.value.split(',').map((s) => s.trim())
                : Number.isNaN(num)
                  ? c.value.trim()
                  : num
          return { field: c.field.trim(), op: c.op as ScreenOp, value }
        }),
      scores: scores.filter((s) => s.factor && s.weight),
      top_n: scores.length ? topN : null,
      index: v.index || null,
      market: idx?.market ?? null,
      when: when ? when.toISOString().slice(0, 10) : null,
      lookback_days: 120,
    })
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>选股</Typography.Title>

      <Card title="候选池 + 条件 + 打分">
        <Form form={form} layout="vertical" onFinish={onSubmit} initialValues={{ index: 'SP500' }}>
          <Form.Item name="index" label="宽基指数(市场由指数推断)">
            <Select options={indexOptions} showSearch optionFilterProp="label" />
          </Form.Item>
          <Form.Item name="when" label="截面日期(留空表示最近交易日)">
            <DatePicker style={{ width: 200 }} />
          </Form.Item>

          {/* 条件过滤器 */}
          <Typography.Text strong>筛选条件(取交集)</Typography.Text>
          <Alert
            type="info"
            showIcon
            style={{ marginTop: 8 }}
            message="筛选字段说明"
            description="估值字段(pe/pb/ps/市值)与财务字段(roe/毛利率/营收同比/净利同比等)取自不同数据源,数值单位见各字段名称。结果为空时可放宽条件或换指数重试。"
          />
          <Space direction="vertical" style={{ width: '100%', marginTop: 8 }}>
            {conditions.map((c, i) => (
              <Space key={i}>
                <Select
                  placeholder="选择字段"
                  value={c.field || undefined}
                  onChange={(v) => setConditions((arr) => arr.map((x, j) => (j === i ? { ...x, field: v } : x)))}
                  options={fieldGroups}
                  showSearch
                  optionFilterProp="label"
                  style={{ width: 180 }}
                />
                <Select
                  value={c.op}
                  onChange={(v) => setConditions((arr) => arr.map((x, j) => (j === i ? { ...x, op: v } : x)))}
                  options={OPS.map((o) => ({ value: o, label: o }))}
                  style={{ width: 110 }}
                />
                <Input
                  placeholder="值(逗号分隔: between/in)"
                  value={c.value}
                  onChange={(e) => setConditions((arr) => arr.map((x, j) => (j === i ? { ...x, value: e.target.value } : x)))}
                  style={{ width: 220 }}
                />
                <Button
                  size="small"
                  onClick={() => setConditions((arr) => arr.filter((_, j) => j !== i))}
                >删除</Button>
              </Space>
            ))}
            <Button size="small" onClick={() => setConditions((arr) => [...arr, { field: '', op: 'lt', value: '' }])}>
              + 添加条件
            </Button>
          </Space>

          {/* 打分因子 */}
          <Typography.Text strong style={{ display: 'block', marginTop: 16 }}>打分因子(可选)</Typography.Text>
          <Space direction="vertical" style={{ width: '100%', marginTop: 8 }}>
            {scores.map((s, i) => (
              <Space key={i}>
                <Select
                  placeholder="因子"
                  value={s.factor || undefined}
                  onChange={(v) => setScores((arr) => arr.map((x, j) => (j === i ? { ...x, factor: v } : x)))}
                  options={factors.map((f) => ({ value: f.name, label: f.name }))}
                  style={{ width: 160 }}
                />
                <InputNumber
                  placeholder="权重"
                  value={s.weight}
                  onChange={(v) => setScores((arr) => arr.map((x, j) => (j === i ? { ...x, weight: Number(v ?? 0) } : x)))}
                  style={{ width: 100 }}
                />
                <Select
                  value={s.direction}
                  onChange={(v) => setScores((arr) => arr.map((x, j) => (j === i ? { ...x, direction: v as 1 | -1 } : x)))}
                  options={[{ value: 1, label: '看多' }, { value: -1, label: '看空' }]}
                  style={{ width: 90 }}
                />
                <Button size="small" onClick={() => setScores((arr) => arr.filter((_, j) => j !== i))}>删除</Button>
              </Space>
            ))}
            <Space>
              <Button size="small" onClick={() => setScores((arr) => [...arr, { factor: '', weight: 1, direction: 1 as const }])}>
                + 添加打分因子
              </Button>
              <span>Top N: </span>
              <InputNumber
                min={1}
                value={topN ?? undefined}
                onChange={(v) => setTopN(v != null ? Number(v) : null)}
                disabled={!scores.length}
              />
            </Space>
          </Space>

          <Form.Item style={{ marginTop: 16 }}>
            <Button type="primary" htmlType="submit" loading={mut.isPending}>开始选股</Button>
          </Form.Item>
        </Form>
      </Card>

      {jobId && job && (
        <Card title={`任务 ${job.title || jobId}`}>
          <Space>
            <Tag color={job.status === 'done' ? 'success' : job.status === 'error' ? 'error' : 'processing'}>
              {job.status}
            </Tag>
            <Progress percent={Math.round(job.progress * 100)} status={job.status === 'error' ? 'exception' : 'active'} style={{ width: 300 }} />
          </Space>
          <div>{job.stage}</div>
          {job.error && <Typography.Text type="danger">{job.error}</Typography.Text>}
        </Card>
      )}

      {jobId && job?.status === 'done' && job.result && (
        <Card title={`选股结果 · ${job.result.count} 只`}>
          <Space direction="vertical" style={{ width: '100%' }}>
            {job.result.count === 0 ? (
              <Alert
                type="warning"
                showIcon
                message="没有标的通过筛选"
                description="可能原因:条件过严、字段值缺失,或截面日期无数据。可放宽条件、改选财务字段,或换指数重试。"
              />
            ) : (
              <>
                <Space wrap>
                  <Button size="small" onClick={exportCsv}>导出 CSV</Button>
                  <Button size="small" type="primary" onClick={startBacktest}>用这组股票发起回测</Button>
                  <Button
                    size="small"
                    onClick={() => { navigator.clipboard.writeText(location.href); message.success('链接已复制') }}
                  >复制链接</Button>
                  <Typography.Text type="secondary">得分降序(若有打分因子)</Typography.Text>
                </Space>
                <Table
                  dataSource={rows}
                  columns={columns}
                  rowKey="symbol"
                  size="small"
                  pagination={{ pageSize: 50 }}
                  scroll={{ x: true }}
                />
              </>
            )}
          </Space>
        </Card>
      )}

      <Card title="历史选股任务">
        <JobHistoryTable
          jobs={(historyJobs || []) as JobStatus[]}
          onOpen={(id) => selectJob(id)}
        />
      </Card>
    </Space>
  )
}

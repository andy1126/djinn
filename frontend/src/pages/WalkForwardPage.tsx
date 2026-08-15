import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Button, Card, Checkbox, Form, Input, InputNumber, message, Progress, Select, Space, Table, Tag, Typography,
} from 'antd'
import { createWalkForward, errDetail, listWalkForwards } from '@/api/client'
import { useConfigStore } from '@/store/configStore'
import JobHistoryTable from '@/components/JobHistoryTable'
import EquityCurveChart from '@/components/charts/EquityCurveChart'
import { useJobTransitionNotify } from '@/hooks/useJobTransitionNotify'
import type { JobStatus, SeriesData, WalkForwardReport, WFWindow } from '@/types'

const TARGET_OPTIONS = [
  'sharpe', 'sortino', 'calmar', 'total_return', 'annual_return',
  'max_drawdown', 'volatility', 'annual_volatility', 'n_trades',
]

function fmtPct(v: unknown): string {
  const n = Number(v ?? 0)
  return `${(n * 100).toFixed(2)}%`
}

function fmtNum(v: unknown): string {
  const n = Number(v ?? 0)
  return Number.isFinite(n) ? n.toFixed(3) : '—'
}

interface WalkFormValues {
  gridText: string
  is_days: number
  oos_days: number
  step?: number
  target: string
  min_is_sharpe?: number
  warmup_days: number
  parallel: boolean
}

export default function WalkForwardPage() {
  const qc = useQueryClient()
  const { config } = useConfigStore()
  const [form] = Form.useForm<WalkFormValues>()
  const [jobId, setJobId] = useState<string | null>(null)

  const { data: jobs } = useQuery({
    queryKey: ['walk-forwards'],
    queryFn: () => listWalkForwards(20),
    refetchInterval: (query) => {
      const data = query.state.data as JobStatus[] | undefined
      return data?.some((j) => j.status === 'pending' || j.status === 'running') ? 3000 : false
    },
  })
  useJobTransitionNotify(jobs, 'walk-forward')

  const mut = useMutation({
    mutationFn: createWalkForward,
    onSuccess: (resp) => {
      setJobId(resp.job_id)
      message.success(`Walk-Forward 任务已创建: ${resp.job_id}`)
      qc.invalidateQueries({ queryKey: ['walk-forwards'] })
    },
    onError: (e) => message.error(errDetail(e)),
  })

  const onSubmit = (v: WalkFormValues) => {
    const grid: Record<string, (number | string)[]> = {}
    v.gridText.split('\n').forEach((line) => {
      const [name, vals] = line.split(':')
      if (name && vals) {
        grid[name.trim()] = vals.split(',').map((s) => {
          const n = Number(s.trim())
          return Number.isNaN(n) ? s.trim() : n
        })
      }
    })
    if (Object.keys(grid).length === 0) {
      message.error('请至少配置一个扫轴(每行 name:value1,value2)')
      return
    }
    mut.mutate({
      config: {
        ...config,
        walk_forward: {
          is_days: v.is_days,
          oos_days: v.oos_days,
          step: v.step || null,
          target: v.target,
          min_is_sharpe: v.min_is_sharpe ?? null,
          warmup_days: v.warmup_days,
          grid,
        },
      },
      grid: null,
      target: null,
      parallel: v.parallel,
    })
  }

  const running = jobs?.find((j) => j.job_id === jobId)

  const report: WalkForwardReport | null = (() => {
    const r = running
    if (!r || r.status !== 'done') return null
    const rep = (r.result as { report?: unknown } | undefined)?.report
    return (rep as WalkForwardReport | undefined) ?? null
  })()

  const windows = report?.windows ?? []
  const metrics = report?.metrics ?? null
  const equity: SeriesData | null = report?.equity_curve ?? null

  const windowCols = [
    { title: '#', key: 'no', width: 44, render: (_: unknown, w: WFWindow) => w.no },
    { title: 'IS 窗口', key: 'is', render: (_: unknown, w: WFWindow) => `${w.is_start}~${w.is_end}` },
    { title: 'OOS 窗口', key: 'oos', render: (_: unknown, w: WFWindow) => `${w.oos_start}~${w.oos_end}` },
    {
      title: '部署', key: 'deployed', width: 80,
      render: (_: unknown, w: WFWindow) =>
        w.deployed ? <Tag color="green">已部署</Tag> : <Tag color="red">未部署</Tag>,
    },
    {
      title: 'IS 最优参数', key: 'params',
      render: (_: unknown, w: WFWindow) => (
        <Space direction="vertical" size={0}>
          {w.best_params
            ? Object.entries(w.best_params).map(([k, v]) => (
              <Typography.Text key={k} code>{k}={String(v)}</Typography.Text>
            ))
            : <span>—</span>}
        </Space>
      ),
    },
    { title: 'OOS sharpe', key: 'osharpe', render: (_: unknown, w: WFWindow) =>
      w.oos_metrics ? fmtNum(w.oos_metrics.sharpe) : '—' },
    { title: 'OOS sortino', key: 'osortino', render: (_: unknown, w: WFWindow) =>
      w.oos_metrics ? fmtNum(w.oos_metrics.sortino) : '—' },
    { title: 'OOS 总收益', key: 'oret', render: (_: unknown, w: WFWindow) =>
      w.oos_metrics ? fmtPct(w.oos_metrics.total_return) : '—' },
    { title: 'OOS 回撤', key: 'omdd', render: (_: unknown, w: WFWindow) =>
      w.oos_metrics ? fmtPct(w.oos_metrics.max_drawdown) : '—' },
    { title: '交易数', key: 'otrades', render: (_: unknown, w: WFWindow) =>
      w.oos_metrics ? String(w.oos_metrics.n_trades ?? '—') : '—' },
  ]

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>Walk-Forward 滚动样本外验证</Typography.Title>

      <Card title="分析配置">
        <Form
          form={form}
          layout="vertical"
          onFinish={onSubmit}
          initialValues={{
            is_days: 250,
            oos_days: 125,
            step: undefined,
            target: 'sharpe',
            min_is_sharpe: undefined,
            warmup_days: 300,
            parallel: false,
            gridText: 'fast:5,10,20\nslow:20,30,60',
          }}
        >
          <Space wrap align="start">
            <Form.Item name="is_days" label="样本内(IS)天数" rules={[{ required: true }]}>
              <InputNumber min={1} />
            </Form.Item>
            <Form.Item name="oos_days" label="样本外(OOS)天数" rules={[{ required: true }]}>
              <InputNumber min={1} />
            </Form.Item>
            <Form.Item name="step" label="滚动步长(默认=OOS,非重叠)">
              <InputNumber min={1} placeholder="可选" />
            </Form.Item>
            <Form.Item name="target" label="IS 优化目标" rules={[{ required: true }]}>
              <Select
                style={{ width: 180 }}
                options={TARGET_OPTIONS.map((t) => ({ value: t, label: t }))}
              />
            </Form.Item>
            <Form.Item name="min_is_sharpe" label="IS 达标门槛(不达标则空仓)">
              <InputNumber step={0.1} placeholder="可选" />
            </Form.Item>
            <Form.Item name="warmup_days" label="暖机交易日" rules={[{ required: true }]}>
              <InputNumber min={0} />
            </Form.Item>
            <Form.Item name="parallel" valuePropName="checked" label="并行">
              <Checkbox>IS 组合并行</Checkbox>
            </Form.Item>
          </Space>
          <Form.Item name="gridText" label="参数网格(每行 name:value1,value2)" rules={[{ required: true }]}>
            <Input.TextArea rows={3} placeholder="fast:5,10,20&#10;slow:20,30,60" />
          </Form.Item>
          <Form.Item label="基础配置">
            <span>
              策略: {config.strategy.name} · 标的: {config.universe.symbols.join(',')} ·
              区间: {config.period.start}~{config.period.end}
            </span>
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={mut.isPending}>开始分析</Button>
          </Form.Item>
        </Form>
      </Card>

      {jobId && running && (
        <Card title={`当前任务 ${running.title || running.job_id}`}>
          <Progress
            percent={Math.round(running.progress * 100)}
            status={running.status === 'done' ? 'success' : running.status === 'error' ? 'exception' : 'active'}
          />
          <div>阶段: {running.stage}</div>
          {running.error && <Typography.Text type="danger">{running.error}</Typography.Text>}
        </Card>
      )}

      {metrics && equity && (
        <Card title={`拼接样本外净值(${report?.target ?? 'sharpe'})`}>
          <EquityCurveChart equity={equity} height={320} />
          <Space wrap style={{ marginTop: 12 }}>
            <Tag color="blue">sharpe {fmtNum(metrics.sharpe)}</Tag>
            <Tag>sortino {fmtNum(metrics.sortino)}</Tag>
            <Tag>calmar {fmtNum(metrics.calmar)}</Tag>
            <Tag color="green">年化 {fmtPct(metrics.annual_return)}</Tag>
            <Tag color="red">回撤 {fmtPct(metrics.max_drawdown)}</Tag>
            <Tag>交易日 {String(metrics.n_days ?? '—')}</Tag>
          </Space>
        </Card>
      )}

      {windows.length > 0 && (
        <Card title="逐窗口结果">
          <Table
            size="small"
            dataSource={windows}
            rowKey={(w) => String(w.no)}
            columns={windowCols}
            pagination={{ pageSize: 10 }}
            scroll={{ x: 'max-content' }}
          />
        </Card>
      )}

      <Card title="历史任务">
        <JobHistoryTable jobs={(jobs || []) as JobStatus[]} onOpen={(id) => setJobId(id)} />
      </Card>
    </Space>
  )
}

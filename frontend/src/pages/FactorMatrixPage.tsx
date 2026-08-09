import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Button, Card, DatePicker, Form, InputNumber, message, Progress, Select, Space, Spin, Table, Tag, Typography,
} from 'antd'
import {
  listFactors,
  createFactorMatrix,
  listFactorMatrices,
  getFactorMatrixJob,
  getFactorMatrixReport,
} from '@/api/client'
import type { FactorInfo, FactorMatrixPoint, FactorMatrixReport, JobStatus, ParamSchema } from '@/types'
import MatrixHeatmap from '@/components/charts/MatrixHeatmap'

const { RangePicker } = DatePicker

interface FormValues {
  index?: string
  market?: string
  range: [Date, Date]
  ic_method: string
}

interface PointDraft extends FactorMatrixPoint {
  /** 临时参数(展开行)。 */
  params: Record<string, number | string | boolean | null>
  uid: number
}

function paramWidget(p: ParamSchema, value: any, onSet: (v: any) => void) {
  const label = <span><b>{p.name}</b> <Typography.Text type="secondary">{p.description || ''}</Typography.Text></span>
  if (p.choices && p.choices.length > 0) {
    return (
      <Form.Item key={p.name} label={label}>
        <Select
          value={value ?? p.default}
          onChange={onSet}
          options={p.choices.map((c) => ({ label: String(c), value: c }))}
          style={{ width: '100%' }}
        />
      </Form.Item>
    )
  }
  if (p.type === 'bool' || p.type === 'boolean') {
    return (
      <Form.Item key={p.name} label={label}>
        <Select
          value={value ?? p.default}
          onChange={onSet}
          options={[{ label: 'true', value: true }, { label: 'false', value: false }]}
          style={{ width: '100%' }}
        />
      </Form.Item>
    )
  }
  if (p.type === 'str' || p.type === 'string' || p.type === 'NoneType') {
    return (
      <Form.Item key={p.name} label={label}>
        <Select placeholder="因子" style={{ width: '100%' }} />
      </Form.Item>
    )
  }
  return (
    <Form.Item key={p.name} label={label}>
      <InputNumber
        value={value != null ? Number(value) : Number(p.default)}
        onChange={(v) => onSet(v ?? 0)}
        min={p.min != null ? Number(p.min) : undefined}
        max={p.max != null ? Number(p.max) : undefined}
        style={{ width: '100%' }}
      />
    </Form.Item>
  )
}

let _uid = 0

/**
 * 多因子诊断页:选 2~8 个因子 + 权重 + 方向 → 后台任务 → 相关矩阵热力图 + 每因子 IC 汇总。
 */
export default function FactorMatrixPage() {
  const qc = useQueryClient()
  const [form] = Form.useForm<FormValues>()
  const [jobId, setJobId] = useState<string | null>(null)
  const [report, setReport] = useState<FactorMatrixReport | null>(null)
  const [drafts, setDrafts] = useState<PointDraft[]>([])

  const { data: factorsResp, isLoading: factorsLoading } = useQuery({
    queryKey: ['factors'],
    queryFn: listFactors,
  })
  const factors: FactorInfo[] = factorsResp?.factors || []

  const { data: historyJobs } = useQuery({
    queryKey: ['factor-matrix-jobs'],
    queryFn: () => listFactorMatrices(20),
    refetchInterval: (query) => {
      const data = query.state.data as JobStatus[] | undefined
      return data?.some((j) => j.status === 'pending' || j.status === 'running') ? 3000 : false
    },
  })

  const poll = useQuery({
    queryKey: ['factor-matrix-job', jobId],
    queryFn: () => getFactorMatrixJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const s = (q.state.data as any)?.status
      return s === 'pending' || s === 'running' ? 2000 : false
    },
  })
  const job = poll.data as any
  useQuery({
    queryKey: ['factor-matrix-report', jobId],
    queryFn: async () => {
      const r = await getFactorMatrixReport(jobId!)
      setReport(r)
      return r
    },
    enabled: !!jobId && job?.status === 'done',
  })

  const mut = useMutation({
    mutationFn: createFactorMatrix,
    onSuccess: (resp) => {
      setJobId(resp.job_id)
      setReport(null)
      message.success(`多因子诊断任务已创建: ${resp.job_id}`)
      qc.invalidateQueries({ queryKey: ['factor-matrix-job', resp.job_id] })
      qc.invalidateQueries({ queryKey: ['factor-matrix-jobs'] })
    },
    onError: (e: any) => message.error(e?.response?.data?.detail || '创建失败'),
  })

  const addFactor = (name?: string) => {
    if (!name) return
    if (drafts.some((d) => d.factor === name)) {
      message.warning('该因子已添加')
      return
    }
    if (drafts.length >= 8) {
      message.warning('最多 8 个因子')
      return
    }
    _uid += 1
    setDrafts((prev) => [...prev, { factor: name, weight: 1.0, direction: 1, params: {}, uid: _uid }])
  }
  const removeFactor = (uid: number) =>
    setDrafts((prev) => prev.filter((d) => d.uid !== uid))
  const update = (uid: number, patch: Partial<PointDraft>) =>
    setDrafts((prev) => prev.map((d) => (d.uid === uid ? { ...d, ...patch } : d)))

  const onSubmit = (v: FormValues) => {
    if (drafts.length < 2) {
      message.error('至少需要 2 个因子')
      return
    }
    mut.mutate({
      factors: drafts.map((d) => ({
        factor: d.factor,
        weight: d.weight,
        direction: d.direction,
        params: d.params,
      })),
      index: v.index || null,
      market: v.market || null,
      start: v.range[0].toISOString().slice(0, 10),
      end: v.range[1].toISOString().slice(0, 10),
      ic_method: v.ic_method,
      periods: [1, 5, 10],
    })
  }

  // IC 摘要表(展平 period × factor)
  const icRows: any[] = []
  if (report) {
    Object.entries(report.ic_summary).forEach(([period, byFactor]) => {
      Object.entries(byFactor).forEach(([fname, s]) => {
        icRows.push({ key: `${period}-${fname}`, period, factor: fname, ...s })
      })
    })
  }
  const turnoverRows = report
    ? Object.entries(report.turnover).map(([name, v]) => ({ key: name, factor: name, turnover: v }))
    : []

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>多因子诊断</Typography.Title>

      <Card title="因子选择(2~8 个)">
        <Space direction="vertical" style={{ width: '100%' }} size="small">
          {drafts.map((d) => (
            <Card key={d.uid} size="small" title={d.factor}>
              <Space wrap>
                <span>权重:</span>
                <InputNumber
                  min={0}
                  step={0.1}
                  value={d.weight}
                  onChange={(v) => update(d.uid, { weight: v ?? 1 })}
                />
                <span>方向:</span>
                <Select
                  value={d.direction}
                  onChange={(v) => update(d.uid, { direction: v as 1 | -1 })}
                  options={[{ value: 1, label: '多(+1)' }, { value: -1, label: '空(-1)' }]}
                  style={{ width: 100 }}
                />
                <Button danger onClick={() => removeFactor(d.uid)}>删除</Button>
              </Space>
              {(() => {
                const info = factors.find((f) => f.name === d.factor)
                if (info && info.params.length > 0) {
                  return (
                    <Card size="small" style={{ marginTop: 8 }} title={`${d.factor} 参数`}>
                      {info.params.map((p) =>
                        paramWidget(p, d.params[p.name], (val) =>
                          update(d.uid, { params: { ...d.params, [p.name]: val } }),
                        ),
                      )}
                    </Card>
                  )
                }
                return null
              })()}
            </Card>
          ))}
          <Select
            placeholder="选择并添加因子"
            loading={factorsLoading}
            style={{ width: 300 }}
            onChange={(name) => addFactor(name as string)}
            options={factors.map((f) => ({ value: f.name, label: `${f.name} (${f.category})` }))}
            showSearch
            optionFilterProp="label"
          />
        </Space>
      </Card>

      <Card title="参数与标的池">
        <Form
          form={form}
          layout="vertical"
          onFinish={onSubmit}
          initialValues={{ index: 'CSI300', market: 'CN', ic_method: 'spearman' }}
        >
          <Form.Item name="index" label="宽基指数">
            <Select options={['CSI300', 'CSI500', 'CSI800', 'HSI', 'SP500'].map((k) => ({ value: k, label: k }))} />
          </Form.Item>
          <Form.Item name="market" label="市场">
            <Select options={['CN', 'HK', 'US'].map((m) => ({ value: m, label: m }))} />
          </Form.Item>
          <Form.Item name="range" label="区间" rules={[{ required: true, message: '请选择区间' }]}>
            <RangePicker />
          </Form.Item>
          <Form.Item name="ic_method" label="IC 相关方法">
            <Select options={[{ value: 'spearman', label: 'Spearman' }, { value: 'pearson', label: 'Pearson' }]} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={mut.isPending} disabled={drafts.length < 2}>
              开始诊断
            </Button>
          </Form.Item>
        </Form>
      </Card>

      {jobId && job && (
        <Card title={`任务 ${job.title || jobId}`}>
          <Space>
            <Tag color={job.status === 'done' ? 'success' : job.status === 'error' ? 'error' : 'processing'}>
              {job.status}
            </Tag>
            <Progress
              percent={Math.round(job.progress * 100)}
              status={job.status === 'error' ? 'exception' : 'active'}
              style={{ width: 300 }}
            />
          </Space>
          <div>{job.stage}</div>
          {job.error && <Typography.Text type="danger">{job.error}</Typography.Text>}
        </Card>
      )}

      {report && (
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Card title="因子两两相关矩阵">
            <MatrixHeatmap matrix={report.correlation} height={460} />
          </Card>
          <Card title="各因子 IC 汇总">
            <Table
              size="small"
              dataSource={icRows}
              pagination={{ pageSize: 20 }}
              columns={[
                { title: '期', dataIndex: 'period', key: 'period', width: 80 },
                { title: '因子', dataIndex: 'factor', key: 'factor' },
                {
                  title: 'IC 均值', dataIndex: 'ic_mean', key: 'ic_mean',
                  render: (v: number) => (v ?? 0).toFixed(4),
                },
                {
                  title: 'ICIR', dataIndex: 'icir', key: 'icir',
                  render: (v: number) => (v ?? 0).toFixed(4),
                },
                {
                  title: 'IC 正值占比', dataIndex: 'ic_pos_ratio', key: 'ic_pos_ratio',
                  render: (v: number) => `${((v ?? 0) * 100).toFixed(1)}%`,
                },
              ]}
            />
          </Card>
          <Card title="各因子换手">
            <Table
              size="small"
              dataSource={turnoverRows}
              pagination={false}
              columns={[
                { title: '因子', dataIndex: 'factor', key: 'factor' },
                {
                  title: '换手率', dataIndex: 'turnover', key: 'turnover',
                  render: (v: number) => (v ?? 0).toFixed(3),
                },
              ]}
            />
          </Card>
        </Space>
      )}
      {jobId && !job && <Spin />}

      <Card title="历史多因子诊断任务">
        <Table
          columns={[
            {
              title: '任务', key: 'job_id',
              render: (_: any, r: JobStatus) => (
                <Space direction="vertical" size={0}>
                  <span>{r.title || r.job_id}</span>
                  <Typography.Text code type="secondary">{r.job_id}</Typography.Text>
                </Space>
              ),
            },
            { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => <Tag color={s === 'done' ? 'success' : s === 'error' ? 'error' : 'processing'}>{s}</Tag> },
            { title: '进度', dataIndex: 'progress', key: 'progress', render: (p: number) => <Progress percent={Math.round(p * 100)} size="small" /> },
            { title: '阶段', dataIndex: 'stage', key: 'stage' },
            { title: '错误', dataIndex: 'error', key: 'error', render: (e: string) => e || '—' },
            {
              title: '操作', key: 'action', render: (_: any, r: JobStatus) => (
                <Button size="small" onClick={() => { setJobId(r.job_id); setReport(null) }}>查看</Button>
              ),
            },
          ]}
          dataSource={(historyJobs || []) as JobStatus[]}
          rowKey="job_id"
          size="small"
          pagination={{ pageSize: 10 }}
        />
      </Card>
    </Space>
  )
}
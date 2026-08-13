import { useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Alert, Button, Card, Collapse, DatePicker, Form, InputNumber, message, Progress, Select, Space, Spin, Table, Tag, Tooltip, Typography,
} from 'antd'
import { QuestionCircleOutlined } from '@ant-design/icons'
import {
  listFactors,
  listIndexes,
  createFactorAnalysis,
  listFactorAnalyses,
  getFactorAnalysisJob,
  getFactorAnalysisReport,
} from '@/api/client'
import type { FactorInfo, IndexInfo, JobStatus, ParamSchema } from '@/types'
import QuantileCurveChart from '@/components/charts/QuantileCurveChart'
import ICBarChart from '@/components/charts/ICBarChart'
import { IC_RANGE_HELP, METRIC_TIP } from '@/components/factorMetricsHelp'

const { RangePicker } = DatePicker

interface FormValues {
  factor: string
  index?: string
  range: [Date, Date]
  ic_method: string
  n_quantiles: number
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

function MetricTag({ label, value, tip }: { label: string; value: string; tip: string }) {
  return (
    <Tooltip title={tip}>
      <Tag style={{ cursor: 'help', userSelect: 'none' }}>
        {label}: {value} <QuestionCircleOutlined style={{ color: '#bbb' }} />
      </Tag>
    </Tooltip>
  )
}

/**
 * 因子分析页:选择因子 + 标的池 + 区间 → 后台任务 → IC / 分层 / 衰减报告。
 */
export default function FactorAnalysisPage() {
  const qc = useQueryClient()
  const [form] = Form.useForm<FormValues>()
  const [jobId, setJobId] = useState<string | null>(null)
  const [report, setReport] = useState<any | null>(null)
  const [selected, setSelected] = useState<FactorInfo | null>(null)
  const [params, setParams] = useState<Record<string, any>>({})

  const { data: factorsResp, isLoading: factorsLoading } = useQuery({
    queryKey: ['factors'],
    queryFn: listFactors,
  })
  const factors: FactorInfo[] = factorsResp?.factors || []

  const { data: indexesResp } = useQuery({ queryKey: ['indexes'], queryFn: listIndexes })
  const indexes: IndexInfo[] = indexesResp?.indexes || []
  const indexByKey = useMemo(() => {
    const m: Record<string, IndexInfo> = {}
    for (const i of indexes) m[i.key] = i
    return m
  }, [indexes])
  const indexOptions = indexes.map((i) => ({ value: i.key, label: `${i.key} · ${i.name}` }))
  const watchIndex = Form.useWatch('index', form)
  const indexMarket = watchIndex ? indexByKey[watchIndex]?.market : undefined
  const isValuation = (f: FactorInfo) => f.category === 'value' || f.category === 'size'
  const factorOptions = factors.map((f) => {
    const blocked = indexMarket === 'CN' && isValuation(f)
    return {
      value: f.name,
      label: blocked ? `${f.name} (${f.category}) · A股估值源暂不可达` : `${f.name} (${f.category})`,
      disabled: blocked,
    }
  })

  const { data: historyJobs } = useQuery({
    queryKey: ['factor-analysis-jobs'],
    queryFn: () => listFactorAnalyses(20),
    refetchInterval: (query) => {
      const data = query.state.data as JobStatus[] | undefined
      return data?.some((j) => j.status === 'pending' || j.status === 'running') ? 3000 : false
    },
  })

  const poll = useQuery({
    queryKey: ['factor-analysis-job', jobId],
    queryFn: () => getFactorAnalysisJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const s = (q.state.data as any)?.status
      return s === 'pending' || s === 'running' ? 2000 : false
    },
  })
  const job = poll.data as any
  // 完成后取报告
  useQuery({
    queryKey: ['factor-analysis-report', jobId],
    queryFn: async () => {
      const r = await getFactorAnalysisReport(jobId!)
      setReport(r)
      return r
    },
    enabled: !!jobId && job?.status === 'done',
  })

  const mut = useMutation({
    mutationFn: createFactorAnalysis,
    onSuccess: (resp) => {
      setJobId(resp.job_id)
      setReport(null)
      message.success(`因子分析任务已创建: ${resp.job_id}`)
      qc.invalidateQueries({ queryKey: ['factor-analysis-job', resp.job_id] })
      qc.invalidateQueries({ queryKey: ['factor-analysis-jobs'] })
    },
    onError: (e: any) => message.error(e?.response?.data?.detail || '创建失败'),
  })

  const onSubmit = (v: FormValues) => {
    const idx = indexes.find((i) => i.key === v.index)
    mut.mutate({
      factor: v.factor,
      params,
      index: v.index || null,
      market: idx?.market ?? null,
      start: v.range[0].toISOString().slice(0, 10),
      end: v.range[1].toISOString().slice(0, 10),
      ic_method: v.ic_method,
      n_quantiles: v.n_quantiles,
      periods: [1, 5, 10],
    })
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>因子分析</Typography.Title>

      <Card title="单因子分析">
        <Form
          form={form}
          layout="vertical"
          onFinish={onSubmit}
          initialValues={{ index: 'CSI300', ic_method: 'spearman', n_quantiles: 5 }}
        >
          <Form.Item name="factor" label="因子" rules={[{ required: true, message: '请选择因子' }]}>
            <Select
              placeholder="选择因子"
              loading={factorsLoading}
              onChange={(name) => {
                setSelected(factors.find((f) => f.name === name) || null)
                setParams({})
              }}
              options={factorOptions}
            />
          </Form.Item>

          {selected && selected.params.length > 0 && (
            <Card size="small" title={`${selected.name} 参数`}>
              {selected.params.map((p) =>
                paramWidget(p, params[p.name], (val) => setParams((prev) => ({ ...prev, [p.name]: val }))),
              )}
            </Card>
          )}

          <Form.Item name="index" label="宽基指数">
            <Select options={indexOptions} showSearch optionFilterProp="label" />
          </Form.Item>
          {indexMarket === 'CN' && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message="A 股估值因子暂不可用"
              description="A 股估值源(东财)当前网络不可达,故 ep/bp/sp/size 等估值类因子无法计算;可换美股/港股指数使用估值因子,或改用财务/行情类因子。"
            />
          )}
          <Form.Item name="range" label="回测区间" rules={[{ required: true, message: '请选择区间' }]}>
            <RangePicker />
          </Form.Item>
          <Form.Item name="ic_method" label="IC 相关方法">
            <Select options={[{ value: 'spearman', label: 'Spearman' }, { value: 'pearson', label: 'Pearson' }]} />
          </Form.Item>
          <Form.Item name="n_quantiles" label="分层数">
            <InputNumber min={2} max={20} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={mut.isPending}>开始分析</Button>
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

      {report && (
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Card title={`IC 汇总 · ${report.factor_name}`}>
            <Space size="large" wrap>
              <MetricTag label="IC 均值" value={(report.ic_summary.ic_mean ?? 0).toFixed(4)} tip={METRIC_TIP['IC 均值']} />
              <MetricTag label="ICIR" value={(report.ic_summary.icir ?? 0).toFixed(4)} tip={METRIC_TIP['ICIR']} />
              <MetricTag label="IC 正值占比" value={`${((report.ic_summary.ic_pos_ratio ?? 0) * 100).toFixed(1)}%`} tip={METRIC_TIP['IC 正值占比']} />
              <MetricTag label="单调性" value={(report.monotonicity ?? 0).toFixed(3)} tip={METRIC_TIP['单调性']} />
              <MetricTag label="换手" value={(report.turnover ?? 0).toFixed(3)} tip={METRIC_TIP['换手']} />
            </Space>
          </Card>
          <Collapse
            size="small"
            items={[{
              key: 'help',
              label: <Space size={4}><QuestionCircleOutlined />指标说明(IC 区间含义)</Space>,
              children: (
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  <div>
                    <Typography.Text strong>IC 区间经验参考(秩相关、截面)</Typography.Text>
                    <ul style={{ margin: '8px 0 0', paddingLeft: 18 }}>
                      {IC_RANGE_HELP.map((r) => (
                        <li key={r.range}><Typography.Text code>{r.range}</Typography.Text> — {r.meaning}</li>
                      ))}
                    </ul>
                    <Typography.Text type="secondary">符号:正 = 因子值越大、未来收益越高;负 = 相反。</Typography.Text>
                  </div>
                  <div>
                    <Typography.Text strong>其他指标</Typography.Text>
                    <ul style={{ margin: '8px 0 0', paddingLeft: 18 }}>
                      {Object.entries(METRIC_TIP).map(([term, desc]) => (
                        <li key={term}><Typography.Text strong>{term}</Typography.Text>: {desc}</li>
                      ))}
                    </ul>
                  </div>
                </Space>
              ),
            }]}
          />
          <Card title="IC 时序">
            <ICBarChart ic={report.ic} />
          </Card>
          <Card title="分层累计收益">
            <QuantileCurveChart quantileReturns={report.quantile_returns} />
          </Card>
        </Space>
      )}
      {jobId && !job && <Spin />}

      <Card title="历史因子分析任务">
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
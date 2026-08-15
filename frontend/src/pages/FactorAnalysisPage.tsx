import { useEffect, useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Alert, Button, Card, Col, Collapse, DatePicker, Form, InputNumber, message, Progress, Row, Select, Space, Spin, Tag, Tooltip, Typography,
} from 'antd'
import { QuestionCircleOutlined } from '@ant-design/icons'
import { useSearchParams } from 'react-router-dom'
import {
  listFactors,
  listIndexes,
  createFactorAnalysis,
  listFactorAnalyses,
  getFactorAnalysisJob,
  getFactorAnalysisReport,
  errDetail,
} from '@/api/client'
import type { FactorInfo, FactorReport, IndexInfo, JobStatus } from '@/types'
import QuantileCurveChart from '@/components/charts/QuantileCurveChart'
import ICBarChart from '@/components/charts/ICBarChart'
import JobHistoryTable from '@/components/JobHistoryTable'
import { useJobTransitionNotify } from '@/hooks/useJobTransitionNotify'
import ParamField from '@/components/ParamFields'
import QueryErrorAlert from '@/components/QueryErrorAlert'
import { IC_RANGE_HELP, METRIC_TIP } from '@/components/factorMetricsHelp'

const { RangePicker } = DatePicker

/** 宽基指数卡片的市场中文标签。 */
const MARKET_LABEL: Record<string, string> = { CN: 'A股', US: '美股', HK: '港股' }

/** 后端 _recommend_freq 输出的调仓频率档 → 中文(C11)。 */
const FREQ_LABEL: Record<string, string> = {
  daily: '日频', weekly: '周频', monthly: '月频', quarterly: '季频',
}

interface FormValues {
  factor: string
  index?: string
  range: [Date, Date]
  ic_method: string
  n_quantiles: number
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
  const [searchParams, setSearchParams] = useSearchParams()
  // F14:深链 —— 从 URL ?job=<id> 恢复;切换任务时写回 URL
  const [jobId, setJobId] = useState<string | null>(() => searchParams.get('job'))
  const [selected, setSelected] = useState<FactorInfo | null>(null)
  const [params, setParams] = useState<Record<string, number | string | boolean | null>>({})

  useEffect(() => {
    if (jobId) setSearchParams({ job: jobId }, { replace: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  const {
    data: factorsResp,
    isLoading: factorsLoading,
    error: factorsError,
    refetch: refetchFactors,
  } = useQuery({
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
  // F16:因子分析任务 running→终态 → 全局通知
  useJobTransitionNotify(historyJobs, 'factor-analysis')

  const poll = useQuery({
    queryKey: ['factor-analysis-job', jobId],
    queryFn: () => getFactorAnalysisJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const s = q.state.data?.status
      return s === 'pending' || s === 'running' ? 2000 : false
    },
  })
  const job = poll.data
  // 完成后取报告(F6:queryFn 无副作用,直接消费 data)
  const { data: report } = useQuery<FactorReport>({
    queryKey: ['factor-analysis-report', jobId],
    queryFn: () => getFactorAnalysisReport(jobId!),
    enabled: !!jobId && job?.status === 'done',
  })

  const mut = useMutation({
    mutationFn: createFactorAnalysis,
    onSuccess: (resp) => {
      setJobId(resp.job_id)
      message.success(`因子分析任务已创建: ${resp.job_id}`)
      qc.invalidateQueries({ queryKey: ['factor-analysis-job', resp.job_id] })
      qc.invalidateQueries({ queryKey: ['factor-analysis-jobs'] })
    },
    onError: (e) => message.error(errDetail(e)),
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
        {factorsError && <QueryErrorAlert error={factorsError} retry={refetchFactors} />}
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
              {selected.params.map((p) => (
                <ParamField
                  key={p.name}
                  p={p}
                  value={params[p.name]}
                  onSet={(val) => setParams((prev) => ({ ...prev, [p.name]: val as number | string | boolean | null }))}
                />
              ))}
            </Card>
          )}

          <Form.Item name="index" label="宽基指数">
            <Row gutter={[8, 8]}>
              {indexes.map((i) => {
                const selected = watchIndex === i.key
                return (
                  <Col key={i.key} xs={12} sm={8} lg={6}>
                    <Card
                      size="small"
                      hoverable
                      onClick={() => form.setFieldValue('index', i.key)}
                      style={{
                        borderColor: selected ? '#1677ff' : undefined,
                        cursor: 'pointer',
                        background: selected ? 'rgba(22,119,255,0.06)' : undefined,
                      }}
                    >
                      <Space direction="vertical" size={2} style={{ width: '100%' }}>
                        <Typography.Text strong>{i.key}</Typography.Text>
                        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                          {i.name}
                        </Typography.Text>
                        <Tag color={selected ? 'blue' : 'default'}>
                          {MARKET_LABEL[i.market] || i.market}
                        </Tag>
                      </Space>
                    </Card>
                  </Col>
                )
              })}
            </Row>
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
          {report.data_caveats?.length > 0 && (
            <Alert
              type="warning"
              showIcon
              message="数据口径提示"
              description={report.data_caveats.join('; ')}
            />
          )}
          <Card
            title={`IC 汇总 · ${report.factor_name}`}
            extra={
              <Button
                size="small"
                onClick={() => { navigator.clipboard.writeText(location.href); message.success('链接已复制') }}
              >复制链接</Button>
            }
          >
            <Space size="large" wrap>
              <MetricTag label="IC 均值" value={(report.ic_summary.ic_mean ?? 0).toFixed(4)} tip={METRIC_TIP['IC 均值']} />
              <MetricTag label="ICIR" value={(report.ic_summary.icir ?? 0).toFixed(4)} tip={METRIC_TIP['ICIR']} />
              <MetricTag label="t 值" value={`${(report.ic_summary.ic_t ?? 0).toFixed(2)} (p=${(report.ic_summary.ic_pvalue ?? 1).toFixed(3)})`} tip={METRIC_TIP['t 值']} />
              <MetricTag label="IC 正值占比" value={`${((report.ic_summary.ic_pos_ratio ?? 0) * 100).toFixed(1)}%`} tip={METRIC_TIP['IC 正值占比']} />
              <MetricTag label="单调性" value={(report.monotonicity ?? 0).toFixed(3)} tip={METRIC_TIP['单调性']} />
              <MetricTag label="换手" value={(report.turnover ?? 0).toFixed(3)} tip={METRIC_TIP['换手']} />
              {report.recommended_rebalance && (
                <Tooltip title={METRIC_TIP['建议调仓频率']}>
                  <Tag color="blue" style={{ cursor: 'help', userSelect: 'none' }}>
                    建议调仓频率: {FREQ_LABEL[report.recommended_rebalance] ?? report.recommended_rebalance}{' '}
                    <QuestionCircleOutlined style={{ color: '#bbb' }} />
                  </Tag>
                </Tooltip>
              )}
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
        <JobHistoryTable
          jobs={(historyJobs || []) as JobStatus[]}
          onOpen={(id) => setJobId(id)}
        />
      </Card>
    </Space>
  )
}
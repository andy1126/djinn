import { useEffect, useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Alert, Button, Card, DatePicker, Form, InputNumber, message, Progress, Select, Space, Spin, Switch, Table, Tag, Tooltip, Typography,
} from 'antd'
import { QuestionCircleOutlined } from '@ant-design/icons'
import { useSearchParams } from 'react-router-dom'
import {
  listFactors,
  listIndexes,
  createFactorMatrix,
  listFactorMatrices,
  getFactorMatrixJob,
  getFactorMatrixReport,
  errDetail,
} from '@/api/client'
import type { FactorInfo, FactorMatrixPoint, FactorMatrixReport, IndexInfo, JobStatus } from '@/types'
import MatrixHeatmap from '@/components/charts/MatrixHeatmap'
import JobHistoryTable from '@/components/JobHistoryTable'
import { useJobTransitionNotify } from '@/hooks/useJobTransitionNotify'
import ParamField from '@/components/ParamFields'
import QueryErrorAlert from '@/components/QueryErrorAlert'
import { CORR_MATRIX_TIP, METRIC_TIP } from '@/components/factorMetricsHelp'

const { RangePicker } = DatePicker

interface FormValues {
  index?: string
  range: [Date, Date]
  ic_method: string
  orthogonalized?: boolean
}

interface PointDraft extends FactorMatrixPoint {
  /** 临时参数(展开行)。 */
  params: Record<string, number | string | boolean | null>
  uid: number
}

// F10:随机种子避免 HMR 重载后计数器归零与既有 state 的 uid 冲突
let _uid = Math.floor(Math.random() * 1_000_000)

/** 带 tooltip 的表头文本(悬停显示指标含义)。 */
function HelpTitle({ text, tip }: { text: string; tip: string }) {
  return (
    <Tooltip title={tip}>
      <Space size={4} style={{ cursor: 'help' }}>{text} <QuestionCircleOutlined style={{ color: '#bbb' }} /></Space>
    </Tooltip>
  )
}

/**
 * 多因子诊断页:选 2~8 个因子 + 权重 + 方向 → 后台任务 → 相关矩阵热力图 + 每因子 IC 汇总。
 */
export default function FactorMatrixPage() {
  const qc = useQueryClient()
  const [form] = Form.useForm<FormValues>()
  const [searchParams, setSearchParams] = useSearchParams()
  // F14:深链 —— 从 URL ?job=<id> 恢复;切换任务时写回 URL
  const [jobId, setJobId] = useState<string | null>(() => searchParams.get('job'))

  useEffect(() => {
    if (jobId) setSearchParams({ job: jobId }, { replace: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])
  const [drafts, setDrafts] = useState<PointDraft[]>([])

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
    queryKey: ['factor-matrix-jobs'],
    queryFn: () => listFactorMatrices(20),
    refetchInterval: (query) => {
      const data = query.state.data as JobStatus[] | undefined
      return data?.some((j) => j.status === 'pending' || j.status === 'running') ? 3000 : false
    },
  })
  // F16:多因子诊断任务 running→终态 → 全局通知
  useJobTransitionNotify(historyJobs, 'factor-matrix')

  const poll = useQuery({
    queryKey: ['factor-matrix-job', jobId],
    queryFn: () => getFactorMatrixJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const s = q.state.data?.status
      return s === 'pending' || s === 'running' ? 2000 : false
    },
  })
  const job = poll.data
  // F6:queryFn 无副作用,直接消费 data
  const { data: report } = useQuery<FactorMatrixReport>({
    queryKey: ['factor-matrix-report', jobId],
    queryFn: () => getFactorMatrixReport(jobId!),
    enabled: !!jobId && job?.status === 'done',
  })

  const mut = useMutation({
    mutationFn: createFactorMatrix,
    onSuccess: (resp) => {
      setJobId(resp.job_id)
      message.success(`多因子诊断任务已创建: ${resp.job_id}`)
      qc.invalidateQueries({ queryKey: ['factor-matrix-job', resp.job_id] })
      qc.invalidateQueries({ queryKey: ['factor-matrix-jobs'] })
    },
    onError: (e) => message.error(errDetail(e)),
  })

  const addFactor = (name?: string) => {
    if (!name) return
    const info = factors.find((f) => f.name === name)
    if (indexMarket === 'CN' && info && isValuation(info)) {
      message.warning('A 股估值源(东财)暂不可达,该估值因子暂不可用')
      return
    }
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
      market: indexes.find((i) => i.key === v.index)?.market ?? null,
      start: v.range[0].toISOString().slice(0, 10),
      end: v.range[1].toISOString().slice(0, 10),
      ic_method: v.ic_method,
      periods: [1, 5, 10],
      orthogonalized: v.orthogonalized ?? false,
    })
  }

  // IC 摘要表(展平 period × factor)
  const icRows: Record<string, unknown>[] = []
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
  const fmbRows = report?.fmb
    ? Object.entries(report.fmb.lambdas).map(([name, l]) => ({ key: name, factor: name, ...l }))
    : []

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>多因子诊断</Typography.Title>

      <Card title="因子选择(2~8 个)">
        {factorsError && <QueryErrorAlert error={factorsError} retry={refetchFactors} />}
        <Space direction="vertical" style={{ width: '100%' }} size="small">
          {drafts.map((d) => {
            const info = factors.find((f) => f.name === d.factor)
            const cnBlocked = indexMarket === 'CN' && !!info && isValuation(info)
            return (
              <Card
                key={d.uid}
                size="small"
                title={<Space size={4}>{d.factor}{cnBlocked && <Tag color="warning">A股估值源不可用</Tag>}</Space>}
              >
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
                {info && info.params.length > 0 && (
                  <Card size="small" style={{ marginTop: 8 }} title={`${d.factor} 参数`}>
                    {info.params.map((p) => (
                      <ParamField
                        key={p.name}
                        p={p}
                        value={d.params[p.name]}
                        onSet={(val) =>
                          update(d.uid, { params: { ...d.params, [p.name]: val as number | string | boolean | null } })
                        }
                      />
                    ))}
                  </Card>
                )}
              </Card>
            )
          })}
          <Select
            placeholder="选择并添加因子"
            loading={factorsLoading}
            style={{ width: 300 }}
            onChange={(name) => addFactor(name as string)}
            options={factorOptions}
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
          initialValues={{ index: 'CSI300', ic_method: 'spearman', orthogonalized: false }}
        >
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
          <Form.Item name="range" label="区间" rules={[{ required: true, message: '请选择区间' }]}>
            <RangePicker />
          </Form.Item>
          <Form.Item name="ic_method" label="IC 相关方法">
            <Select options={[{ value: 'spearman', label: 'Spearman' }, { value: 'pearson', label: 'Pearson' }]} />
          </Form.Item>
          <Form.Item
            name="orthogonalized"
            label="正交化相关矩阵"
            valuePropName="checked"
            tooltip="用 Schmidt 正交化后的因子重算相关矩阵(按添加顺序,后序因子对前序取残差),验证正交化是否让因子间相关归零。IC 汇总与 FMB 仍用原始因子。"
          >
            <Switch />
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
          <Card
            title={<HelpTitle text="因子两两相关矩阵" tip={CORR_MATRIX_TIP} />}
            extra={
              <Button
                size="small"
                onClick={() => { navigator.clipboard.writeText(location.href); message.success('链接已复制') }}
              >复制链接</Button>
            }
          >
            <MatrixHeatmap matrix={report.correlation} height={460} />
          </Card>
          <Card title="各因子 IC 汇总">
            <Table
              size="small"
              dataSource={icRows}
              pagination={{ pageSize: 20 }}
              scroll={{ x: true }}
              columns={[
                { title: '期', dataIndex: 'period', key: 'period', width: 80 },
                { title: '因子', dataIndex: 'factor', key: 'factor' },
                {
                  title: <HelpTitle text="IC 均值" tip={METRIC_TIP['IC 均值']} />, dataIndex: 'ic_mean', key: 'ic_mean',
                  render: (v: number) => (v ?? 0).toFixed(4),
                },
                {
                  title: <HelpTitle text="ICIR" tip={METRIC_TIP['ICIR']} />, dataIndex: 'icir', key: 'icir',
                  render: (v: number) => (v ?? 0).toFixed(4),
                },
                {
                  title: <HelpTitle text="IC 正值占比" tip={METRIC_TIP['IC 正值占比']} />, dataIndex: 'ic_pos_ratio', key: 'ic_pos_ratio',
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
                  title: <HelpTitle text="换手率" tip={METRIC_TIP['换手']} />, dataIndex: 'turnover', key: 'turnover',
                  render: (v: number) => (v ?? 0).toFixed(3),
                },
              ]}
            />
          </Card>
          {report.fmb && (
            <Card
              title={<HelpTitle text={`Fama-MacBeth 因子收益 (n=${report.fmb.n_days})`} tip="Fama-MacBeth 逐日截面回归得到每个因子的风险溢价 λ 时序,再对其均值做 Newey-West t 检验,判断多因子风险溢价的统计显著性。" />}
            >
              <Table
                size="small"
                dataSource={fmbRows}
                pagination={false}
                columns={[
                  { title: '因子', dataIndex: 'factor', key: 'factor' },
                  {
                    title: <HelpTitle text="λ 均值" tip={METRIC_TIP['λ 均值']} />, dataIndex: 'lambda_mean', key: 'lambda_mean',
                    render: (v: number) => (v ?? 0).toFixed(6),
                  },
                  {
                    title: <HelpTitle text="λ t 值" tip={METRIC_TIP['λ t 值']} />, dataIndex: 'lambda_t', key: 'lambda_t',
                    render: (v: number) => (v ?? 0).toFixed(3),
                  },
                  {
                    title: 'λ p 值', dataIndex: 'lambda_pvalue', key: 'lambda_pvalue',
                    render: (v: number) => (v ?? 1).toFixed(3),
                  },
                  {
                    title: '正值占比', dataIndex: 'pos_ratio', key: 'pos_ratio',
                    render: (v: number) => `${((v ?? 0) * 100).toFixed(1)}%`,
                  },
                ]}
              />
            </Card>
          )}
        </Space>
      )}
      {jobId && !job && <Spin />}

      <Card title="历史多因子诊断任务">
        <JobHistoryTable
          jobs={(historyJobs || []) as JobStatus[]}
          onOpen={(id) => setJobId(id)}
        />
      </Card>
    </Space>
  )
}
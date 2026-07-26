import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button, Card, Input, Space, Table, Tag, Typography, message } from 'antd'
import ReactECharts from 'echarts-for-react'
import { listBacktests, getBacktestReport } from '@/api/client'
import type { BacktestReport, JobStatus, Metrics } from '@/types'

interface CompareRow {
  jobId: string
  report: BacktestReport
}

const fmtPct = (v: number, d = 2) => (v * 100).toFixed(d) + '%'
const fmtNum = (v: number, d = 2) => v.toFixed(d)

export default function ComparePage() {
  const [jobIds, setJobIds] = useState<string[]>([])
  const [inputId, setInputId] = useState('')

  const { data: jobs } = useQuery({
    queryKey: ['backtests-all'],
    queryFn: () => listBacktests(100),
  })

  // 并行获取多个报告
  const { data: reports, isLoading } = useQuery({
    queryKey: ['compare-reports', jobIds],
    queryFn: async () => {
      const results = await Promise.all(
        jobIds.map(async (id) => {
          try {
            const r = await getBacktestReport(id)
            return { jobId: id, report: r } as CompareRow
          } catch {
            return null
          }
        }),
      )
      return results.filter(Boolean) as CompareRow[]
    },
    enabled: jobIds.length > 0,
  })

  const addJob = () => {
    if (!inputId.trim()) return
    if (jobIds.includes(inputId.trim())) {
      message.warning('已添加该任务')
      return
    }
    setJobIds([...jobIds, inputId.trim()])
    setInputId('')
  }

  const removeJob = (id: string) => setJobIds(jobIds.filter((j) => j !== id))

  // 对比图表:多净值曲线叠加
  const chartOption = (() => {
    if (!reports || reports.length === 0) return {}
    const series = reports.map((r) => ({
      name: r.jobId,
      type: 'line',
      showSymbol: false,
      lineStyle: { width: 1.5 },
      data: r.report.equity_curve.values.map((v, i) => [r.report.equity_curve.index[i], v]),
    }))
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: reports.map((r) => r.jobId) },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'time' },
      yAxis: { type: 'value' },
      series,
    }
  })()

  const columns = [
    { title: '指标', dataIndex: 'metric', key: 'metric', fixed: 'left' as const, width: 120 },
    ...jobIds.map((id) => ({
      title: <span>{id} <a onClick={() => removeJob(id)} style={{ color: '#ff4d4f', fontSize: 11 }}>[移除]</a></span>,
      dataIndex: id, key: id, width: 120,
      render: (v: any) => v,
    })),
  ]

  const metricRows = ['total_return', 'annual_return', 'sharpe', 'max_drawdown', 'calmar', 'win_rate', 'n_trades', 'turnover']
  const dataSource = metricRows.map((m) => {
    const row: any = { key: m, metric: m }
    reports?.forEach((r) => {
      const v = (r.report.metrics as any)[m] as number
      row[r.jobId] = m.includes('return') || m === 'max_drawdown' || m === 'win_rate' || m === 'turnover'
        ? fmtPct(v) : m === 'n_trades' ? String(v) : fmtNum(v)
    })
    return row
  })

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="添加对比任务">
        <Space>
          <Input placeholder="输入任务 ID" value={inputId} onChange={(e) => setInputId(e.target.value)} style={{ width: 240 }} />
          <Button type="primary" onClick={addJob}>添加</Button>
        </Space>
        {jobs && (
          <div style={{ marginTop: 12 }}>
            <Typography.Text type="secondary">已完成任务(点击添加):</Typography.Text>
            <div style={{ marginTop: 8 }}>
              {(jobs as JobStatus[]).filter((j) => j.status === 'done').slice(0, 10).map((j) => (
                <Tag key={j.job_id} style={{ cursor: 'pointer', marginBottom: 4 }}
                  onClick={() => setJobIds([...jobIds, j.job_id].filter((v, i, a) => a.indexOf(v) === i))}>
                  {j.job_id}
                </Tag>
              ))}
            </div>
          </div>
        )}
      </Card>

      {jobIds.length > 0 && isLoading && <Card loading>加载报告中...</Card>}

      {reports && reports.length > 0 && (
        <>
          <Card title="净值曲线对比">
            <ReactECharts option={chartOption} style={{ height: 400 }} />
          </Card>
          <Card title="指标对比表">
            <Table columns={columns} dataSource={dataSource} size="small" pagination={false} scroll={{ x: 'max-content' }} />
          </Card>
        </>
      )}

      {jobIds.length === 0 && (
        <Card><Typography.Text type="secondary">添加 2 个或更多已完成任务以对比</Typography.Text></Card>
      )}
    </Space>
  )
}
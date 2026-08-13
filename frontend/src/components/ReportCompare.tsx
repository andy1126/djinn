import { useQuery } from '@tanstack/react-query'
import { Card, Empty, Table, Typography } from 'antd'
import ReactECharts from 'echarts-for-react'
import { getBacktestReport } from '@/api/client'
import type { BacktestReport } from '@/types'

interface CompareRow {
  jobId: string
  report: BacktestReport
}

interface Props {
  jobIds: string[]
}

const fmtPct = (v: number, d = 2) => (v * 100).toFixed(d) + '%'
const fmtNum = (v: number, d = 2) => v.toFixed(d)

export default function ReportCompare({ jobIds }: Props) {
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
      title: id,
      dataIndex: id,
      key: id,
      width: 120,
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

  if (jobIds.length === 0) {
    return (
      <Empty description="请在任务列表中勾选 2 个或更多已完成任务进行对比" style={{ padding: 48 }} />
    )
  }

  if (isLoading) {
    return <Card loading style={{ minHeight: 200 }} />
  }

  if (!reports || reports.length === 0) {
    return <Typography.Text type="secondary">加载报告失败,请确认所选任务已完成。</Typography.Text>
  }

  return (
    <>
      <Card title="净值曲线对比" style={{ marginBottom: 16 }}>
        <ReactECharts option={chartOption} style={{ height: 400 }} />
      </Card>
      <Card title="指标对比表">
        <Table columns={columns} dataSource={dataSource} size="small" pagination={false} scroll={{ x: 'max-content' }} />
      </Card>
    </>
  )
}

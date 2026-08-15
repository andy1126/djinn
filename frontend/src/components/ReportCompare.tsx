import { useQuery } from '@tanstack/react-query'
import { Card, Empty, Table, Typography } from 'antd'
import ReactECharts from 'echarts-for-react'
import { getBacktestReport } from '@/api/client'
import type { BacktestReport } from '@/types'
import { fmtNum, fmtPct } from '@/utils/format'

interface CompareRow {
  jobId: string
  report: BacktestReport
}

interface Props {
  jobIds: string[]
}

// F15:legend/表头用可读短码(前 6 位),完整 id 放 tooltip
const shortId = (id: string) => (id.length > 6 ? id.slice(0, 6) : id)

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
    const series = reports.map((r) => {
      const vals = r.report.equity_curve.values
      const base = vals[0] || 1 // F15:起点归一化(各序列起点对齐到 1.0)
      return {
        name: shortId(r.jobId),
        type: 'line',
        showSymbol: false,
        lineStyle: { width: 1.5 },
        data: vals.map((v, i) => [r.report.equity_curve.index[i], v / base]),
      }
    })
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: reports.map((r) => shortId(r.jobId)) },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'time' },
      yAxis: { type: 'value' },
      series,
    }
  })()

  const metricRows = ['total_return', 'annual_return', 'sharpe', 'max_drawdown', 'calmar', 'win_rate', 'n_trades', 'turnover']
  // F15:越大越好的指标(含 max_drawdown,存为 ≤0 负值,越接近 0 越好);
  // n_trades/turnover 无方向,不高亮。
  const METRIC_BETTER_HIGHER = new Set(['total_return', 'annual_return', 'sharpe', 'calmar', 'win_rate', 'max_drawdown'])
  const bestByMetric: Record<string, string> = {}
  for (const m of metricRows) {
    if (!METRIC_BETTER_HIGHER.has(m)) continue
    let bestId = ''
    let bestVal = -Infinity
    reports?.forEach((r) => {
      const v = (r.report.metrics as Record<string, number>)[m]
      if (v != null && !Number.isNaN(v) && v > bestVal) { bestVal = v; bestId = r.jobId }
    })
    if (bestId) bestByMetric[m] = bestId
  }

  const columns = [
    { title: '指标', dataIndex: 'metric', key: 'metric', fixed: 'left' as const, width: 120 },
    ...jobIds.map((id) => ({
      title: shortId(id),
      dataIndex: id,
      key: id,
      width: 120,
      render: (v: string, row: Record<string, string>) => ({
        children: v,
        props: bestByMetric[row.metric] === id ? { style: { background: '#f6ffed', fontWeight: 600 } } : {},
      }),
    })),
  ]

  const dataSource = metricRows.map((m) => {
    const row: Record<string, string> = { key: m, metric: m }
    reports?.forEach((r) => {
      const v = (r.report.metrics as Record<string, number>)[m]
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
        <ReactECharts option={chartOption} notMerge style={{ height: 400 }} />
      </Card>
      <Card title="指标对比表">
        <Table columns={columns} dataSource={dataSource} size="small" pagination={false} scroll={{ x: 'max-content' }} />
      </Card>
    </>
  )
}

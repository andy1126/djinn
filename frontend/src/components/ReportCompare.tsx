import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Alert, Card, Empty, Space, Switch, Table, Typography } from 'antd'
import ReactECharts from 'echarts-for-react'
import { getBacktestReport } from '@/api/client'
import type { BacktestReport } from '@/types'
import { fmtNum, fmtPct } from '@/utils/format'
import { METRIC_BETTER_HIGHER } from '@/utils/metricDirections'

interface CompareRow {
  jobId: string
  report: BacktestReport
}

interface Props {
  jobIds: string[]
  /** F15:job_id → 可读标题(策略名 + 参数),缺省退化为短码。 */
  titles?: Record<string, string>
}

// F15:列名优先用任务 title 元数据,缺省用短码(前 6 位)
const shortId = (id: string) => (id.length > 6 ? id.slice(0, 6) : id)

export default function ReportCompare({ jobIds, titles }: Props) {
  const label = (id: string) => titles?.[id] ?? shortId(id)
  const { data, isLoading } = useQuery({
    queryKey: ['compare-reports', jobIds],
    queryFn: async () => {
      // F7:allSettled —— 单份报告失败不影响其余,失败列单独标注
      const settled = await Promise.allSettled(
        jobIds.map(async (id) => ({ jobId: id, report: await getBacktestReport(id) })),
      )
      const rows: CompareRow[] = []
      const failed: string[] = []
      settled.forEach((r, i) => {
        if (r.status === 'fulfilled') rows.push(r.value)
        else failed.push(jobIds[i])
      })
      return { rows, failed }
    },
    enabled: jobIds.length > 0,
  })
  const reports = data?.rows
  const failedIds = data?.failed ?? []

  const chartOption = (() => {
    if (!reports || reports.length === 0) return {}
    const series = reports.map((r) => {
      const vals = r.report.equity_curve.values
      const base = vals[0] || 1 // F15:起点归一化(各序列起点对齐到 1.0)
      return {
        name: label(r.jobId),
        type: 'line',
        showSymbol: false,
        lineStyle: { width: 1.5 },
        data: vals.map((v, i) => [r.report.equity_curve.index[i], v / base]),
      }
    })
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: reports.map((r) => label(r.jobId)), top: 0 },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'time' },
      yAxis: { type: 'value' },
      series,
    }
  })()

  const metricRows = ['total_return', 'annual_return', 'sharpe', 'sortino', 'calmar', 'max_drawdown', 'win_rate', 'n_trades', 'turnover']
  // F15:越大越好的指标(方向表见 utils/metricDirections,与后端 REVERSE_MIN_TARGETS 同步);
  // n_trades/turnover 无方向,不高亮。
  const [showDiff, setShowDiff] = useState(false)
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
      title: label(id),
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
  // F15:差异行 —— 各列 vs 首列的差值(仅数值类指标)
  if (showDiff && reports && reports.length >= 2) {
    const first = reports[0]
    for (const m of metricRows) {
      const base = (first.report.metrics as Record<string, number>)[m]
      if (base == null || Number.isNaN(base)) continue
      const row: Record<string, string> = { key: `diff-${m}`, metric: `${m}(Δ)` }
      row[first.jobId] = '—'
      reports.slice(1).forEach((r) => {
        const v = (r.report.metrics as Record<string, number>)[m]
        const d = v == null || Number.isNaN(v) ? Number.NaN : v - base
        row[r.jobId] = Number.isNaN(d) ? '—' : fmtNum(d)
      })
      dataSource.push(row)
    }
  }

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
      {failedIds.length > 0 && (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          message={`${failedIds.length} 份报告加载失败(已跳过)`}
          description={failedIds.map((id) => label(id)).join('、')}
        />
      )}
      <Card title="净值曲线对比" style={{ marginBottom: 16 }}>
        <ReactECharts option={chartOption} notMerge style={{ height: 400 }} />
      </Card>
      <Card
        title="指标对比表"
        extra={
          <Space size={4}>
            <Switch size="small" checked={showDiff} onChange={setShowDiff} />
            <Typography.Text type="secondary">差异行</Typography.Text>
          </Space>
        }
      >
        <Table columns={columns} dataSource={dataSource} size="small" pagination={false} scroll={{ x: 'max-content' }} />
      </Card>
    </>
  )
}

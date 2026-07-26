import ReactECharts from 'echarts-for-react'
import type { DataFrameData } from '@/types'

interface Props {
  weights: DataFrameData
  height?: number
}

export default function PositionAreaChart({ weights, height = 280 }: Props) {
  if (!weights.data.length) {
    return <div style={{ height, padding: 40, textAlign: 'center', color: '#999' }}>无持仓数据</div>
  }
  const symbols = weights.columns
  const series = symbols.map((sym, si) => ({
    name: sym,
    type: 'line',
    stack: '仓位',
    areaStyle: {},
    data: weights.data.map((row, di) => [weights.index[di], row[si] ?? 0]),
    showSymbol: false,
  }))

  const option = {
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => (v * 100).toFixed(2) + '%' },
    legend: { data: symbols, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', top: 40, containLabel: true },
    xAxis: { type: 'time' },
    yAxis: { type: 'value', max: 1, axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' } },
    series,
  }
  return <ReactECharts option={option} style={{ height }} />
}
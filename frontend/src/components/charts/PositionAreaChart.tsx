import ReactECharts from 'echarts-for-react'
import { useChartTheme } from '@/hooks/useChartTheme'
import type { DataFrameData } from '@/types'

interface Props {
  weights: DataFrameData
  height?: number
}

export default function PositionAreaChart({ weights, height = 280 }: Props) {
  const theme = useChartTheme() // F18:暗色主题
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
    ...theme,
    tooltip: { ...theme.tooltip, trigger: 'axis', valueFormatter: (v: number) => (v * 100).toFixed(2) + '%' },
    legend: { ...theme.legend, data: symbols, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', top: 40, containLabel: true },
    xAxis: { ...theme.xAxis, type: 'time' },
    yAxis: { ...theme.yAxis, type: 'value', max: 1, axisLabel: { ...theme.yAxis.axisLabel, formatter: (v: number) => (v * 100).toFixed(0) + '%' } },
    series,
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
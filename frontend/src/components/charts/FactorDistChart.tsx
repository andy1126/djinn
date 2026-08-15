import ReactECharts from 'echarts-for-react'
import { useChartTheme } from '@/hooks/useChartTheme'
import type { DataFrameData } from '@/types'

interface Props {
  exposures: DataFrameData
  height?: number
}

/** 因子暴露时序折线图(date × factor)。 */
export default function FactorDistChart({ exposures, height = 320 }: Props) {
  const theme = useChartTheme() // F18:暗色主题
  const series = exposures.columns.map((c, idx) => ({
    name: c,
    type: 'line',
    data: exposures.data.map((row, i) => [exposures.index[i], (row[idx] as number) ?? 0]),
    showSymbol: false,
    lineStyle: { width: 2 },
  }))
  const option = {
    ...theme,
    tooltip: { ...theme.tooltip, trigger: 'axis' },
    legend: { data: exposures.columns, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { ...theme.xAxis, type: 'time' },
    yAxis: { ...theme.yAxis, type: 'value' },
    dataZoom: [{ type: 'inside', start: 0, end: 100 }],
    series,
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
import ReactECharts from 'echarts-for-react'
import { useChartTheme } from '@/hooks/useChartTheme'
import type { SeriesData } from '@/types'

interface Props {
  equity: SeriesData
  benchmark?: SeriesData | null
  logScale?: boolean
  height?: number
}

interface LineSeries {
  name: string
  type: 'line'
  data: (string | number)[][]
  showSymbol: boolean
  smooth?: boolean
  lineStyle: { width: number; type?: string }
}

export default function EquityCurveChart({ equity, benchmark, logScale, height = 360 }: Props) {
  const theme = useChartTheme()
  const series: LineSeries[] = [
    {
      name: '策略净值',
      type: 'line',
      data: equity.values.map((v, i) => [equity.index[i], v]),
      showSymbol: false,
      smooth: false,
      lineStyle: { width: 2 },
    },
  ]
  if (benchmark && benchmark.values.length > 0) {
    series.push({
      name: '基准',
      type: 'line',
      data: benchmark.values.map((v, i) => [benchmark.index[i], v]),
      showSymbol: false,
      lineStyle: { width: 1.5, type: 'dashed' },
    })
  }

  const option = {
    ...theme,
    tooltip: { trigger: 'axis' },
    legend: { data: ['策略净值', '基准'], textStyle: theme.legend.textStyle },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'time', ...theme.xAxis },
    yAxis: {
      type: logScale ? 'log' : 'value',
      ...theme.yAxis,
      axisLabel: { ...theme.yAxis.axisLabel, formatter: (v: number) => v.toFixed(2) },
    },
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', start: 0, end: 100, height: 20 },
    ],
    series,
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
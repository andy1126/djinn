import ReactECharts from 'echarts-for-react'
import { useChartTheme } from '@/hooks/useChartTheme'
import type { DataFrameData } from '@/types'

interface Props {
  quantileReturns: DataFrameData
  height?: number
}

/** 分层累计收益曲线(评估因子单调性:顶组 vs 底组是否持续分化)。 */
export default function QuantileCurveChart({ quantileReturns, height = 340 }: Props) {
  const theme = useChartTheme() // F18:暗色主题
  const series = quantileReturns.columns.map((_c, idx) => ({
    name: `Q${idx + 1}`,
    type: 'line',
    data: quantileReturns.data.map((row, i) => [
      quantileReturns.index[i],
      (row[idx] as number) ?? 0,
    ]),
    showSymbol: false,
    smooth: true,
    lineStyle: { width: 2 },
    emphasis: { focus: 'series' },
  }))
  const option = {
    ...theme,
    tooltip: { ...theme.tooltip, trigger: 'axis' },
    legend: { data: quantileReturns.columns.map((_, idx) => `Q${idx + 1}`), top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { ...theme.xAxis, type: 'category', data: quantileReturns.index },
    yAxis: {
      ...theme.yAxis,
      type: 'value',
      axisLabel: { ...theme.yAxis.axisLabel, formatter: (v: number) => (v * 100).toFixed(1) + '%' },
    },
    series,
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
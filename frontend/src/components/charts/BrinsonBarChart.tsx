import ReactECharts from 'echarts-for-react'
import { useChartTheme } from '@/hooks/useChartTheme'
import type { BrinsonResult } from '@/types'

interface Props {
  brinson: BrinsonResult
  height?: number
}

/** Brinson 三效应按行业可视化(配置 / 选股 / 交互堆叠柱)。 */
export default function BrinsonBarChart({ brinson, height = 340 }: Props) {
  const theme = useChartTheme() // F18:暗色主题
  const ind = brinson.allocation.index
  const series = ['allocation', 'selection', 'interaction'].map((key) => {
    const s = (brinson as unknown as Record<string, { values: number[] }>)[key]
    return {
      name: key === 'allocation' ? '配置' : key === 'selection' ? '选股' : '交互',
      type: 'bar',
      stack: 'effect',
      data: s.values.map((v: number, i: number) => [ind[i], v ?? 0]),
    }
  })
  const option = {
    ...theme,
    tooltip: { ...theme.tooltip, trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { ...theme.legend, data: series.map((s) => s.name) },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: {
      ...theme.xAxis,
      type: 'category',
      data: ind,
      axisLabel: { ...theme.xAxis.axisLabel, rotate: 30 },
    },
    yAxis: { ...theme.yAxis, type: 'value' },
    series,
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
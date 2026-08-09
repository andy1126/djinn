import ReactECharts from 'echarts-for-react'
import type { BrinsonResult } from '@/types'

interface Props {
  brinson: BrinsonResult
  height?: number
}

/** Brinson 三效应按行业可视化(配置 / 选股 / 交互堆叠柱)。 */
export default function BrinsonBarChart({ brinson, height = 340 }: Props) {
  const ind = brinson.allocation.index
  const series = ['allocation', 'selection', 'interaction'].map((key, _idx) => {
    const s = (brinson as any)[key]
    return {
      name: key === 'allocation' ? '配置' : key === 'selection' ? '选股' : '交互',
      type: 'bar',
      stack: 'effect',
      data: s.values.map((v: number, i: number) => [ind[i], v ?? 0]),
    }
  })
  const option = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { data: series.map((s) => s.name) },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: {
      type: 'category',
      data: ind,
      axisLabel: { rotate: 30 },
    },
    yAxis: { type: 'value' },
    series,
  }
  return <ReactECharts option={option} style={{ height }} />
}
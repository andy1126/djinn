import ReactECharts from 'echarts-for-react'
import type { SeriesData } from '@/types'

interface Props {
  drawdown: SeriesData
  height?: number
}

export default function DrawdownChart({ drawdown, height = 240 }: Props) {
  const option = {
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => (v * 100).toFixed(2) + '%' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'time' },
    yAxis: {
      type: 'value',
      axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' },
    },
    series: [
      {
        name: '回撤',
        type: 'line',
        data: drawdown.values.map((v, i) => [drawdown.index[i], v]),
        showSymbol: false,
        areaStyle: { color: 'rgba(255,77,79,0.3)' },
        lineStyle: { color: '#ff4d4f' },
      },
    ],
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
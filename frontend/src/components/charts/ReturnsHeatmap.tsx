import ReactECharts from 'echarts-for-react'
import { useChartTheme } from '@/hooks/useChartTheme'
import type { DataFrameData } from '@/types'

interface Props {
  monthly: DataFrameData
  height?: number
}

export default function ReturnsHeatmap({ monthly, height = 320 }: Props) {
  const theme = useChartTheme() // F18:暗色主题
  if (!monthly.data.length) {
    return <div style={{ height, padding: 40, textAlign: 'center', color: '#999' }}>无月度收益数据</div>
  }
  // monthly_returns DataFrame: index=年, columns=月(1-12)
  const years = monthly.index
  const months = monthly.columns
  const data: [number, number, number][] = []
  let min = 0, max = 0
  years.forEach((_y, yi) => {
    months.forEach((_m, mi) => {
      const r = monthly.data[yi]?.[mi]
      const v = typeof r === 'number' ? r : 0
      if (v < min) min = v
      if (v > max) max = v
      data.push([mi, yi, v])
    })
  })

  const option = {
    ...theme,
    tooltip: {
      ...theme.tooltip,
      position: 'top',
      valueFormatter: (v: number) => (v * 100).toFixed(2) + '%',
    },
    grid: { left: '3%', right: '4%', bottom: '5%', containLabel: true },
    xAxis: { ...theme.xAxis, type: 'category', data: months, splitArea: { show: true } },
    yAxis: { ...theme.yAxis, type: 'category', data: years, splitArea: { show: true } },
    visualMap: {
      min: min,
      max: max,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '0%',
      inRange: { color: ['#52c41a', '#f0f0f0', '#ff4d4f'] },
      formatter: (v: number) => (v * 100).toFixed(1) + '%',
    },
    series: [
      {
        name: '月度收益',
        type: 'heatmap',
        data,
        label: {
          show: true,
          formatter: (p: { value: number[] }) => (p.value[2] * 100).toFixed(1) + '%',
        },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } },
      },
    ],
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
import ReactECharts from 'echarts-for-react'
import { useChartTheme } from '@/hooks/useChartTheme'
import type { SeriesData } from '@/types'

interface Props {
  ic: SeriesData
  height?: number
}

/** IC 时序柱状图(正负分别着色,辅助判断因子稳定性)。 */
export default function ICBarChart({ ic, height = 320 }: Props) {
  const theme = useChartTheme() // F18:暗色主题
  const data = ic.values.map((v, i) => [ic.index[i], v ?? 0])
  const option = {
    ...theme,
    tooltip: { ...theme.tooltip, trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { ...theme.xAxis, type: 'time' },
    yAxis: { ...theme.yAxis, type: 'value', axisLabel: { ...theme.yAxis.axisLabel, formatter: (v: number) => v.toFixed(2) } },
    dataZoom: [{ type: 'inside', start: 0, end: 100 }],
    series: [
      {
        name: 'IC',
        type: 'bar',
        data,
        itemStyle: {
          color: (p: { value: [string, number] }) => (p.value[1] >= 0 ? '#52c41a' : '#f5222d'),
        },
      },
    ],
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
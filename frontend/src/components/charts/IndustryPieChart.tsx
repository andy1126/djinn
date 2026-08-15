import ReactECharts from 'echarts-for-react'
import { useChartTheme } from '@/hooks/useChartTheme'
import type { DataFrameData } from '@/types'

interface Props {
  industryDistribution: DataFrameData
  height?: number
}

/** 持仓行业占比饼图(取最末一交易日截面)。 */
export default function IndustryPieChart({ industryDistribution, height = 320 }: Props) {
  const theme = useChartTheme() // F18:暗色主题
  if (!industryDistribution.index.length) {
    return <div style={{ height, lineHeight: `${height}px`, textAlign: 'center', color: '#999' }}>无行业分布数据</div>
  }
  const lastIdx = industryDistribution.data.length - 1
  const data = industryDistribution.columns
    .map((name, ci) => ({ name, value: (industryDistribution.data[lastIdx][ci] as number) ?? 0 }))
    .filter((d) => d.value > 1e-9)
  const option = {
    ...theme,
    tooltip: { ...theme.tooltip, trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { orient: 'vertical', right: 0, top: 'center' },
    series: [
      {
        name: '行业占比',
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['40%', '50%'],
        data,
        label: { formatter: '{b}: {d}%' },
      },
    ],
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
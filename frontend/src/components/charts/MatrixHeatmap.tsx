import ReactECharts from 'echarts-for-react'
import type { DataFrameData } from '@/types'

interface Props {
  matrix: DataFrameData
  height?: number
  /** 数值格式化(默认两位小数)。 */
  formatter?: (v: number) => string
  /** 配色区:默认 [-1,1] 红(正)→ 白(0)→ 绿(负)对称。 */
  symmetric?: boolean
}

/**
 * 泛用 px×px 矩阵热力图(因子相关 / 任意方阵)。
 * matrix.index 为 y 轴、matrix.columns 为 x 轴、matrix.data[row][col] 为值。
 */
export default function MatrixHeatmap({
  matrix,
  height = 420,
  formatter,
  symmetric = true,
}: Props) {
  if (!matrix.data.length) {
    return (
      <div style={{ height, padding: 40, textAlign: 'center', color: '#999' }}>
        无矩阵数据
      </div>
    )
  }
  const rows = matrix.index
  const cols = matrix.columns
  const data: [number, number, number][] = []
  let max = 0
  rows.forEach((r, ri) => {
    cols.forEach((c, ci) => {
      const v = matrix.data[ri]?.[ci]
      const n = typeof v === 'number' ? v : 0
      if (Math.abs(n) > max) max = Math.abs(n)
      data.push([ci, ri, n])
    })
  })
  // 对称配色上下界
  const bound = symmetric ? (max || 1) : max

  const fmt = formatter ?? ((v: number) => v.toFixed(2))

  const option = {
    tooltip: {
      position: 'top',
      formatter: (p: any) => {
        const v = p.value[2]
        return `${rows[p.value[1]]} × ${cols[p.value[0]]}: ${fmt(v)}`
      },
    },
    grid: { left: '3%', right: '4%', bottom: '12%', containLabel: true },
    xAxis: {
      type: 'category',
      data: cols,
      splitArea: { show: true },
      axisLabel: { rotate: 30 },
    },
    yAxis: {
      type: 'category',
      data: rows,
      splitArea: { show: true },
    },
    visualMap: {
      min: -bound,
      max: bound,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '0%',
      inRange: { color: ['#52c41a', '#f0f0f0', '#f5222d'] },
      formatter: (v: number) => fmt(v),
    },
    series: [
      {
        name: '相关',
        type: 'heatmap',
        data,
        label: {
          show: true,
          formatter: (p: any) => fmt(p.value[2]),
        },
        emphasis: {
          itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' },
        },
      },
    ],
  }
  return <ReactECharts option={option} notMerge style={{ height }} />
}
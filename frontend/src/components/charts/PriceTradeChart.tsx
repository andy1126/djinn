import ReactECharts from 'echarts-for-react'
import { useChartTheme } from '@/hooks/useChartTheme'
import type { DataFrameData, TradeRecord } from '@/types'

interface Props {
  prices: DataFrameData
  trades: TradeRecord[]
  height?: number
}

/** 股票收盘价走势曲线 + 买卖点标记(绿三角=买,红倒三角=卖)。 */
export default function PriceTradeChart({ prices, trades, height = 420 }: Props) {
  const theme = useChartTheme() // F18:暗色主题
  const dates = prices?.index || []
  const symbols = prices?.columns || []
  const data = prices?.data || []

  // 每个标的一条收盘价线
  const priceSeries = symbols.map((sym, colIdx) => ({
    name: sym,
    type: 'line' as const,
    showSymbol: false,
    lineStyle: { width: 1.5 },
    data: dates.map((d, i) => [d, (data[i]?.[colIdx] as number | null) ?? null]),
  }))

  const buys = trades.filter((t) => t.side === 'buy')
  const sells = trades.filter((t) => t.side === 'sell')

  const buySeries = {
    name: '买入',
    type: 'scatter' as const,
    symbol: 'triangle',
    symbolSize: 12,
    itemStyle: { color: '#52c41a' },
    data: buys.map((t) => [t.timestamp as string, t.price as number]),
  }
  const sellSeries = {
    name: '卖出',
    type: 'scatter' as const,
    symbol: 'triangle',
    symbolRotate: 180,
    symbolSize: 12,
    itemStyle: { color: '#f5222d' },
    data: sells.map((t) => [t.timestamp as string, t.price as number]),
  }

  const option = {
    ...theme,
    tooltip: { ...theme.tooltip, trigger: 'axis' },
    legend: { data: [...symbols, '买入', '卖出'], top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { ...theme.xAxis, type: 'time' },
    yAxis: { ...theme.yAxis, type: 'value', scale: true },
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', start: 0, end: 100, height: 20 },
    ],
    series: [...priceSeries, buySeries, sellSeries],
  }

  return <ReactECharts option={option} notMerge style={{ height }} />
}

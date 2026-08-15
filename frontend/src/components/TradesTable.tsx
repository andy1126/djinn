import { Table, Tag } from 'antd'
import type { TradeRecord } from '@/types'

interface Props {
  trades: TradeRecord[]
}

export default function TradesTable({ trades }: Props) {
  if (!trades.length) {
    return <div style={{ padding: 24, textAlign: 'center', color: '#999' }}>无交易记录</div>
  }
  // 动态提取列(trades 中每个记录的 union keys)
  const keys = Array.from(new Set(trades.flatMap((t) => Object.keys(t))))
  const columns = keys.map((k) => ({
    title: k,
    dataIndex: k,
    key: k,
    render: (v: unknown) => {
      if (typeof v === 'number') return v.toFixed(4).replace(/\.?0+$/, '') || v
      if (k === 'side' || k.toLowerCase().includes('side')) {
        const isBuy = String(v).toLowerCase().includes('buy')
        return <Tag color={isBuy ? 'green' : 'red'}>{String(v)}</Tag>
      }
      return String(v)
    },
  }))
  return (
    <Table
      columns={columns}
      dataSource={trades.map((t, i) => ({ key: i, ...t }))}
      size="small"
      pagination={{ pageSize: 20 }}
      scroll={{ x: 'max-content' }}
    />
  )
}
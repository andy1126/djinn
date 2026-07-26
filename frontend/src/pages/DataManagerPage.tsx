import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Button, Card, DatePicker, Form, Input, Select, Space, Table, message } from 'antd'
import dayjs from 'dayjs'
import { fetchData, listCache, clearCache } from '@/api/client'
import type { CacheEntry } from '@/types'

const { RangePicker } = DatePicker

export default function DataManagerPage() {
  const qc = useQueryClient()
  const [form] = Form.useForm()
  const [symbols, setSymbols] = useState<string[]>(['NVDA'])

  const { data: cache } = useQuery({
    queryKey: ['cache'],
    queryFn: listCache,
  })

  const fetchMut = useMutation({
    mutationFn: fetchData,
    onSuccess: () => {
      message.success('拉取成功')
      qc.invalidateQueries({ queryKey: ['cache'] })
    },
    onError: (e: any) => message.error(e?.response?.data?.detail || String(e)),
  })

  const clearMut = useMutation({
    mutationFn: clearCache,
    onSuccess: () => {
      message.success('缓存已清空')
      qc.invalidateQueries({ queryKey: ['cache'] })
    },
  })

  const onSubmit = (v: any) => {
    const [start, end] = v.range || []
    fetchMut.mutate({
      symbols: v.symbols.split(',').map((s: string) => s.trim()).filter(Boolean),
      market: v.market,
      start: start.format('YYYY-MM-DD'),
      end: end.format('YYYY-MM-DD'),
      adjust: v.adjust,
    })
  }

  const cacheColumns = [
    { title: '文件', dataIndex: 'file', key: 'file' },
    { title: '行数', dataIndex: 'rows', key: 'rows' },
    { title: '起始', dataIndex: 'start', key: 'start' },
    { title: '结束', dataIndex: 'end', key: 'end' },
  ]

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="拉取数据">
        <Form
          form={form}
          layout="inline"
          onFinish={onSubmit}
          initialValues={{
            symbols: 'NVDA',
            market: 'US',
            adjust: 'backward',
            range: [dayjs('2024-01-01'), dayjs('2024-12-31')],
          }}
        >
          <Form.Item name="symbols" label="标的(逗号分隔)">
            <Input placeholder="NVDA,AAPL" style={{ width: 220 }} />
          </Form.Item>
          <Form.Item name="market" label="市场">
            <Select style={{ width: 100 }} options={[
              { label: '美股', value: 'US' },
              { label: 'A股', value: 'CN' },
              { label: '港股', value: 'HK' },
            ]} />
          </Form.Item>
          <Form.Item name="range" label="区间">
            <RangePicker />
          </Form.Item>
          <Form.Item name="adjust" label="复权">
            <Select style={{ width: 110 }} options={[
              { label: '后复权', value: 'backward' },
              { label: '前复权', value: 'forward' },
              { label: '不复权', value: 'none' },
            ]} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={fetchMut.isPending}>拉取</Button>
          </Form.Item>
        </Form>
      </Card>

      <Card
        title="缓存状态"
        extra={<Button danger onClick={() => clearMut.mutate()} loading={clearMut.isPending}>清空缓存</Button>}
      >
        <Table
          columns={cacheColumns}
          dataSource={(cache?.entries || []) as CacheEntry[]}
          rowKey="file"
          size="small"
          pagination={false}
        />
      </Card>
    </Space>
  )
}
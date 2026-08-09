import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, Input, Segmented, Space, Spin, Table, Tag, Typography } from 'antd'
import { getStockList, getIndexComponents, getIndustries, listIndexes } from '@/api/client'
import type { IndexInfo, UniverseStock } from '@/types'

/**
 * 股票池浏览:全市场列表 / 宽基指数成分 / 行业分布三个视图。
 */
export default function UniversePage() {
  const [mode, setMode] = useState<'stock' | 'index' | 'industry'>('stock')
  const [market, setMarket] = useState<string>('CN')
  const [index, setIndex] = useState<string>('CSI300')

  const { data: indexes } = useQuery({ queryKey: ['indexes'], queryFn: listIndexes })

  const stockList = useQuery({
    queryKey: ['stock-list', market],
    queryFn: () => getStockList(market),
    enabled: mode === 'stock',
  })

  const indexComps = useQuery({
    queryKey: ['index-components', index],
    queryFn: () => getIndexComponents(index),
    enabled: mode === 'index',
  })

  const industries = useQuery({
    queryKey: ['industries', index],
    queryFn: () => getIndustries(index),
    enabled: mode === 'industry',
  })

  const indexNames = (indexes?.indexes || []).reduce(
    (acc, i: IndexInfo) => ({ ...acc, [i.key]: i.name }),
    {} as Record<string, string>,
  )

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>股票池</Typography.Title>
      <Segmented
        value={mode}
        onChange={(v) => setMode(v as typeof mode)}
        options={[
          { label: '全市场列表', value: 'stock' },
          { label: '指数成分', value: 'index' },
          { label: '行业分布', value: 'industry' },
        ]}
      />

      {mode === 'stock' && (
        <Card
          title="全市场股票列表"
          extra={
            <Segmented
              value={market}
              onChange={(v) => setMarket(String(v))}
              options={['CN', 'HK', 'US']}
            />
          }
        >
          {stockList.isLoading ? (
            <Spin />
          ) : stockList.isError ? (
            <Typography.Text type="danger">股票列表需 akshare,目前不可用(A 股支持)。</Typography.Text>
          ) : (
            <Table<UniverseStock>
              size="small"
              rowKey="symbol"
              dataSource={stockList.data?.stocks || []}
              pagination={{ pageSize: 20 }}
              columns={[
                { title: '代码', dataIndex: 'symbol', key: 'symbol' },
                { title: '名称', dataIndex: 'name', key: 'name' },
                { title: '市场', dataIndex: 'market', key: 'market', render: (m: string) => <Tag>{m}</Tag> },
              ]}
            />
          )}
        </Card>
      )}

      {mode === 'index' && (
        <Card
          title="指数成分股"
          extra={
            <Input
              style={{ width: 200 }}
              value={index}
              onChange={(e) => setIndex(e.target.value.toUpperCase())}
              placeholder="如 CSI300"
            />
          }
        >
          {indexComps.isLoading ? (
            <Spin />
          ) : indexComps.isError ? (
            <Typography.Text type="danger">无法解析指数成分(确认指数键与 provider 支持)。</Typography.Text>
          ) : (
            <Space direction="vertical" style={{ width: '100%' }}>
              <Typography.Text>
                {indexNames[index] || index} · {indexComps.data?.count ?? 0} 只
              </Typography.Text>
              <Table
                size="small"
                rowKey={(_, i) => String(i)}
                dataSource={(indexComps.data?.symbols || []).map((s, i) => ({ symbol: s, key: i }))}
                pagination={{ pageSize: 50 }}
                columns={[{ title: '代码', dataIndex: 'symbol', key: 'symbol' }]}
              />
            </Space>
          )}
        </Card>
      )}

      {mode === 'industry' && (
        <Card
          title="行业分布"
          extra={
            <Input
              style={{ width: 200 }}
              value={index}
              onChange={(e) => setIndex(e.target.value.toUpperCase())}
              placeholder="如 CSI300"
            />
          }
        >
          {industries.isLoading ? (
            <Spin />
          ) : industries.isError ? (
            <Typography.Text type="danger">行业映射需 akshare,目前不可用。</Typography.Text>
          ) : (
            <Table
              size="small"
              rowKey="name"
              dataSource={industries.data?.industries || []}
              pagination={false}
              columns={[
                { title: '行业', dataIndex: 'name', key: 'name' },
                { title: '股票数', dataIndex: 'count', key: 'count' },
              ]}
            />
          )}
        </Card>
      )}
    </Space>
  )
}
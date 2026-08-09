import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, Select, Segmented, Space, Spin, Table, Tag, Typography } from 'antd'
import { getStockList, getIndexComponents, getIndustries, listIndexes } from '@/api/client'
import type { IndexInfo, UniverseStock } from '@/types'

/** 全市场列表视图:HK / US 无全市场接口,映射到对应宽基指数成分作为代表池。 */
const MARKET_INDEX: Record<string, string> = { HK: 'HSI', US: 'SP500' }

/**
 * 股票池浏览:全市场列表 / 宽基指数成分 / 行业分布三个视图。
 */
export default function UniversePage() {
  const [mode, setMode] = useState<'stock' | 'index' | 'industry'>('stock')
  const [market, setMarket] = useState<string>('CN')
  const [index, setIndex] = useState<string>('CSI300')

  const { data: indexes } = useQuery({ queryKey: ['indexes'], queryFn: listIndexes })
  const indexOptions = (indexes?.indexes || []).map((i: IndexInfo) => ({
    value: i.key,
    label: `${i.key} · ${i.name}`,
  }))
  const indexNames = (indexes?.indexes || []).reduce(
    (acc, i: IndexInfo) => ({ ...acc, [i.key]: i.name }),
    {} as Record<string, string>,
  )

  // 全市场列表:CN 走全市场;HK / US 无全市场接口,映射到宽基指数成分
  const stockTarget = MARKET_INDEX[market] ?? null
  const stockList = useQuery({
    queryKey: ['stock-list', market],
    queryFn: () => getStockList(market),
    enabled: mode === 'stock' && stockTarget === null,
  })
  const stockIndexComps = useQuery({
    queryKey: ['index-components', stockTarget],
    queryFn: () => getIndexComponents(stockTarget!),
    enabled: mode === 'stock' && stockTarget !== null,
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

  const indexSelect = (
    <Select
      style={{ width: 240 }}
      value={index}
      onChange={setIndex}
      options={indexOptions}
      showSearch
      optionFilterProp="label"
    />
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
          title={stockTarget ? `${indexNames[stockTarget] || stockTarget} 成分股` : '全市场股票列表'}
          extra={
            <Segmented
              value={market}
              onChange={(v) => setMarket(String(v))}
              options={['CN', 'HK', 'US']}
            />
          }
        >
          {stockTarget !== null ? (
            stockIndexComps.isLoading ? (
              <Spin />
            ) : stockIndexComps.isError ? (
              <Typography.Text type="danger">无法解析指数成分(确认指数键与 provider 支持)。</Typography.Text>
            ) : (
              <Space direction="vertical" style={{ width: '100%' }}>
                <Typography.Text>
                  {indexNames[stockTarget] || stockTarget} · {stockIndexComps.data?.count ?? 0} 只
                </Typography.Text>
                <Table
                  size="small"
                  rowKey={(_, i) => String(i)}
                  dataSource={(stockIndexComps.data?.symbols || []).map((s, i) => ({ symbol: s, key: i }))}
                  pagination={{ pageSize: 50 }}
                  columns={[
                    { title: '代码', dataIndex: 'symbol', key: 'symbol' },
                    {
                      title: '市场', key: 'market',
                      render: () => <Tag>{market}</Tag>,
                    },
                  ]}
                />
              </Space>
            )
          ) : stockList.isLoading ? (
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
        <Card title="指数成分股" extra={indexSelect}>
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
        <Card title="行业分布" extra={indexSelect}>
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

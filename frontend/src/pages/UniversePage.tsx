import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  AutoComplete, Button, Card, Col, Descriptions, Empty, Input, Row, Segmented, Select, Space, Spin, Tag, Typography,
} from 'antd'
import { SearchOutlined } from '@ant-design/icons'
import { getIndexComponents, getStockDetail, listIndexes, searchStocks } from '@/api/client'
import type { IndexInfo, StockDetail as StockDetailT, SymbolSearchResult } from '@/types'

const HISTORY_KEY = 'djinn:recent_stocks'
const HISTORY_MAX = 10

function loadHistory(): string[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (!raw) return []
    const arr = JSON.parse(raw)
    return Array.isArray(arr) ? arr.filter((s) => typeof s === 'string') : []
  } catch {
    return []
  }
}

function saveHistory(symbols: string[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(symbols.slice(0, HISTORY_MAX)))
  } catch {
    // localStorage 不可用(隐私模式等)则忽略
  }
}

/** 展示单只股票详情(字段按数据源能力降级,缺失显示 —)。 */
function DetailCard({ detail, onSearch }: { detail: StockDetailT; onSearch: (s: string, m: string) => void }) {
  const fmt = (v: number | null | undefined, suffix = '', digits = 2) =>
    v == null || Number.isNaN(v) ? '—' : `${v.toFixed(digits)}${suffix}`
  const cap = (v: number | null | undefined) =>
    v == null || Number.isNaN(v) ? '—' : `${(v / 1e8).toFixed(1)} 亿`
  return (
    <Card
      size="small"
      title={
        <Space>
          <span>{detail.name || detail.symbol}</span>
          <Typography.Text code>{detail.symbol}</Typography.Text>
          <Tag color="blue">{detail.market}</Tag>
        </Space>
      }
      extra={<Button size="small" type="primary" icon={<SearchOutlined />} onClick={() => onSearch(detail.symbol, detail.market)}>再查</Button>}
    >
      <Row gutter={16}>
        <Col span={8}>
          <Descriptions size="small" column={1} title="估值">
            <Descriptions.Item label="价格">{fmt(detail.price)}</Descriptions.Item>
            <Descriptions.Item label="PE">{fmt(detail.pe)}</Descriptions.Item>
            <Descriptions.Item label="PB">{fmt(detail.pb)}</Descriptions.Item>
            <Descriptions.Item label="PS">{fmt(detail.ps)}</Descriptions.Item>
            <Descriptions.Item label="总市值">{cap(detail.market_cap)}</Descriptions.Item>
            <Descriptions.Item label="流通市值">{cap(detail.float_cap)}</Descriptions.Item>
          </Descriptions>
        </Col>
        <Col span={8}>
          <Descriptions size="small" column={1} title="财务">
            <Descriptions.Item label="ROE">{fmt(detail.roe, '%')}</Descriptions.Item>
            <Descriptions.Item label="毛利率">{fmt(detail.gross_margin, '%')}</Descriptions.Item>
            <Descriptions.Item label="营收同比">{fmt(detail.revenue_yoy, '%')}</Descriptions.Item>
            <Descriptions.Item label="净利同比">{fmt(detail.profit_yoy, '%')}</Descriptions.Item>
          </Descriptions>
        </Col>
      </Row>
    </Card>
  )
}

/**
 * 股票池:股票搜索(详情 + 历史) + 宽基指数成分 两个视图。
 */
export default function UniversePage() {
  const [tab, setTab] = useState<'search' | 'index'>('search')
  const [market, setMarket] = useState<string>('CN')
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<SymbolSearchResult | null>(null)
  const [detail, setDetail] = useState<StockDetailT | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [history, setHistory] = useState<string[]>(loadHistory)
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

  const indexComps = useQuery({
    queryKey: ['index-components', index],
    queryFn: () => getIndexComponents(index),
    enabled: tab === 'index',
  })

  const { data: searchResp, isFetching: searching } = useQuery({
    queryKey: ['stock-search', query, market],
    queryFn: () => searchStocks(query, market),
    enabled: tab === 'search' && query.trim().length >= 1,
  })
  const searchOptions = (searchResp?.results || []).map((r) => ({
    value: r.symbol,
    label: `${r.symbol} · ${r.name}`,
  }))

  const loadDetail = async (symbol: string, m: string) => {
    setSelected({ symbol, market: m, name: '' })
    setDetailLoading(true)
    setDetail(null)
    try {
      const d = await getStockDetail(symbol, m)
      setDetail(d)
      // 写入历史(去重,最近在前)
      const next = [symbol, ...history.filter((s) => s !== symbol)].slice(0, HISTORY_MAX)
      setHistory(next)
      saveHistory(next)
    } catch (e: any) {
      setDetail(null)
    } finally {
      setDetailLoading(false)
    }
  }

  const onPick = (symbol: string) => {
    if (!symbol) return
    setQuery(symbol)
    loadDetail(symbol, market)
  }

  const searchMarket = (
    <Segmented
      value={market}
      onChange={(v) => { setMarket(String(v)); setDetail(null) }}
      options={['CN', 'HK', 'US']}
    />
  )

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>股票池</Typography.Title>
      <Segmented
        value={tab}
        onChange={(v) => setTab(v as 'search' | 'index')}
        options={[
          { label: '股票搜索', value: 'search' },
          { label: '指数成分', value: 'index' },
        ]}
      />

      {tab === 'search' && (
        <>
          <Card title="股票搜索" extra={searchMarket}>
            <AutoComplete
              style={{ width: 360 }}
              value={query}
              onChange={setQuery}
              options={searchOptions}
              onSelect={onPick}
              placeholder="输入代码或名称(如 AAPL / 600519 / 0700.HK)"
              filterOption={false}
            >
              <Input
                prefix={searching ? <Spin size="small" /> : <SearchOutlined />}
                allowClear
                onPressEnter={() => { if (query.trim()) onPick(query.trim().toUpperCase()) }}
              />
            </AutoComplete>
            {searchOptions.length === 0 && query.trim() && !searching && (
              <Typography.Text type="secondary" style={{ marginLeft: 12 }}>
                无匹配(美股用代码搜索)
              </Typography.Text>
            )}
          </Card>

          {detailLoading ? (
            <Spin />
          ) : detail ? (
            <DetailCard detail={detail} onSearch={(s, m) => loadDetail(s, m)} />
          ) : selected && !detailLoading ? (
            <Card size="small">
              <Typography.Text type="warning">未找到 {selected.symbol} 的详情,请确认代码与市场。</Typography.Text>
            </Card>
          ) : (
            <Typography.Text type="secondary">搜索股票以查看详情。</Typography.Text>
          )}

          <Card
            title="最近搜索"
            extra={<Button size="small" onClick={() => { setHistory([]); saveHistory([]) }}>清空</Button>}
          >
            {history.length === 0 ? (
              <Empty description="暂无历史搜索" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              <Space wrap>
                {history.map((sym) => (
                  <Button
                    key={sym}
                    size="small"
                    onClick={() => { setQuery(sym); loadDetail(sym, market) }}
                  >
                    {sym}
                  </Button>
                ))}
              </Space>
            )}
          </Card>
        </>
      )}

      {tab === 'index' && (
        <Card
          title="指数成分股"
          extra={
            <Select
              style={{ width: 240 }}
              value={index}
              onChange={setIndex}
              options={indexOptions}
              showSearch
              optionFilterProp="label"
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
              <Row gutter={[8, 8]}>
                {(indexComps.data?.symbols || []).map((s, i) => (
                  <Col span={6} key={i}>
                    <Typography.Text code>{s}</Typography.Text>
                    <Typography.Text type="secondary" style={{ marginLeft: 6 }}>
                      {indexComps.data?.names?.[i] || ''}
                    </Typography.Text>
                  </Col>
                ))}
              </Row>
            </Space>
          )}
        </Card>
      )}
    </Space>
  )
}

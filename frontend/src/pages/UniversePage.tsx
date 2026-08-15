import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import {
  AutoComplete, Button, Card, Col, Descriptions, Dropdown, Empty, Input, Pagination, Row, Segmented, Select, Space, Spin, Tag, Typography, message,
} from 'antd'
import { PlusOutlined, SearchOutlined } from '@ant-design/icons'
import { errDetail, getIndexComponents, getStockDetail, listIndexes, listProfiles, searchStocks, updateProfile } from '@/api/client'
import type { IndexInfo, Profile, StockDetail as StockDetailT, SymbolSearchResult } from '@/types'
import ProfileManager from '@/components/ProfileManager'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'

const HISTORY_KEY = 'djinn:recent_stocks'
const HISTORY_MAX = 10

/** 历史搜索条目:记住当时的代码 + 市场。 */
interface HistoryItem {
  symbol: string
  market: string
}

function loadHistory(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (!raw) return []
    const arr = JSON.parse(raw)
    if (!Array.isArray(arr)) return []
    // 兼容旧格式(纯字符串列表)直接丢弃;只保留含 symbol+market 的条目
    return arr.filter(
      (x): x is HistoryItem =>
        !!x && typeof x === 'object' && typeof x.symbol === 'string' && typeof x.market === 'string',
    )
  } catch {
    return []
  }
}

function saveHistory(items: HistoryItem[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX)))
  } catch {
    // localStorage 不可用(隐私模式等)则忽略
  }
}

/** 展示单只股票详情(字段按数据源能力降级,缺失显示 —)。 */
function DetailCard({ detail, profiles, onAddToProfile }: {
  detail: StockDetailT
  profiles: Profile[]
  onAddToProfile: (profileId: string, symbol: string) => void
}) {
  const fmt = (v: number | null | undefined, suffix = '', digits = 2) =>
    v == null || Number.isNaN(v) ? '—' : `${v.toFixed(digits)}${suffix}`
  const cap = (v: number | null | undefined) =>
    v == null || Number.isNaN(v) ? '—' : `${(v / 1e8).toFixed(1)} 亿`

  const inProfiles = profiles.filter((pf) => pf.symbols.includes(detail.symbol))
  const notInProfiles = profiles.filter((pf) => !pf.symbols.includes(detail.symbol))

  const p = detail.profile
  type PF = { label: string; value: string | number | null; kind: 'num' | 'int' | 'pct' | 'cap' | 'str' }
  const profileGroups: { title: string; items: PF[] }[] = p
    ? [
        {
          title: '估值扩展',
          items: [
            { label: '预测 PE', value: p.forward_pe, kind: 'num' },
            { label: 'EPS TTM', value: p.eps_ttm, kind: 'num' },
            { label: '预测 EPS', value: p.forward_eps, kind: 'num' },
            { label: 'PEG', value: p.peg_ratio, kind: 'num' },
            { label: '每股净资产', value: p.book_value, kind: 'num' },
            { label: '企业价值', value: p.enterprise_value, kind: 'cap' },
            { label: 'EV/EBITDA', value: p.ev_to_ebitda, kind: 'num' },
            { label: 'Beta', value: p.beta, kind: 'num' },
          ],
        },
        {
          title: '盈利质量',
          items: [
            { label: '营业利润率', value: p.operating_margin, kind: 'pct' },
            { label: '净利率', value: p.profit_margin, kind: 'pct' },
            { label: 'ROA', value: p.return_on_assets, kind: 'pct' },
          ],
        },
        {
          title: '财务健康',
          items: [
            { label: '流动比率', value: p.current_ratio, kind: 'num' },
            { label: '速动比率', value: p.quick_ratio, kind: 'num' },
            { label: '负债权益比', value: p.debt_to_equity, kind: 'pct' },
            { label: '现金', value: p.total_cash, kind: 'cap' },
            { label: '总负债', value: p.total_debt, kind: 'cap' },
            { label: '自由现金流', value: p.free_cashflow, kind: 'cap' },
          ],
        },
        {
          title: '行情动量',
          items: [
            { label: '52 周高', value: p.fifty_two_week_high, kind: 'num' },
            { label: '52 周低', value: p.fifty_two_week_low, kind: 'num' },
            { label: '50 日均线', value: p.fifty_day_avg, kind: 'num' },
            { label: '200 日均线', value: p.two_hundred_day_avg, kind: 'num' },
          ],
        },
        {
          title: '分析师',
          items: [
            { label: '目标价均值', value: p.target_mean_price, kind: 'num' },
            { label: '目标价高', value: p.target_high_price, kind: 'num' },
            { label: '目标价低', value: p.target_low_price, kind: 'num' },
            { label: '分析师数', value: p.number_of_analysts, kind: 'int' },
            { label: '评级', value: p.recommendation, kind: 'str' },
          ],
        },
        {
          title: '公司概况',
          items: [
            { label: '板块', value: p.sector, kind: 'str' },
            { label: '行业', value: p.industry, kind: 'str' },
            { label: '网站', value: p.website, kind: 'str' },
          ],
        },
      ]
    : []

  const renderPF = (f: PF) => {
    const v = f.value
    if (v == null || (typeof v === 'number' && Number.isNaN(v))) return '—'
    if (f.kind === 'str') return String(v)
    const n = v as number
    if (f.kind === 'pct') return `${n.toFixed(2)}%`
    if (f.kind === 'cap') return `${(n / 1e8).toFixed(1)} 亿`
    if (f.kind === 'int') return String(Math.round(n))
    return n.toFixed(2)
  }

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
      extra={
        <Space size={6}>
          <Typography.Text type="secondary">Profile</Typography.Text>
          {inProfiles.length > 0 ? (
            inProfiles.map((pf) => <Tag key={pf.profile_id} color="green" style={{ fontSize: 13 }}>{pf.name}</Tag>)
          ) : (
            <Typography.Text type="secondary">未加入</Typography.Text>
          )}
          <Dropdown
            menu={{
              items: notInProfiles.map((pf) => ({ key: pf.profile_id, label: pf.name })),
              onClick: ({ key }) => onAddToProfile(key, detail.symbol),
            }}
            disabled={notInProfiles.length === 0}
            trigger={['click']}
          >
            <Button icon={<PlusOutlined />}>加入 Profile</Button>
          </Dropdown>
        </Space>
      }
    >
      <Row gutter={16}>
        <Col xs={24} sm={12} lg={8}>
          <Descriptions size="small" column={1} title="估值">
            <Descriptions.Item label="价格">{fmt(detail.price)}</Descriptions.Item>
            <Descriptions.Item label="PE">{fmt(detail.pe)}</Descriptions.Item>
            <Descriptions.Item label="PB">{fmt(detail.pb)}</Descriptions.Item>
            <Descriptions.Item label="PS">{fmt(detail.ps)}</Descriptions.Item>
            <Descriptions.Item label="总市值">{cap(detail.market_cap)}</Descriptions.Item>
            <Descriptions.Item label="流通市值">{cap(detail.float_cap)}</Descriptions.Item>
          </Descriptions>
        </Col>
        <Col xs={24} sm={12} lg={8}>
          <Descriptions size="small" column={1} title="财务">
            <Descriptions.Item label="ROE">{fmt(detail.roe, '%')}</Descriptions.Item>
            <Descriptions.Item label="毛利率">{fmt(detail.gross_margin, '%')}</Descriptions.Item>
            <Descriptions.Item label="营业收入">{cap(detail.revenue)}</Descriptions.Item>
            <Descriptions.Item label="净利润">{cap(detail.net_profit)}</Descriptions.Item>
            <Descriptions.Item label="营收同比">{fmt(detail.revenue_yoy, '%')}</Descriptions.Item>
            <Descriptions.Item label="净利同比">{fmt(detail.profit_yoy, '%')}</Descriptions.Item>
          </Descriptions>
        </Col>
      </Row>

      {p && (
        <>
          <Typography.Title level={5} style={{ marginTop: 16 }}>扩展档案</Typography.Title>
          <Row gutter={16}>
            {profileGroups.map((g) => (
              <Col key={g.title} span={8} style={{ marginBottom: 12 }}>
                <Descriptions size="small" column={1} title={g.title} bordered>
                  {g.items.map((f) => (
                    <Descriptions.Item key={f.label} label={f.label}>
                      {renderPF(f)}
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </Col>
            ))}
          </Row>
          {p.summary && (
            <Typography.Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
              {p.summary}
            </Typography.Paragraph>
          )}
        </>
      )}
    </Card>
  )
}

/**
 * 股票池:股票搜索(详情 + 历史) + 宽基指数成分 两个视图。
 */
export default function UniversePage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [tab, setTab] = useState<'search' | 'index' | 'profile'>(() => {
    const t = searchParams.get('tab')
    return t === 'index' || t === 'profile' ? t : 'search'
  })
  const [market, setMarket] = useState<string>('US')
  const [query, setQuery] = useState('')
  // F11:搜索防抖(输入 300ms 后触发请求,快速输入不逐字打网络)
  const debouncedQuery = useDebouncedValue(query, 300)
  const [selected, setSelected] = useState<SymbolSearchResult | null>(null)
  const [detail, setDetail] = useState<StockDetailT | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>(loadHistory)
  const [index, setIndex] = useState<string>('CSI300')
  // F20:指数成分大列表分页(每页 100,避免 800 只全量渲染)
  const [compPage, setCompPage] = useState<number>(1)

  const { data: indexes } = useQuery({ queryKey: ['indexes'], queryFn: listIndexes })
  const qc = useQueryClient()
  const { data: profiles } = useQuery({ queryKey: ['profiles'], queryFn: listProfiles })
  const profileList = profiles || []
  const addMut = useMutation({
    mutationFn: (req: { profileId: string; symbols: string[] }) => updateProfile(req.profileId, { symbols: req.symbols }),
    onSuccess: () => {
      message.success('已加入 Profile')
      qc.invalidateQueries({ queryKey: ['profiles'] })
    },
    onError: (e) => message.error(errDetail(e)),
  })
  const onAddToProfile = (profileId: string, symbol: string) => {
    const p = profileList.find((x) => x.profile_id === profileId)
    if (!p) return
    addMut.mutate({ profileId, symbols: [...p.symbols, symbol] })
  }
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
    queryKey: ['stock-search', debouncedQuery, market],
    queryFn: () => searchStocks(debouncedQuery, market),
    enabled: tab === 'search' && debouncedQuery.trim().length >= 1,
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
      // 写入历史(按代码去重,记录当时市场,最近在前)
      const next = [
        { symbol, market: m },
        ...history.filter((h) => h.symbol !== symbol),
      ].slice(0, HISTORY_MAX)
      setHistory(next)
      saveHistory(next)
    } catch {
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
        onChange={(v) => {
          const t = v as 'search' | 'index' | 'profile'
          setTab(t)
          setSearchParams({ tab: t }, { replace: true })
        }}
        options={[
          { label: '股票搜索', value: 'search' },
          { label: '指数成分', value: 'index' },
          { label: 'Profile', value: 'profile' },
        ]}
      />

      {tab === 'search' && (
        <>
          <Card title="股票搜索">
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Space>
                <Typography.Text type="secondary">市场</Typography.Text>
                {searchMarket}
              </Space>
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
                <Typography.Text type="secondary">
                  无匹配(美股用代码搜索)
                </Typography.Text>
              )}
            </Space>
          </Card>

          {detailLoading ? (
            <Spin />
          ) : detail ? (
            <DetailCard detail={detail} profiles={profileList} onAddToProfile={onAddToProfile} />
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
                {history.map((h) => (
                  <Button
                    key={h.symbol}
                    size="small"
                    onClick={() => { setMarket(h.market); setQuery(h.symbol); loadDetail(h.symbol, h.market) }}
                  >
                    {h.symbol}
                    <Typography.Text type="secondary" style={{ marginLeft: 6 }}>{h.market}</Typography.Text>
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
              onChange={(v) => { setIndex(v); setCompPage(1) }}
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
                {(indexComps.data?.symbols || []).slice((compPage - 1) * 100, compPage * 100).map((s, i) => {
                  const notIn = profileList.filter((pf) => !pf.symbols.includes(s))
                  const name = indexComps.data?.names?.[(compPage - 1) * 100 + i] || ''
                  return (
                    <Col xs={24} sm={12} lg={8} key={s}>
                      <Card
                        size="small"
                        title={<Typography.Text code style={{ fontSize: 13 }}>{s}</Typography.Text>}
                        extra={
                          <Dropdown
                            menu={{
                              items: notIn.map((pf) => ({ key: pf.profile_id, label: pf.name })),
                              onClick: ({ key }) => onAddToProfile(key, s),
                            }}
                            disabled={notIn.length === 0}
                            trigger={['click']}
                          >
                            <Button
                              type="text"
                              size="small"
                              icon={<PlusOutlined />}
                              title="加入 Profile"
                              aria-label={`加入 ${s} 到 Profile`}
                            />
                          </Dropdown>
                        }
                      >
                        <Typography.Text ellipsis style={{ fontSize: 14 }}>
                          {name || '—'}
                        </Typography.Text>
                      </Card>
                    </Col>
                  )
                })}
              </Row>
              {(indexComps.data?.symbols?.length || 0) > 100 && (
                <Pagination
                  simple
                  pageSize={100}
                  current={compPage}
                  total={indexComps.data?.symbols?.length || 0}
                  onChange={setCompPage}
                />
              )}
            </Space>
          )}
        </Card>
      )}

      {tab === 'profile' && <ProfileManager />}
    </Space>
  )
}

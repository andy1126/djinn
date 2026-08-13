import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Button, Card, DatePicker, Form, Input, Modal, Select, Space, Table, Tabs, Tag, Typography, message } from 'antd'
import dayjs from 'dayjs'
import { fetchData, listCache, clearCache, getCacheContent } from '@/api/client'
import ProfilePicker from '@/components/ProfilePicker'
import type { CacheEntry, Profile } from '@/types'

const { RangePicker } = DatePicker

/** 缓存文件按 ``{provider}::{dtype}::{key}::{adjust}`` 命名,取第二段为类型。 */
function dtypeOf(file: string): string {
  return file.split('::')[1] || 'other'
}

const DTYPE_LABEL: Record<string, string> = {
  quote: '行情',
  fundamental: '基本面',
  universe: '股票池',
}

export default function DataManagerPage() {
  const qc = useQueryClient()
  const [form] = Form.useForm()
  const [symbols, setSymbols] = useState<string[]>(['NVDA'])
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  const { data: cache } = useQuery({
    queryKey: ['cache'],
    queryFn: listCache,
  })
  const entries = (cache?.entries || []) as CacheEntry[]
  const dtypeCounts: Record<string, number> = {}
  for (const e of entries) {
    const d = dtypeOf(e.file)
    dtypeCounts[d] = (dtypeCounts[d] || 0) + 1
  }
  const dtypeTabs = [
    { key: 'quote', label: `行情 (${dtypeCounts.quote || 0})` },
    { key: 'fundamental', label: `基本面 (${dtypeCounts.fundamental || 0})` },
    { key: 'universe', label: `股票池 (${dtypeCounts.universe || 0})` },
  ]

  const { data: content } = useQuery({
    queryKey: ['cache-content', selectedFile],
    queryFn: () => getCacheContent(selectedFile!),
    enabled: !!selectedFile,
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
    {
      title: '类型', key: 'dtype', width: 90,
      render: (_: unknown, r: CacheEntry) => <Tag>{DTYPE_LABEL[dtypeOf(r.file)] || dtypeOf(r.file)}</Tag>,
    },
    { title: '文件', dataIndex: 'file', key: 'file', ellipsis: true },
    { title: '行数', dataIndex: 'rows', key: 'rows' },
    { title: '起始', dataIndex: 'start', key: 'start' },
    { title: '结束', dataIndex: 'end', key: 'end' },
    {
      title: '操作',
      key: 'actions',
      render: (_: unknown, r: CacheEntry) => (
        <Button size="small" onClick={() => setSelectedFile(r.file)} disabled={r.rows < 0}>
          查看
        </Button>
      ),
    },
  ]

  const previewColumns = content
    ? [
        { title: '日期/索引', dataIndex: '_index', key: '_index', fixed: 'left' as const, width: 120 },
        ...content.columns.map((c) => ({
          title: c.name,
          dataIndex: c.name,
          key: c.name,
          render: (v: unknown) => (v == null ? '—' : String(v)),
        })),
      ]
    : []

  const omittedRows = content ? Math.max(0, content.rows - content.head.length - content.tail.length) : 0

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
          <Form.Item label="从 Profile 载入">
            <ProfilePicker
              onSelect={(p: Profile) =>
                form.setFieldsValue({
                  symbols: p.symbols.join(','),
                  ...(p.market ? { market: p.market } : {}),
                })
              }
            />
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
        <Tabs
          items={dtypeTabs.map((t) => ({
            key: t.key,
            label: t.label,
            children: (
              <Table
                columns={cacheColumns}
                dataSource={entries.filter((e) => dtypeOf(e.file) === t.key)}
                rowKey="file"
                size="small"
                pagination={{ pageSize: 20, showSizeChanger: true }}
              />
            ),
          }))}
        />
      </Card>

      <Modal
        title="缓存文件内容"
        open={!!selectedFile}
        onCancel={() => setSelectedFile(null)}
        footer={null}
        width={960}
      >
        {content && (
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <div>
              <Typography.Text code>{content.file}</Typography.Text>
              <Typography.Text type="secondary" style={{ marginLeft: 8 }}>
                {content.rows} 行 · 索引类型 {content.index_type}
              </Typography.Text>
            </div>

            <div>
              <Typography.Text type="secondary">字段</Typography.Text>
              <div style={{ marginTop: 4 }}>
                <Space wrap>
                  {content.columns.map((c) => (
                    <Tag key={c.name} color="blue">
                      {c.name}<span style={{ opacity: 0.65 }}> : {c.dtype}</span>
                    </Tag>
                  ))}
                </Space>
              </div>
            </div>

            <Typography.Text type="secondary">前 {content.head.length} 行</Typography.Text>
            <Table
              columns={previewColumns}
              dataSource={content.head}
              rowKey={(_, i) => String(i)}
              size="small"
              pagination={false}
              scroll={{ x: 'max-content' }}
            />

            {omittedRows > 0 && (
              <Typography.Text type="secondary" style={{ display: 'block', textAlign: 'center' }}>
                ⋯ 中间省略 {omittedRows} 行 ⋯
              </Typography.Text>
            )}

            <Typography.Text type="secondary">后 {content.tail.length} 行</Typography.Text>
            <Table
              columns={previewColumns}
              dataSource={content.tail}
              rowKey={(_, i) => String(i)}
              size="small"
              pagination={false}
              scroll={{ x: 'max-content' }}
            />
          </Space>
        )}
      </Modal>
    </Space>
  )
}
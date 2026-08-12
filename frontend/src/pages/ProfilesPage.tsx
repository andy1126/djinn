import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Card, Form, Input, Modal, Popconfirm, Select, Space, Table, Tag, Typography, message } from 'antd'
import { PlusOutlined } from '@ant-design/icons'
import { createProfile, deleteProfile, listProfiles, updateProfile } from '@/api/client'
import type { Market, Profile } from '@/types'

interface FormValues {
  name: string
  market?: Market
  symbols: string
}

const MARKET_OPTIONS = [
  { label: '美股', value: 'US' },
  { label: 'A股', value: 'CN' },
  { label: '港股', value: 'HK' },
]

const MARKET_LABEL: Record<string, string> = { US: '美股', CN: 'A股', HK: '港股' }

export default function ProfilesPage() {
  const qc = useQueryClient()
  const [form] = Form.useForm<FormValues>()
  const [editing, setEditing] = useState<Profile | null>(null)
  const [open, setOpen] = useState(false)

  const { data: profiles, isLoading } = useQuery({ queryKey: ['profiles'], queryFn: listProfiles })

  const invalidate = () => qc.invalidateQueries({ queryKey: ['profiles'] })

  const createMut = useMutation({
    mutationFn: createProfile,
    onSuccess: () => { message.success('已创建'); setOpen(false); form.resetFields(); invalidate() },
    onError: (e: any) => message.error(e?.response?.data?.detail || '创建失败'),
  })
  const updateMut = useMutation({
    mutationFn: ({ id, req }: { id: string; req: Parameters<typeof updateProfile>[1] }) => updateProfile(id, req),
    onSuccess: () => { message.success('已更新'); setOpen(false); form.resetFields(); invalidate() },
    onError: (e: any) => message.error(e?.response?.data?.detail || '更新失败'),
  })
  const deleteMut = useMutation({
    mutationFn: deleteProfile,
    onSuccess: () => { message.success('已删除'); invalidate() },
    onError: (e: any) => message.error(e?.response?.data?.detail || '删除失败'),
  })

  const openCreate = () => {
    setEditing(null)
    form.resetFields()
    setOpen(true)
  }

  const openEdit = (p: Profile) => {
    setEditing(p)
    form.setFieldsValue({
      name: p.name,
      market: p.market ?? undefined,
      symbols: p.symbols.join(','),
    })
    setOpen(true)
  }

  const onSubmit = (v: FormValues) => {
    const symbols = v.symbols.split(',').map((s) => s.trim()).filter(Boolean)
    const req = { name: v.name, symbols, market: v.market ?? null }
    if (editing) updateMut.mutate({ id: editing.profile_id, req })
    else createMut.mutate(req)
  }

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    {
      title: '市场',
      dataIndex: 'market',
      key: 'market',
      render: (m: string | null) => (m ? <Tag color="blue">{MARKET_LABEL[m] || m}</Tag> : '—'),
    },
    {
      title: '标的数',
      key: 'count',
      render: (_: unknown, p: Profile) => p.symbols.length,
    },
    {
      title: '标的',
      key: 'symbols',
      render: (_: unknown, p: Profile) => (
        <Typography.Text code style={{ wordBreak: 'break-all' }}>
          {p.symbols.join(', ')}
        </Typography.Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: unknown, p: Profile) => (
        <Space>
          <Button size="small" onClick={() => openEdit(p)}>编辑</Button>
          <Popconfirm title="删除该 Profile?" onConfirm={() => deleteMut.mutate(p.profile_id)}>
            <Button size="small" danger>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card
        title="标的 Profile"
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>新建 Profile</Button>}
      >
        <Typography.Paragraph type="secondary">
          保存常用的股票/ETF 列表,在回测/组合/数据页面一键载入,省去重复输入。
        </Typography.Paragraph>
        <Table
          columns={columns}
          dataSource={profiles || []}
          rowKey="profile_id"
          size="small"
          loading={isLoading}
          pagination={false}
          locale={{ emptyText: '暂无 Profile,点击右上角「新建 Profile」创建' }}
        />
      </Card>

      <Modal
        title={editing ? '编辑 Profile' : '新建 Profile'}
        open={open}
        onCancel={() => setOpen(false)}
        onOk={() => form.submit()}
        confirmLoading={createMut.isPending || updateMut.isPending}
        destroyOnClose
      >
        <Form form={form} layout="vertical" onFinish={onSubmit} initialValues={{ market: 'US' }}>
          <Form.Item name="name" label="名称" rules={[{ required: true, message: '请输入名称' }]}>
            <Input placeholder="如 美股科技" />
          </Form.Item>
          <Form.Item name="market" label="市场">
            <Select options={MARKET_OPTIONS} allowClear placeholder="可不选(载入时不变更市场)" />
          </Form.Item>
          <Form.Item
            name="symbols"
            label="标的(逗号分隔)"
            rules={[{ required: true, message: '请输入至少一个标的' }]}
          >
            <Input.TextArea placeholder="NVDA,AAPL,MSFT" rows={3} />
          </Form.Item>
        </Form>
      </Modal>
    </Space>
  )
}

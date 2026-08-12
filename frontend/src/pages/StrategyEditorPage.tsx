import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import {
  Alert, Button, Card, Col, Input, Popconfirm, Row, Select, Space, Table, Tag, Typography, message,
} from 'antd'
import { PlusOutlined } from '@ant-design/icons'
import {
  createUserStrategy, deleteUserStrategy, listUserStrategies, updateUserStrategy, validateUserStrategy,
} from '@/api/client'
import type { ParamSchema, UserStrategy, UserStrategyValidateResponse } from '@/types'

const DEFAULT_TEMPLATE = `fast = param(10, min=2, max=100)
slow = param(30, min=5, max=250)

def signals(self, data):
    close = data["close"]
    ma_fast = sma(close, self.fast)
    ma_slow = sma(close, self.slow)
    up = cross_over(ma_fast, ma_slow)
    down = cross_under(ma_fast, ma_slow)
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[up] = 1
    sig[down] = -1
    return state_from_signals(sig)
`

const KIND_LABEL: Record<string, string> = { python: 'Python', pine: 'Pine Script' }

export default function StrategyEditorPage() {
  const qc = useQueryClient()
  const [editingId, setEditingId] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [kind, setKind] = useState('python')
  const [description, setDescription] = useState('')
  const [code, setCode] = useState(DEFAULT_TEMPLATE)
  const [validate, setValidate] = useState<UserStrategyValidateResponse | null>(null)

  const { data: strategies, isLoading } = useQuery({ queryKey: ['user-strategies'], queryFn: listUserStrategies })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['user-strategies'] })
    qc.invalidateQueries({ queryKey: ['strategies'] })
  }

  const createMut = useMutation({
    mutationFn: createUserStrategy,
    onSuccess: (s) => { message.success('已保存'); setEditingId(s.strategy_id); invalidate() },
    onError: (e: any) => message.error(e?.response?.data?.detail || '保存失败'),
  })
  const updateMut = useMutation({
    mutationFn: ({ id, req }: { id: string; req: Parameters<typeof updateUserStrategy>[1] }) => updateUserStrategy(id, req),
    onSuccess: () => { message.success('已更新'); invalidate() },
    onError: (e: any) => message.error(e?.response?.data?.detail || '更新失败'),
  })
  const deleteMut = useMutation({
    mutationFn: deleteUserStrategy,
    onSuccess: () => { message.success('已删除'); if (editingId) reset(); invalidate() },
    onError: (e: any) => message.error(e?.response?.data?.detail || '删除失败'),
  })
  const validateMut = useMutation({
    mutationFn: validateUserStrategy,
    onSuccess: (r) => setValidate(r),
    onError: (e: any) => message.error(e?.response?.data?.detail || '验证失败'),
  })

  const reset = () => {
    setEditingId(null)
    setName('')
    setKind('python')
    setDescription('')
    setCode(DEFAULT_TEMPLATE)
    setValidate(null)
  }

  const openEdit = (s: UserStrategy) => {
    setEditingId(s.strategy_id)
    setName(s.name)
    setKind(s.kind)
    setDescription(s.description)
    setCode(s.source_code)
    setValidate(null)
  }

  const doValidate = () => validateMut.mutate({ name: name || 'unnamed', source_code: code, kind, description })

  const save = () => {
    const req = { name, source_code: code, kind, description }
    if (editingId) updateMut.mutate({ id: editingId, req })
    else createMut.mutate(req)
  }

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '类型', dataIndex: 'kind', key: 'kind', render: (k: string) => <Tag>{KIND_LABEL[k] || k}</Tag> },
    { title: '说明', dataIndex: 'description', key: 'description', render: (d: string) => d || '—' },
    {
      title: '操作',
      key: 'actions',
      render: (_: unknown, s: UserStrategy) => (
        <Space>
          <Button size="small" onClick={() => openEdit(s)}>编辑</Button>
          <Popconfirm title="删除该策略?" onConfirm={() => deleteMut.mutate(s.strategy_id)}>
            <Button size="small" danger>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card
        title="我的策略"
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={reset}>新建策略</Button>}
      >
        <Typography.Paragraph type="secondary">
          用 Python 写 <Typography.Text code>signals(self, data)</Typography.Text> 或{' '}
          <Typography.Text code>on_bar(self, ctx)</Typography.Text>,参数用{' '}
          <Typography.Text code>param(...)</Typography.Text> 声明、在方法里以{' '}
          <Typography.Text code>self.参数名</Typography.Text> 读取。可用指标见{' '}
          <Typography.Text code>sma/ema/rsi/macd/cross_over/...</Typography.Text>。
        </Typography.Paragraph>
        <Table
          columns={columns}
          dataSource={strategies || []}
          rowKey="strategy_id"
          size="small"
          loading={isLoading}
          pagination={false}
          locale={{ emptyText: '暂无自定义策略,点右上角「新建策略」' }}
        />
      </Card>

      <Card title={editingId ? `编辑策略: ${name || editingId}` : '新建策略'}>
        <Row gutter={16}>
          <Col span={12}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Row gutter={8}>
                <Col span={12}>
                  <Typography.Text>名称</Typography.Text>
                  <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="如 MyMAC" />
                </Col>
                <Col span={6}>
                  <Typography.Text>类型</Typography.Text>
                  <Select
                    style={{ width: '100%' }}
                    value={kind}
                    onChange={setKind}
                    options={[{ label: 'Python', value: 'python' }, { label: 'Pine Script', value: 'pine' }]}
                  />
                </Col>
                <Col span={6}>
                  <Typography.Text>说明(可选)</Typography.Text>
                  <Input value={description} onChange={(e) => setDescription(e.target.value)} />
                </Col>
              </Row>

              <Typography.Text>代码</Typography.Text>
              <CodeMirror
                value={code}
                onChange={(v) => setCode(v)}
                height="420px"
                theme="light"
                extensions={kind === 'python' ? [python()] : []}
              />

              <Space>
                <Button onClick={doValidate} loading={validateMut.isPending}>验证</Button>
                <Button type="primary" onClick={save} loading={createMut.isPending || updateMut.isPending}>保存</Button>
                {editingId && <Button onClick={reset}>取消编辑</Button>}
              </Space>
            </Space>
          </Col>

          <Col span={12}>
            <Card size="small" title="验证结果 / 参数">
              {!validate && <Typography.Text type="secondary">点「验证」编译代码并预览参数表单。</Typography.Text>}
              {validate && !validate.valid && (
                <Alert type="error" showIcon message={validate.error} style={{ whiteSpace: 'pre-wrap' }} />
              )}
              {validate && validate.valid && (
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Alert type="success" showIcon message="编译通过" />
                  {validate.params.map((p: ParamSchema) => (
                    <Row key={p.name} gutter={8}>
                      <Col span={8}><Typography.Text code>{p.name}</Typography.Text></Col>
                      <Col span={8}><Typography.Text type="secondary">{p.type}</Typography.Text></Col>
                      <Col span={8}><Typography.Text>默认 {String(p.default)}</Typography.Text></Col>
                    </Row>
                  ))}
                </Space>
              )}
            </Card>
          </Col>
        </Row>
      </Card>
    </Space>
  )
}

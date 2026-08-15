import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import {
  Alert, Button, Card, Col, Input, Modal, Popconfirm, Row, Space, Table, Tag, Typography, message,
} from 'antd'
import { PlusOutlined } from '@ant-design/icons'
import {
  createUserIndicator, deleteUserIndicator, listIndicators, listUserIndicators,
  updateUserIndicator, validateUserIndicator,
  errDetail,
} from '@/api/client'
import type { IndicatorInfo, UserIndicator, UserIndicatorValidateResponse } from '@/types'
import QueryErrorAlert from '@/components/QueryErrorAlert'

const DEFAULT_TEMPLATE = `def my_roc(close, n=5):
    # N 日变动率
    return close / close.shift(n) - 1
`

export default function IndicatorLibraryPage() {
  const qc = useQueryClient()
  const [viewing, setViewing] = useState<IndicatorInfo | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [code, setCode] = useState(DEFAULT_TEMPLATE)
  const [validate, setValidate] = useState<UserIndicatorValidateResponse | null>(null)

  const { data: catalog, isLoading, isError: indicatorsError, refetch: refetchIndicators } = useQuery({ queryKey: ['indicators'], queryFn: listIndicators })
  const { data: userIndicators } = useQuery({ queryKey: ['user-indicators'], queryFn: listUserIndicators })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['indicators'] })
    qc.invalidateQueries({ queryKey: ['user-indicators'] })
  }

  const createMut = useMutation({
    mutationFn: createUserIndicator,
    onSuccess: (r) => { message.success('已保存'); setEditingId(r.indicator_id); invalidate() },
    onError: (e) => message.error(errDetail(e)),
  })
  const updateMut = useMutation({
    mutationFn: ({ id, req }: { id: string; req: Parameters<typeof updateUserIndicator>[1] }) => updateUserIndicator(id, req),
    onSuccess: () => { message.success('已更新'); invalidate() },
    onError: (e) => message.error(errDetail(e)),
  })
  const deleteMut = useMutation({
    mutationFn: deleteUserIndicator,
    onSuccess: () => { message.success('已删除'); if (editingId) reset(); invalidate() },
    onError: (e) => message.error(errDetail(e)),
  })
  const validateMut = useMutation({
    mutationFn: validateUserIndicator,
    onSuccess: (r) => setValidate(r),
    onError: (e) => message.error(errDetail(e)),
  })

  const reset = () => {
    setEditingId(null)
    setName('')
    setDescription('')
    setCode(DEFAULT_TEMPLATE)
    setValidate(null)
  }

  const openEdit = (u: UserIndicator) => {
    setEditingId(u.indicator_id)
    setName(u.name)
    setDescription(u.description)
    setCode(u.source_code)
    setValidate(null)
  }

  const doValidate = () => validateMut.mutate({ name: name || 'unnamed', source_code: code, description })

  const save = () => {
    const req = { name, source_code: code, description }
    if (editingId) updateMut.mutate({ id: editingId, req })
    else createMut.mutate(req)
  }

  const catalogColumns = [
    { title: '名称', dataIndex: 'name', key: 'name', width: 110, render: (n: string) => <Typography.Text code>{n}</Typography.Text> },
    { title: '分类', dataIndex: 'category', key: 'category', width: 80, render: (c: string) => <Tag color="blue">{c}</Tag> },
    { title: '签名', dataIndex: 'signature', key: 'signature', width: 240, ellipsis: true, render: (s: string) => <Typography.Text code>{s}</Typography.Text> },
    { title: '说明', dataIndex: 'description', key: 'description', width: 380 },
    {
      title: '来源',
      dataIndex: 'origin',
      key: 'origin',
      width: 80,
      render: (o: string) => (o === 'builtin' ? '内置' : <Tag color="green">自定义</Tag>),
    },
    {
      title: '操作',
      key: 'actions',
      width: 70,
      render: (_: unknown, r: IndicatorInfo) => (
        <Button size="small" onClick={() => setViewing(r)}>查看</Button>
      ),
    },
  ]

  const userColumns = [
    { title: '名称', dataIndex: 'name', key: 'name', render: (n: string) => <Typography.Text code>{n}</Typography.Text> },
    { title: '签名', dataIndex: 'signature', key: 'signature', render: (s: string) => <Typography.Text code>{s}</Typography.Text> },
    { title: '说明', dataIndex: 'description', key: 'description', render: (d: string) => d || '—' },
    {
      title: '操作',
      key: 'actions',
      render: (_: unknown, u: UserIndicator) => (
        <Space>
          <Button size="small" onClick={() => openEdit(u)}>编辑</Button>
          <Popconfirm title="删除该指标?" onConfirm={() => deleteMut.mutate(u.indicator_id)}>
            <Button size="small" danger>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="指标库">
        <Typography.Paragraph type="secondary">
          编写策略时可直接调用的指标。点「查看」可看签名与实现逻辑;自定义指标见下方编辑器。
        </Typography.Paragraph>
        {indicatorsError && (
          <QueryErrorAlert error={indicatorsError} retry={refetchIndicators} />
        )}
        <Table
          columns={catalogColumns}
          dataSource={catalog?.indicators || []}
          rowKey="name"
          size="small"
          loading={isLoading}
          pagination={false}
          scroll={{ x: true }}
          tableLayout="fixed"
        />
      </Card>

      <Card
        title="自定义指标"
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={reset}>新建指标</Button>}
      >
        <Typography.Paragraph type="secondary">
          定义 <Typography.Text code>def 函数名(...)</Typography.Text> 返回一个 Series/DataFrame,
          保存后即可在策略里调用(与内置指标同级,也能调用内置指标)。
        </Typography.Paragraph>
        <Table
          columns={userColumns}
          dataSource={userIndicators || []}
          rowKey="indicator_id"
          size="small"
          pagination={false}
          locale={{ emptyText: '暂无自定义指标' }}
        />
        <Row gutter={16} style={{ marginTop: 16 }}>
          <Col xs={24} md={12}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Row gutter={8}>
                <Col xs={24} md={12}>
                  <Typography.Text>函数名</Typography.Text>
                  <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="如 my_roc" />
                </Col>
                <Col xs={24} md={12}>
                  <Typography.Text>说明(可选)</Typography.Text>
                  <Input value={description} onChange={(e) => setDescription(e.target.value)} />
                </Col>
              </Row>
              <Typography.Text>代码</Typography.Text>
              <CodeMirror value={code} onChange={(v) => setCode(v)} height="260px" theme="light" extensions={[python()]} />
              <Space>
                <Button onClick={doValidate} loading={validateMut.isPending}>验证</Button>
                <Button type="primary" onClick={save} loading={createMut.isPending || updateMut.isPending}>保存</Button>
                {editingId && <Button onClick={reset}>取消编辑</Button>}
              </Space>
            </Space>
          </Col>
          <Col xs={24} md={12}>
            <Card size="small" title="验证结果">
              {!validate && <Typography.Text type="secondary">点「验证」编译代码并预览签名。</Typography.Text>}
              {validate && !validate.valid && <Alert type="error" showIcon message={validate.error} style={{ whiteSpace: 'pre-wrap' }} />}
              {validate && validate.valid && (
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Alert type="success" showIcon message="编译通过" />
                  <Typography.Text code>{validate.signature}</Typography.Text>
                </Space>
              )}
            </Card>
          </Col>
        </Row>
      </Card>

      <Modal
        title={viewing ? `${viewing.name} · ${viewing.category}` : '指标详情'}
        open={!!viewing}
        onCancel={() => setViewing(null)}
        footer={null}
        width={720}
      >
        {viewing && (
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <div>
              <Typography.Text code>{viewing.signature}</Typography.Text>
              <Tag style={{ marginLeft: 8 }} color={viewing.origin === 'builtin' ? 'blue' : 'green'}>
                {viewing.origin === 'builtin' ? '内置' : '自定义'}
              </Tag>
            </div>
            {viewing.doc && (
              <Typography.Paragraph type="secondary" style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
                {viewing.doc}
              </Typography.Paragraph>
            )}
            {viewing.source && (
              <div>
                <Typography.Text type="secondary">实现</Typography.Text>
                <pre style={{
                  background: '#f6f8fa', padding: 12, borderRadius: 6, fontSize: 12,
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', overflow: 'auto',
                }}>
                  {viewing.source}
                </pre>
              </div>
            )}
          </Space>
        )}
      </Modal>
    </Space>
  )
}

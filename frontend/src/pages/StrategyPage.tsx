import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import {
  Alert, Button, Card, Col, Divider, Empty, Input, List, Popconfirm, Row, Select, Space, Tag, Typography, message,
} from 'antd'
import { PlusOutlined } from '@ant-design/icons'
import {
  createUserStrategy, deleteUserStrategy, listStrategies, listUserStrategies, updateUserStrategy, validateUserStrategy,
} from '@/api/client'
import { useConfigStore } from '@/store/configStore'
import StrategyParamForm from '@/components/StrategyParamForm'
import type { UserStrategy, UserStrategyValidateResponse } from '@/types'

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

export default function StrategyPage() {
  const qc = useQueryClient()
  const { config, updateConfig } = useConfigStore()

  const { data: allResp } = useQuery({ queryKey: ['strategies'], queryFn: listStrategies })
  const { data: userList, isLoading: userLoading } = useQuery({
    queryKey: ['user-strategies'],
    queryFn: listUserStrategies,
  })

  const allStrategies = allResp?.strategies || []
  const userStrategies = userList || []
  const userMap = new Map(userStrategies.map((u) => [u.name, u]))
  const builtins = allStrategies.filter((s) => !userMap.has(s.name))

  const selectedName = config.strategy.name
  const isUser = userMap.has(selectedName)
  const selectedUser = userMap.get(selectedName)
  const selectedInfo = allStrategies.find((s) => s.name === selectedName)
  const schema = selectedInfo?.params || []

  // 新建策略草稿
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newKind, setNewKind] = useState('python')
  const [newDesc, setNewDesc] = useState('')
  const [newCode, setNewCode] = useState(DEFAULT_TEMPLATE)

  // 选中用户策略的编辑草稿
  const [editCode, setEditCode] = useState('')
  const [editDesc, setEditDesc] = useState('')
  const [validate, setValidate] = useState<UserStrategyValidateResponse | null>(null)

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['user-strategies'] })
    qc.invalidateQueries({ queryKey: ['strategies'] })
  }

  // 选中策略的 params 与 schema 不一致时重置为默认值
  // (覆盖:切换策略、代码改动导致 param 声明变化后)
  useEffect(() => {
    if (creating) return
    const info = allStrategies.find((s) => s.name === selectedName)
    if (!info) return
    const keys = info.params.map((p) => p.name)
    const cur = config.strategy.params || {}
    const curKeys = Object.keys(cur)
    if (keys.length !== curKeys.length || keys.some((k) => !(k in cur))) {
      const defaults: Record<string, number | string | boolean | null> = {}
      info.params.forEach((p) => { defaults[p.name] = p.default })
      updateConfig('strategy', { name: selectedName, params: defaults })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedName, allStrategies, creating])

  // 选中用户策略时同步编辑器草稿
  useEffect(() => {
    if (creating) return
    const u = userMap.get(selectedName)
    setEditCode(u?.source_code ?? '')
    setEditDesc(u?.description ?? '')
    setValidate(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedName, userList, creating])

  const createMut = useMutation({
    mutationFn: createUserStrategy,
    onSuccess: (s) => {
      message.success('策略已创建')
      invalidate()
      setCreating(false)
      updateConfig('strategy', { name: s.name, params: {} })
    },
    onError: (e: any) => message.error(e?.response?.data?.detail || '创建失败'),
  })
  const updateMut = useMutation({
    mutationFn: ({ id, req }: { id: string; req: Parameters<typeof updateUserStrategy>[1] }) =>
      updateUserStrategy(id, req),
    onSuccess: () => { message.success('代码已保存'); invalidate() },
    onError: (e: any) => message.error(e?.response?.data?.detail || '保存失败'),
  })
  const deleteMut = useMutation({
    mutationFn: deleteUserStrategy,
    onSuccess: (_r, id) => {
      message.success('已删除')
      invalidate()
      const deleted = userStrategies.find((u) => u.strategy_id === id)
      if (deleted && deleted.name === selectedName) {
        const first = builtins[0]?.name
        if (first) {
          updateConfig('strategy', { name: first, params: {} })
        }
      }
    },
    onError: (e: any) => message.error(e?.response?.data?.detail || '删除失败'),
  })
  const validateMut = useMutation({
    mutationFn: validateUserStrategy,
    onSuccess: (r) => setValidate(r),
    onError: (e: any) => message.error(e?.response?.data?.detail || '验证失败'),
  })

  const startCreate = () => {
    setCreating(true)
    setNewName('')
    setNewKind('python')
    setNewDesc('')
    setNewCode(DEFAULT_TEMPLATE)
    setValidate(null)
  }

  const selectStrategy = (name: string) => {
    setCreating(false)
    setValidate(null)
    const info = allStrategies.find((s) => s.name === name)
    const params: Record<string, number | string | boolean | null> = {}
    info?.params.forEach((p) => { params[p.name] = p.default })
    updateConfig('strategy', { name, params })
  }

  const validateCreate = () =>
    validateMut.mutate({ name: newName || 'unnamed', source_code: newCode, kind: newKind, description: newDesc })
  const validateEdit = () =>
    validateMut.mutate({
      name: selectedName,
      source_code: editCode,
      kind: selectedUser?.kind || 'python',
      description: editDesc,
    })

  const saveCreate = () => {
    if (!newName.trim()) { message.warning('请填写策略名称'); return }
    createMut.mutate({ name: newName.trim(), source_code: newCode, kind: newKind, description: newDesc })
  }
  const saveEdit = () => {
    if (!selectedUser) return
    updateMut.mutate({ id: selectedUser.strategy_id, req: { source_code: editCode, description: editDesc } })
  }

  return (
    <Row gutter={16}>
      <Col span={8}>
        <Card
          title="策略列表"
          extra={<Button type="primary" icon={<PlusOutlined />} onClick={startCreate}>新建</Button>}
        >
          <Typography.Text type="secondary">内置策略</Typography.Text>
          <List
            size="small"
            dataSource={builtins}
            renderItem={(s) => (
              <List.Item
                onClick={() => selectStrategy(s.name)}
                style={{
                  cursor: 'pointer',
                  padding: '8px 12px',
                  background: !creating && s.name === selectedName ? '#e6f4ff' : undefined,
                  borderRadius: 6,
                }}
              >
                <List.Item.Meta title={s.name} description={s.description} />
              </List.Item>
            )}
          />
          <Divider style={{ margin: '12px 0' }} />
          <Typography.Text type="secondary">自定义策略</Typography.Text>
          <List
            size="small"
            loading={userLoading}
            dataSource={userStrategies}
            locale={{ emptyText: '暂无,点右上角「新建」' }}
            renderItem={(u: UserStrategy) => (
              <List.Item
                onClick={() => selectStrategy(u.name)}
                style={{
                  cursor: 'pointer',
                  padding: '8px 12px',
                  background: !creating && u.name === selectedName ? '#e6f4ff' : undefined,
                  borderRadius: 6,
                }}
                actions={[
                  <Popconfirm
                    key="del"
                    title="删除该策略?"
                    onConfirm={() => deleteMut.mutate(u.strategy_id)}
                  >
                    <Button size="small" type="link" danger onClick={(e) => e.stopPropagation()}>删除</Button>
                  </Popconfirm>,
                ]}
              >
                <List.Item.Meta
                  title={u.name}
                  description={KIND_LABEL[u.kind] || u.kind}
                />
              </List.Item>
            )}
          />
        </Card>
      </Col>

      <Col span={16}>
        {creating ? (
          <Card title="新建策略" extra={<Button onClick={() => setCreating(false)}>取消</Button>}>
            <Row gutter={8} style={{ marginBottom: 12 }}>
              <Col span={8}>
                <Typography.Text>名称</Typography.Text>
                <Input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="如 MyMAC" />
              </Col>
              <Col span={8}>
                <Typography.Text>类型</Typography.Text>
                <Select
                  style={{ width: '100%' }}
                  value={newKind}
                  onChange={setNewKind}
                  options={[{ label: 'Python', value: 'python' }, { label: 'Pine Script', value: 'pine' }]}
                />
              </Col>
              <Col span={8}>
                <Typography.Text>说明(可选)</Typography.Text>
                <Input value={newDesc} onChange={(e) => setNewDesc(e.target.value)} />
              </Col>
            </Row>
            <Typography.Text>代码</Typography.Text>
            <CodeMirror
              value={newCode}
              onChange={(v) => setNewCode(v)}
              height="420px"
              theme="light"
              extensions={newKind === 'python' ? [python()] : []}
            />
            <Space style={{ marginTop: 12 }}>
              <Button onClick={validateCreate} loading={validateMut.isPending}>验证</Button>
              <Button type="primary" onClick={saveCreate} loading={createMut.isPending}>保存</Button>
            </Space>
            {validate && (
              validate.valid
                ? <Alert style={{ marginTop: 12 }} type="success" showIcon message="编译通过" />
                : <Alert style={{ marginTop: 12, whiteSpace: 'pre-wrap' }} type="error" showIcon message={validate.error} />
            )}
          </Card>
        ) : selectedInfo ? (
          <Card
            title={selectedName}
            extra={<Tag color={isUser ? 'purple' : 'blue'}>{isUser ? '自定义' : '内置'}</Tag>}
          >
            <Typography.Paragraph type="secondary">{selectedInfo.description}</Typography.Paragraph>

            {isUser && selectedUser && (
              <>
                <Card
                  size="small"
                  title={`代码 (${KIND_LABEL[selectedUser.kind] || selectedUser.kind})`}
                  style={{ marginBottom: 16 }}
                >
                  <CodeMirror
                    value={editCode}
                    onChange={(v) => setEditCode(v)}
                    height="360px"
                    theme="light"
                    extensions={selectedUser.kind === 'python' ? [python()] : []}
                  />
                  <Space style={{ marginTop: 12 }}>
                    <Input value={editDesc} onChange={(e) => setEditDesc(e.target.value)} placeholder="说明(可选)" style={{ width: 240 }} />
                    <Button onClick={validateEdit} loading={validateMut.isPending}>验证</Button>
                    <Button type="primary" onClick={saveEdit} loading={updateMut.isPending}>保存代码</Button>
                  </Space>
                </Card>
                {validate && (
                  validate.valid
                    ? <Alert style={{ marginBottom: 12 }} type="success" showIcon message="编译通过" />
                    : <Alert style={{ marginBottom: 12, whiteSpace: 'pre-wrap' }} type="error" showIcon message={validate.error} />
                )}
              </>
            )}

            <Card size="small" title="策略参数">
              {schema.length === 0 ? (
                <Typography.Text type="secondary">该策略无可调参数。</Typography.Text>
              ) : (
                <StrategyParamForm
                  schema={schema}
                  value={config.strategy.params}
                  onChange={(params) => updateConfig('strategy', { name: selectedName, params })}
                />
              )}
            </Card>

            <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
              当前回测将使用策略「{selectedName}」及其参数。
            </Typography.Paragraph>
          </Card>
        ) : (
          <Card><Empty description="请从左侧选择策略" /></Card>
        )}
      </Col>
    </Row>
  )
}

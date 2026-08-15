import { useQuery } from '@tanstack/react-query'
import { Alert, Card, Space, Switch } from 'antd'
import { healthCheck } from '@/api/client'
import { useUiStore } from '@/store/uiStore'

export default function SettingsPage() {
  const { data: health, error } = useQuery({
    queryKey: ['health'],
    queryFn: healthCheck,
    refetchInterval: 5000,
  })
  const dark = useUiStore((s) => s.dark)
  const toggle = useUiStore((s) => s.toggle)

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="外观">
        <Space>
          <span>暗色模式</span>
          <Switch checked={dark} onChange={toggle} />
        </Space>
      </Card>
      <Card title="系统状态">
        {/* F5:错误内联展示,去掉 5s 一次的 message 弹窗刷屏 */}
        {error ? (
          <Alert type="error" showIcon message="后端服务不可达" />
        ) : (
          <p>后端: {health?.status === 'healthy' ? '🟢 正常' : '🔴 不可达'}</p>
        )}
        <p>API 地址: <code>localhost:8000</code></p>
      </Card>
    </Space>
  )
}

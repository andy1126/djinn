import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, Space, Switch, message } from 'antd'
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

  useEffect(() => {
    if (error) {
      message.error('后端服务不可达')
    }
  }, [error])

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="外观">
        <Space>
          <span>暗色模式</span>
          <Switch checked={dark} onChange={toggle} />
        </Space>
      </Card>
      <Card title="系统状态">
        <p>后端: {health?.status === 'healthy' ? '🟢 正常' : '🔴 不可达'}</p>
        <p>API 地址: <code>localhost:8000</code></p>
      </Card>
    </Space>
  )
}

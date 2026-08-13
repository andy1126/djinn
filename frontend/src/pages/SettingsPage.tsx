import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, Space, message } from 'antd'
import { healthCheck } from '@/api/client'

export default function SettingsPage() {
  const { data: health, error } = useQuery({
    queryKey: ['health'],
    queryFn: healthCheck,
    refetchInterval: 5000,
  })

  useEffect(() => {
    if (error) {
      message.error('后端服务不可达')
    }
  }, [error])

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="系统状态">
        <p>后端: {health?.status === 'healthy' ? '🟢 正常' : '🔴 不可达'}</p>
        <p>API 地址: <code>localhost:8000</code></p>
      </Card>
    </Space>
  )
}

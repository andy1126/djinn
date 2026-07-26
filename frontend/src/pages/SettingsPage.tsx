import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, Form, Input, Select, Slider, InputNumber, Space, message } from 'antd'
import { healthCheck } from '@/api/client'
import { useConfigStore } from '@/store/configStore'

export default function SettingsPage() {
  const { config, updateConfig } = useConfigStore()
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

      <Card title="默认风险参数">
        <Form layout="vertical" initialValues={config.risk}>
          <Form.Item label="单标的最大权重">
            <Slider min={0} max={1} step={0.05} value={config.risk.max_single_weight}
              onChange={(v) => updateConfig('risk', { ...config.risk, max_single_weight: v })}
            />
          </Form.Item>
          <Form.Item label="总仓位上限">
            <InputNumber min={0} max={1} step={0.05} value={config.risk.max_total_position}
              onChange={(v) => updateConfig('risk', { ...config.risk, max_total_position: v ?? 1 })}
              style={{ width: '100%' }}
            />
          </Form.Item>
        </Form>
      </Card>

      <Card title="默认投资组合">
        <Form layout="vertical">
          <Form.Item label="组合模式">
            <Select value={config.portfolio.mode}
              onChange={(v) => updateConfig('portfolio', { ...config.portfolio, mode: v })}
              options={[
                { label: '单标的', value: 'single' },
                { label: '组合', value: 'portfolio' },
              ]}
            />
          </Form.Item>
          <Form.Item label="分配方式">
            <Select value={config.portfolio.allocation}
              onChange={(v) => updateConfig('portfolio', { ...config.portfolio, allocation: v })}
              options={[
                { label: '等权', value: 'equal' },
                { label: '市值加权', value: 'market_cap' },
                { label: '自定义', value: 'custom' },
              ]}
            />
          </Form.Item>
          <Form.Item label="再平衡周期">
            <Select value={config.portfolio.rebalance.period}
              onChange={(v) => updateConfig('portfolio', { ...config.portfolio,
                rebalance: { ...config.portfolio.rebalance, period: v } })}
              options={[
                { label: '不调仓', value: 'none' },
                { label: '每日', value: 'daily' },
                { label: '每周', value: 'weekly' },
                { label: '每月', value: 'monthly' },
                { label: '每季度', value: 'quarterly' },
                { label: '每年', value: 'yearly' },
              ]}
            />
          </Form.Item>
        </Form>
      </Card>
    </Space>
  )
}
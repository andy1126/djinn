import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, Descriptions, Select, Space, Typography } from 'antd'
import { useConfigStore } from '@/store/configStore'
import { listStrategies } from '@/api/client'
import StrategyParamForm from '@/components/StrategyParamForm'

export default function StrategyConfigPage() {
  const { config, updateConfig } = useConfigStore()
  const { data: stratResp } = useQuery({
    queryKey: ['strategies'],
    queryFn: listStrategies,
  })

  const strategies = stratResp?.strategies || []
  const current = strategies.find((s) => s.name === config.strategy.name)

  useEffect(() => {
    // 切换策略时重置 params 为 schema 默认值
    if (current) {
      const defaults: Record<string, any> = {}
      current.params.forEach((p) => { defaults[p.name] = p.default })
      updateConfig('strategy', { name: current.name, params: defaults })
    }
  }, [config.strategy.name])

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="选择策略">
        <Space direction="vertical" style={{ width: '100%' }}>
          <Select
            value={config.strategy.name}
            onChange={(v) => updateConfig('strategy', { name: v, params: {} })}
            style={{ width: 320 }}
            options={strategies.map((s) => ({ label: `${s.name} — ${s.description}`, value: s.name }))}
          />
          {current && (
            <Typography.Text type="secondary">{current.description}</Typography.Text>
          )}
        </Space>
      </Card>

      {current && (
        <Card title="策略参数">
          <StrategyParamForm
            schema={current.params}
            value={config.strategy.params}
            onChange={(params) => updateConfig('strategy', { ...config.strategy, params })}
          />
        </Card>
      )}

      <Card title="当前策略配置预览">
        <Descriptions column={1} size="small" bordered>
          <Descriptions.Item label="策略名称">{config.strategy.name}</Descriptions.Item>
          <Descriptions.Item label="参数">{JSON.stringify(config.strategy.params)}</Descriptions.Item>
        </Descriptions>
      </Card>
    </Space>
  )
}
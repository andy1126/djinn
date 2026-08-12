import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button, Card, Col, Form, Input, InputNumber, Row, Select, Space, Slider, Tag, Typography } from 'antd'
import { PlusOutlined, MinusCircleOutlined } from '@ant-design/icons'
import ProfilePicker from '@/components/ProfilePicker'
import { useConfigStore } from '@/store/configStore'
import type { Profile } from '@/types'

export default function PortfolioConfigPage() {
  const navigate = useNavigate()
  const { config, updateConfig } = useConfigStore()
  const [symbols, setSymbols] = useState<{ sym: string; weight: number }[]>(
    config.universe.symbols.map((s) => ({ sym: s, weight: 1 / config.universe.symbols.length })),
  )

  const addSymbol = () => setSymbols([...symbols, { sym: '', weight: 0 }])
  const removeSymbol = (idx: number) => setSymbols(symbols.filter((_, i) => i !== idx))
  const updateSymbol = (idx: number, key: 'sym' | 'weight', v: string | number) =>
    setSymbols(symbols.map((s, i) => (i === idx ? { ...s, [key]: v } : s)))

  const onLoadProfile = (p: Profile) => {
    setSymbols(p.symbols.map((s) => ({ sym: s, weight: 1 / p.symbols.length })))
    if (p.market) updateConfig('universe', { ...config.universe, market: p.market })
  }

  const apply = () => {
    const syms = symbols.filter((s) => s.sym.trim()).map((s) => s.sym.trim())
    const weights: Record<string, number> = {}
    symbols.forEach((s) => { if (s.sym.trim()) weights[s.sym.trim()] = s.weight })
    const total = Object.values(weights).reduce((a, b) => a + b, 0)
    updateConfig('universe', { ...config.universe, symbols: syms })
    if (config.portfolio.allocation === 'custom' && total > 0) {
      updateConfig('portfolio', { ...config.portfolio, weights })
    }
  }

  const run = () => { apply(); navigate('/backtest') }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="标的池与权重">
        <ProfilePicker onSelect={onLoadProfile} />
        <Typography.Paragraph type="secondary">
          等权/市值加权时权重会自动计算;自定义权重需手动指定(总和应接近 1)。
        </Typography.Paragraph>
        {symbols.map((s, i) => (
          <Row key={i} gutter={8} style={{ marginBottom: 8 }} align="middle">
            <Col span={10}>
              <Input placeholder="标的代码" value={s.sym} onChange={(e) => updateSymbol(i, 'sym', e.target.value)} />
            </Col>
            <Col span={10}>
              <Slider min={0} max={1} step={0.01} value={s.weight}
                onChange={(v) => updateSymbol(i, 'weight', v ?? 0)}
                tooltip={{ formatter: (v) => `${((v ?? 0) * 100).toFixed(0)}%` }}
              />
            </Col>
            <Col span={4}>
              <Button danger icon={<MinusCircleOutlined />} onClick={() => removeSymbol(i)} disabled={symbols.length <= 1} />
            </Col>
          </Row>
        ))}
        <Button type="dashed" icon={<PlusOutlined />} onClick={addSymbol} style={{ marginTop: 8 }}>添加标的</Button>
      </Card>

      <Card title="分配与再平衡">
        <Form layout="vertical">
          <Form.Item label="组合模式">
            <Select value={config.portfolio.mode}
              onChange={(v) => updateConfig('portfolio', { ...config.portfolio, mode: v })}
              options={[{ label: '单标的', value: 'single' }, { label: '组合', value: 'portfolio' }]} />
          </Form.Item>
          <Form.Item label="分配方式">
            <Select value={config.portfolio.allocation}
              onChange={(v) => updateConfig('portfolio', { ...config.portfolio, allocation: v })}
              options={[
                { label: '等权', value: 'equal' },
                { label: '市值加权', value: 'market_cap' },
                { label: '自定义', value: 'custom' },
              ]} />
          </Form.Item>
          <Form.Item label={`再平衡周期 (${config.portfolio.rebalance.period})`}>
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
              ]} />
          </Form.Item>
          <Form.Item label={`权重偏离阈值: ${(config.portfolio.rebalance.threshold * 100).toFixed(0)}%`}>
            <Slider min={0} max={0.5} step={0.01} value={config.portfolio.rebalance.threshold}
              onChange={(v) => updateConfig('portfolio', { ...config.portfolio,
                rebalance: { ...config.portfolio.rebalance, threshold: v } })} />
          </Form.Item>
        </Form>
      </Card>

      <Card title="资金与费用">
        <Form layout="vertical">
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item label="初始资金">
                <InputNumber min={1} value={config.account.initial_cash} style={{ width: '100%' }}
                  onChange={(v) => updateConfig('account', { ...config.account, initial_cash: v ?? 100000 })} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="币种">
                <Select value={config.account.currency}
                  onChange={(v) => updateConfig('account', { ...config.account, currency: v })}
                  options={[{ label: 'USD', value: 'USD' }, { label: 'CNY', value: 'CNY' }, { label: 'HKD', value: 'HKD' }]} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="佣金模型">
                <Select value={config.costs.commission.type}
                  onChange={(v) => updateConfig('costs', { ...config.costs, commission: { ...config.costs.commission, type: v } })}
                  options={[
                    { label: '按市场默认', value: 'default' },
                    { label: 'A股', value: 'china' },
                    { label: '美股', value: 'us' },
                    { label: '港股', value: 'hk' },
                  ]} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item label={`滑点 bps: ${config.costs.slippage.bps ?? 0}`}>
            <Slider min={0} max={50} value={config.costs.slippage.bps ?? 0}
              onChange={(v) => updateConfig('costs', { ...config.costs, slippage: { ...config.costs.slippage, bps: v } })} />
          </Form.Item>
        </Form>
      </Card>

      <Card title="风险约束">
        <Form layout="vertical">
          <Form.Item label={`单标的最大权重: ${(config.risk.max_single_weight * 100).toFixed(0)}%`}>
            <Slider min={0} max={1} step={0.05} value={config.risk.max_single_weight}
              onChange={(v) => updateConfig('risk', { ...config.risk, max_single_weight: v })} />
          </Form.Item>
          <Form.Item label={`总仓位上限: ${(config.risk.max_total_position * 100).toFixed(0)}%`}>
            <Slider min={0} max={1} step={0.05} value={config.risk.max_total_position}
              onChange={(v) => updateConfig('risk', { ...config.risk, max_total_position: v })} />
          </Form.Item>
        </Form>
      </Card>

      <Space>
        <Button type="primary" onClick={run}>应用并运行回测</Button>
        <Button onClick={apply}>仅保存配置</Button>
        <Tag color="blue">当前 {symbols.length} 个标的</Tag>
      </Space>
    </Space>
  )
}
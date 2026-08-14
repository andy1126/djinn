import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Alert, Button, Card, Col, DatePicker, Descriptions, Form, Input, InputNumber, Progress, Row, Select, Space, message } from 'antd'
import dayjs from 'dayjs'
import { createBacktest, listStrategies } from '@/api/client'
import ProfilePicker from '@/components/ProfilePicker'
import StrategyParamForm from '@/components/StrategyParamForm'
import { useJobProgress } from '@/hooks/useJobProgress'
import { useConfigStore } from '@/store/configStore'
import type { BacktestConfig, Profile, StrategyInfo } from '@/types'

const ALLOC_LABEL: Record<string, string> = {
  equal: '等权', market_cap: '市值加权', custom: '自定义', score: '打分',
  risk_parity: '风险平价', min_variance: '最小方差', mean_variance: '均值方差',
}
const REBALANCE_LABEL: Record<string, string> = {
  none: '不调仓', daily: '每日', weekly: '每周', monthly: '每月', quarterly: '每季度', yearly: '每年',
}
const COMMISSION_LABEL: Record<string, string> = {
  default: '市场默认', china: 'A股', us: '美股', hk: '港股',
}

const { RangePicker } = DatePicker

export default function BacktestRunPage() {
  const navigate = useNavigate()
  const { config, setConfig, updateConfig, reset } = useConfigStore()
  const [form] = Form.useForm()
  const [jobId, setJobId] = useState<string | null>(null)
  const { job: progress, via } = useJobProgress(jobId)

  const { data: strategiesResp } = useQuery({ queryKey: ['strategies'], queryFn: listStrategies })
  const strategies: StrategyInfo[] = strategiesResp?.strategies || []
  const selectedStrategy = strategies.find((s) => s.name === config.strategy.name)

  const selectStrategy = (name: string) => {
    const info = strategies.find((s) => s.name === name)
    const params: Record<string, number | string | boolean | null> = {}
    info?.params.forEach((p) => { params[p.name] = p.default })
    updateConfig('strategy', { name, params })
  }

  // F3:表单跟随 store 配置(跨页变更同步),依赖 [config] 而非 mount-only
  useEffect(() => {
    form.setFieldsValue({
      symbols: config.universe.symbols.join(','),
      market: config.universe.market,
      range: [dayjs(config.period.start), dayjs(config.period.end)],
      initialCash: config.account.initial_cash,
      currency: config.account.currency,
      adjust: config.adjust,
    })
  }, [config, form])

  // 任务完成 / 失败后的跳转与提示
  useEffect(() => {
    if (!progress) return
    if (progress.status === 'done') {
      message.success('回测完成')
      setTimeout(() => navigate(`/results/${progress.job_id}`), 500)
    } else if (progress.status === 'error') {
      message.error(`回测失败: ${progress.error}`)
    }
  }, [progress?.status])

  const syncConfig = (v: any): BacktestConfig => {
    const [start, end] = v.range || []
    const next = {
      ...config,
      universe: {
        ...config.universe,
        symbols: v.symbols.split(',').map((s: string) => s.trim()).filter(Boolean),
        market: v.market,
      },
      period: { start: start.format('YYYY-MM-DD'), end: end.format('YYYY-MM-DD') },
      account: { ...config.account, initial_cash: v.initialCash, currency: v.currency },
      adjust: v.adjust,
    } as BacktestConfig
    setConfig(next)
    return next
  }

  const onSubmit = async (v: any) => {
    const cfg = syncConfig(v)
    try {
      const resp = await createBacktest({ config: cfg })
      setJobId(resp.job_id)
      // 进度订阅交给 useJobProgress(jobId) 处理(含卸载清理与断连降级)
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '创建失败')
    }
  }

  return (
    <Row gutter={16}>
      <Col span={12}>
        <Card title="回测配置">
          <Form
            form={form}
            layout="vertical"
            onFinish={onSubmit}
            initialValues={{
              market: 'US',
              adjust: 'backward',
              initialCash: 100000,
              currency: 'USD',
            }}
          >
            <Form.Item name="symbols" label="标的(逗号分隔)" rules={[{ required: true }]}>
              <Input placeholder="NVDA,AAPL" />
            </Form.Item>
            <Form.Item label="从 Profile 载入">
              <ProfilePicker
                onSelect={(p: Profile) =>
                  form.setFieldsValue({
                    symbols: p.symbols.join(','),
                    ...(p.market ? { market: p.market } : {}),
                  })
                }
              />
            </Form.Item>
            <Row gutter={8}>
              <Col span={8}>
                <Form.Item name="market" label="市场">
                  <Select options={[
                    { label: '美股', value: 'US' },
                    { label: 'A股', value: 'CN' },
                    { label: '港股', value: 'HK' },
                  ]} />
                </Form.Item>
              </Col>
              <Col span={16}>
                <Form.Item name="range" label="区间" rules={[{ required: true }]}>
                  <RangePicker style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={8}>
              <Col span={12}>
                <Form.Item name="initialCash" label="初始资金">
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="currency" label="币种">
                  <Input />
                </Form.Item>
              </Col>
            </Row>
            <Form.Item name="adjust" label="复权方式">
              <Select options={[
                { label: '后复权(推荐)', value: 'backward' },
                { label: '前复权', value: 'forward' },
                { label: '不复权', value: 'none' },
              ]} />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit">开始回测</Button>
                <Button onClick={() => { if (confirm('重置为默认配置?')) reset() }}>重置为默认</Button>
              </Space>
            </Form.Item>
          </Form>
        </Card>

        <Card title="策略" style={{ marginTop: 16 }}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Select
              style={{ width: '100%' }}
              value={config.strategy.name}
              onChange={selectStrategy}
              options={strategies.map((s) => ({ value: s.name, label: s.name }))}
              showSearch
              optionFilterProp="label"
            />
            {selectedStrategy && selectedStrategy.params.length > 0 && (
              <StrategyParamForm
                schema={selectedStrategy.params}
                value={config.strategy.params}
                onChange={(params) => updateConfig('strategy', { name: selectedStrategy.name, params })}
              />
            )}
            <Button size="small" onClick={() => navigate('/strategies')}>配置策略(编辑代码)</Button>
          </Space>
        </Card>

        <Card title="当前组合配置" size="small" style={{ marginTop: 16 }}>
          <Descriptions size="small" column={2} bordered>
            <Descriptions.Item label="组合模式">{config.portfolio.mode === 'portfolio' ? '组合' : '单标的'}</Descriptions.Item>
            <Descriptions.Item label="分配方式">{ALLOC_LABEL[config.portfolio.allocation] || config.portfolio.allocation}</Descriptions.Item>
            <Descriptions.Item label="再平衡">{REBALANCE_LABEL[config.portfolio.rebalance.period] || config.portfolio.rebalance.period}</Descriptions.Item>
            <Descriptions.Item label="偏离阈值">{(config.portfolio.rebalance.threshold * 100).toFixed(0)}%</Descriptions.Item>
            <Descriptions.Item label="单标的最大权重">{(config.risk.max_single_weight * 100).toFixed(0)}%</Descriptions.Item>
            <Descriptions.Item label="总仓位上限">{(config.risk.max_total_position * 100).toFixed(0)}%</Descriptions.Item>
            <Descriptions.Item label="佣金">{COMMISSION_LABEL[config.costs.commission.type] || config.costs.commission.type}</Descriptions.Item>
            <Descriptions.Item label="滑点">{config.costs.slippage.bps ?? 0} bps</Descriptions.Item>
          </Descriptions>
        </Card>
      </Col>

      <Col span={12}>
        <Card title="运行状态">
          {!jobId && <Alert message='点击"开始回测"提交任务' type="info" showIcon />}
          {jobId && progress && (
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <div>任务: <strong>{progress.title || jobId}</strong></div>
              <div>任务 ID: <strong>{jobId}</strong></div>
              <Progress percent={Math.round(progress.progress * 100)} status={progress.status === 'error' ? 'exception' : progress.status === 'done' ? 'success' : 'active'} />
              <div>状态: {progress.status}</div>
              <div>阶段: {progress.stage || '—'}</div>
              {via === 'poll' && (
                <Alert type="warning" showIcon message="实时连接已断开,已切换轮询" />
              )}
              {progress.error && <Alert type="error" message={progress.error} />}
              {progress.status === 'done' && (
                <Button type="primary" onClick={() => navigate(`/results/${jobId}`)}>查看结果</Button>
              )}
            </Space>
          )}
        </Card>
      </Col>
    </Row>
  )
}
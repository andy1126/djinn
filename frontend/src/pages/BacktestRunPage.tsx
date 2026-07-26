import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Alert, Button, Card, Col, DatePicker, Form, Input, InputNumber, Progress, Row, Select, Space, message } from 'antd'
import dayjs from 'dayjs'
import { createBacktest, subscribeProgress } from '@/api/client'
import { useConfigStore } from '@/store/configStore'
import type { JobStatus } from '@/types'

const { RangePicker } = DatePicker

export default function BacktestRunPage() {
  const navigate = useNavigate()
  const { config, updateConfig } = useConfigStore()
  const [form] = Form.useForm()
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<JobStatus | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    form.setFieldsValue({
      symbols: config.universe.symbols.join(','),
      market: config.universe.market,
      range: [dayjs(config.period.start), dayjs(config.period.end)],
      initialCash: config.account.initial_cash,
      currency: config.account.currency,
      adjust: config.adjust,
    })
  }, [])

  const syncConfig = (v: any) => {
    const [start, end] = v.range || []
    updateConfig('universe', {
      symbols: v.symbols.split(',').map((s: string) => s.trim()).filter(Boolean),
      benchmark: config.universe.benchmark,
      market: v.market,
    })
    updateConfig('period', { start: start.format('YYYY-MM-DD'), end: end.format('YYYY-MM-DD') })
    updateConfig('account', { initial_cash: v.initialCash, currency: v.currency })
    updateConfig('adjust', v.adjust)
  }

  const onSubmit = async (v: any) => {
    syncConfig(v)
    try {
      const resp = await createBacktest({ config })
      setJobId(resp.job_id)
      setProgress({ job_id: resp.job_id, title: '', status: 'pending', progress: 0, stage: '排队中', error: null, result_path: null })
      // 订阅 WebSocket 进度
      wsRef.current?.close()
      wsRef.current = subscribeProgress(resp.job_id, (job) => {
        setProgress(job)
        if (job.status === 'done') {
          message.success('回测完成')
          setTimeout(() => navigate(`/results/${job.job_id}`), 500)
        } else if (job.status === 'error') {
          message.error(`回测失败: ${job.error}`)
        }
      })
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
            <Form.Item label="策略">
              <Space>
                <span>{config.strategy.name}</span>
                <Button size="small" onClick={() => navigate('/strategies')}>修改策略</Button>
              </Space>
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit">开始回测</Button>
            </Form.Item>
          </Form>
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
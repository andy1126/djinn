import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Button, Card, Checkbox, Form, Input, InputNumber, message, Progress, Space, Table, Tag, Typography } from 'antd'
import { createSweep, listSweeps } from '@/api/client'
import { useConfigStore } from '@/store/configStore'
import type { JobStatus } from '@/types'

export default function SweepPage() {
  const qc = useQueryClient()
  const { config } = useConfigStore()
  const [form] = Form.useForm()
  const [jobId, setJobId] = useState<string | null>(null)
  const [result, setResult] = useState<JobStatus | null>(null)

  const { data: jobs } = useQuery({
    queryKey: ['sweeps'],
    queryFn: () => listSweeps(20),
    refetchInterval: (query) => {
      const data = query.state.data as JobStatus[] | undefined
      return data?.some((j) => j.status === 'pending' || j.status === 'running') ? 3000 : false
    },
  })

  const sweepMut = useMutation({
    mutationFn: createSweep,
    onSuccess: (resp) => {
      setJobId(resp.job_id)
      message.success(`扫描任务已创建: ${resp.job_id}`)
      qc.invalidateQueries({ queryKey: ['sweeps'] })
    },
    onError: (e: any) => message.error(e?.response?.data?.detail || '创建失败'),
  })

  const onSubmit = (v: any) => {
    // 解析网格:每行 "name:value1,value2"
    const grid: Record<string, (number | string)[]> = {}
    v.gridText.split('\n').forEach((line: string) => {
      const [name, vals] = line.split(':')
      if (name && vals) {
        grid[name.trim()] = vals.split(',').map((s) => {
          const n = Number(s.trim())
          return Number.isNaN(n) ? s.trim() : n
        })
      }
    })
    sweepMut.mutate({ config, grid, target: v.target, parallel: v.parallel })
  }

  const running = jobs?.find((j) => j.job_id === jobId)

  const columns = [
    {
      title: '任务',
      key: 'job_id',
      render: (_: any, r: JobStatus) => (
        <Space direction="vertical" size={0}>
          <span>{r.title || r.job_id}</span>
          <Typography.Text code type="secondary">{r.job_id}</Typography.Text>
        </Space>
      ),
    },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => <Tag color={s === 'done' ? 'success' : s === 'error' ? 'error' : 'processing'}>{s}</Tag> },
    { title: '进度', dataIndex: 'progress', key: 'progress', render: (p: number) => <Progress percent={Math.round(p * 100)} size="small" /> },
    { title: '阶段', dataIndex: 'stage', key: 'stage' },
    { title: '错误', dataIndex: 'error', key: 'error', render: (e: string) => e || '—' },
  ]

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="参数扫描">
        <Form
          form={form}
          layout="vertical"
          onFinish={onSubmit}
          initialValues={{
            target: 'sharpe',
            parallel: false,
            gridText: 'fast:5,10,20\nslow:20,30,50',
          }}
        >
          <Form.Item name="gridText" label="参数网格(每行:name:value1,value2)" rules={[{ required: true }]}>
            <Input.TextArea rows={4} placeholder="fast:5,10,20&#10;slow:20,30,50" />
          </Form.Item>
          <Form.Item name="target" label="优化目标">
            <Input placeholder="sharpe" style={{ width: 200 }} />
          </Form.Item>
          <Form.Item name="parallel" valuePropName="checked" label="并行执行">
            <Checkbox>joblib 多进程</Checkbox>
          </Form.Item>
          <Form.Item label="基础配置">
            <span>策略: {config.strategy.name} · 标的: {config.universe.symbols.join(',')}</span>
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={sweepMut.isPending}>开始扫描</Button>
          </Form.Item>
        </Form>
      </Card>

      {jobId && running && (
        <Card title={`当前扫描 ${running.title || running.job_id}`}>
          <Progress percent={Math.round(running.progress * 100)} status={running.status === 'done' ? 'success' : running.status === 'error' ? 'exception' : 'active'} />
          <div>阶段: {running.stage}</div>
        </Card>
      )}

      <Card title="历史扫描任务">
        <Table
          columns={columns}
          dataSource={(jobs || []) as JobStatus[]}
          rowKey="job_id"
          size="small"
          pagination={{ pageSize: 10 }}
        />
      </Card>
    </Space>
  )
}
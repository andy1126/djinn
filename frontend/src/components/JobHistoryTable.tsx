import { Button, Progress, Space, Table, Tag, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import type { JobStatus } from '@/types'

interface Props {
  jobs: JobStatus[]
  loading?: boolean
  onOpen: (jobId: string) => void
  extraColumns?: ColumnsType<JobStatus>
}

/** 历史任务表格(F9 共享):任务/状态/进度/阶段/错误/操作 + 页面自定义列。 */
export default function JobHistoryTable({ jobs, loading, onOpen, extraColumns }: Props) {
  const columns: ColumnsType<JobStatus> = [
    {
      title: '任务',
      key: 'job_id',
      render: (_: unknown, r: JobStatus) => (
        <Space direction="vertical" size={0}>
          <span>{r.title || r.job_id}</span>
          <Typography.Text code type="secondary">{r.job_id}</Typography.Text>
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 90,
      render: (s: string) => (
        <Tag color={s === 'done' ? 'success' : s === 'error' ? 'error' : s === 'cancelled' ? 'default' : 'processing'}>
          {s === 'cancelled' ? '已取消' : s}
        </Tag>
      ),
    },
    { title: '进度', dataIndex: 'progress', key: 'progress', width: 100, render: (p: number) => <Progress percent={Math.round(p * 100)} size="small" /> },
    { title: '阶段', dataIndex: 'stage', key: 'stage' },
    { title: '错误', dataIndex: 'error', key: 'error', render: (e: string) => e || '—' },
    {
      title: '操作',
      key: 'action',
      width: 80,
      render: (_: unknown, r: JobStatus) => (
        <Button size="small" onClick={() => onOpen(r.job_id)}>查看</Button>
      ),
    },
  ]
  const allColumns = extraColumns ? [...extraColumns, ...columns] : columns
  return (
    <Table
      columns={allColumns}
      dataSource={jobs}
      rowKey="job_id"
      size="small"
      loading={loading}
      pagination={{ pageSize: 10 }}
      scroll={{ x: true }}
    />
  )
}

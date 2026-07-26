import { useQuery } from '@tanstack/react-query'
import { Button, Card, List, Tag, Typography, Space, Empty } from 'antd'
import { useNavigate } from 'react-router-dom'
import { listBacktests } from '@/api/client'
import type { JobStatus } from '@/types'

const statusColor: Record<string, string> = {
  pending: 'default', running: 'processing', done: 'success', error: 'error',
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const { data: jobs, isLoading } = useQuery({
    queryKey: ['backtests'],
    queryFn: () => listBacktests(20),
    refetchInterval: 3000,
  })

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="快捷操作">
        <Space>
          <Button type="primary" onClick={() => navigate('/backtest')}>新建回测</Button>
          <Button onClick={() => navigate('/data')}>管理数据</Button>
          <Button onClick={() => navigate('/strategies')}>配置策略</Button>
          <Button onClick={() => navigate('/sweep')}>参数扫描</Button>
        </Space>
      </Card>

      <Card title="最近回测任务" loading={isLoading}>
        {!jobs || jobs.length === 0 ? (
          <Empty description="暂无任务" />
        ) : (
          <List
            dataSource={jobs as JobStatus[]}
            renderItem={(job) => (
              <List.Item
                actions={[
                  <Button
                    type="link"
                    onClick={() => navigate(`/results/${job.job_id}`)}
                    disabled={job.status !== 'done'}
                  >
                    查看结果
                  </Button>,
                ]}
              >
                <List.Item.Meta
                  title={<Typography.Text>{job.title || job.job_id}</Typography.Text>}
                  description={
                    <span>
                      <Typography.Text code type="secondary">{job.job_id}</Typography.Text>
                      {' · '}进度 {Math.round(job.progress * 100)}% · {job.stage || '—'}
                    </span>
                  }
                />
                <Tag color={statusColor[job.status]}>{job.status}</Tag>
              </List.Item>
            )}
          />
        )}
      </Card>
    </Space>
  )
}
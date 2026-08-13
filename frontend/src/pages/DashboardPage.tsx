import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button, Card, Empty, Space, Table, Tabs, Tag, Typography } from 'antd'
import { useNavigate, useParams } from 'react-router-dom'
import { listBacktests } from '@/api/client'
import type { JobStatus } from '@/types'
import ReportDetail from '@/components/ReportDetail'
import ReportCompare from '@/components/ReportCompare'

const statusColor: Record<string, string> = {
  pending: 'default', running: 'processing', done: 'success', error: 'error',
}

const statusLabel: Record<string, string> = {
  pending: '排队中', running: '运行中', done: '已完成', error: '失败',
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const { jobId } = useParams()

  const [detailJobId, setDetailJobId] = useState<string | null>(null)
  const [compareJobIds, setCompareJobIds] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState<'detail' | 'compare'>('detail')
  const resultRef = useRef<HTMLDivElement>(null)

  const scrollToResult = () => {
    // 等报告渲染一帧后再滚动,避免目标高度尚未确定
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 80)
  }

  // 深链 /results/:jobId → 预选该任务并切到详情
  useEffect(() => {
    if (jobId) {
      setDetailJobId(jobId)
      setActiveTab('detail')
      scrollToResult()
    }
  }, [jobId])

  const { data: jobs, isLoading } = useQuery({
    queryKey: ['backtests'],
    queryFn: () => listBacktests(100),
    refetchInterval: 3000,
  })

  const openDetail = (id: string) => {
    setDetailJobId(id)
    setActiveTab('detail')
    scrollToResult()
  }

  const columns = [
    {
      title: '标题',
      dataIndex: 'title',
      key: 'title',
      render: (t: string, rec: JobStatus) => (
        <Typography.Text ellipsis style={{ maxWidth: 320 }}>
          {t || rec.job_id}
        </Typography.Text>
      ),
    },
    {
      title: '任务 ID',
      dataIndex: 'job_id',
      key: 'job_id',
      render: (id: string) => <Typography.Text code type="secondary">{id}</Typography.Text>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 90,
      render: (s: string) => <Tag color={statusColor[s]}>{statusLabel[s] || s}</Tag>,
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 100,
      render: (p: number) => `${Math.round(p * 100)}%`,
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      render: (_: unknown, rec: JobStatus) => (
        <Button
          type="link"
          disabled={rec.status !== 'done'}
          onClick={() => openDetail(rec.job_id)}
        >
          查看结果
        </Button>
      ),
    },
  ]

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3} style={{ margin: 0 }}>回测结果</Typography.Title>

      <Card title="快捷操作">
        <Space>
          <Button type="primary" onClick={() => navigate('/backtest')}>新建回测</Button>
          <Button onClick={() => navigate('/sweep')}>参数扫描</Button>
          <Button onClick={() => navigate('/strategies')}>配置策略</Button>
          <Button onClick={() => navigate('/data')}>管理数据</Button>
        </Space>
      </Card>

      <Card
        title="回测任务"
        extra={
          <Button
            type="primary"
            ghost
            disabled={compareJobIds.length < 2}
            onClick={() => setActiveTab('compare')}
          >
            对比所选 ({compareJobIds.length})
          </Button>
        }
      >
        <Table
          rowKey="job_id"
          loading={isLoading}
          columns={columns}
          dataSource={jobs}
          size="middle"
          pagination={{ pageSize: 10, showSizeChanger: false }}
          rowSelection={{
            selectedRowKeys: compareJobIds,
            onChange: (keys) => setCompareJobIds(keys as string[]),
            getCheckboxProps: (rec: JobStatus) => ({
              disabled: rec.status !== 'done',
            }),
          }}
        />
      </Card>

      <div ref={resultRef}>
        <Card>
          <Tabs
            activeKey={activeTab}
            onChange={(k) => setActiveTab(k as 'detail' | 'compare')}
            items={[
              {
                key: 'detail',
                label: '结果详情',
                children: detailJobId ? (
                  <ReportDetail jobId={detailJobId} />
                ) : (
                  <Empty description="点击任务列表中的「查看结果」加载报告" style={{ padding: 48 }} />
                ),
              },
              {
                key: 'compare',
                label: '结果对比',
                children: <ReportCompare jobIds={compareJobIds} />,
              },
            ]}
          />
        </Card>
      </div>
    </Space>
  )
}

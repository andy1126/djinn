import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Card, Empty, Popconfirm, Space, Table, Tabs, Tag, Typography, message } from 'antd'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { cancelJob, listBacktests } from '@/api/client'
import type { JobStatus } from '@/types'
import ReportDetail from '@/components/ReportDetail'
import ReportCompare from '@/components/ReportCompare'
import { useJobTransitionNotify } from '@/hooks/useJobTransitionNotify'

const statusColor: Record<string, string> = {
  pending: 'default', running: 'processing', done: 'success', error: 'error', cancelled: 'default',
}

const statusLabel: Record<string, string> = {
  pending: '排队中', running: '运行中', done: '已完成', error: '失败', cancelled: '已取消',
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const { jobId } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()

  const [detailJobId, setDetailJobId] = useState<string | null>(null)
  // F14:勾选状态与 URL ?compare= 双向同步(可分享/跨设备恢复)
  const [compareJobIds, setCompareJobIds] = useState<string[]>(() => {
    const c = searchParams.get('compare')
    return c ? c.split(',').filter(Boolean) : []
  })
  const [activeTab, setActiveTab] = useState<'detail' | 'compare'>('detail')

  useEffect(() => {
    const next = new URLSearchParams(searchParams)
    if (compareJobIds.length > 0) next.set('compare', compareJobIds.join(','))
    else next.delete('compare')
    setSearchParams(next, { replace: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compareJobIds])

  const cancelMut = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => {
      message.success('已请求取消')
      qc.invalidateQueries({ queryKey: ['backtests'] })
    },
    onError: (e: unknown) =>
      message.error((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || '取消失败'),
  })

  // 深链 /results/:jobId → 预选该任务并切到详情(滚动由 ReportDetail 挂载后自处理)
  useEffect(() => {
    if (jobId) {
      setDetailJobId(jobId)
      setActiveTab('detail')
    }
  }, [jobId])

  const { data: jobs, isLoading } = useQuery({
    queryKey: ['backtests'],
    queryFn: () => listBacktests(100),
    // F4:有进行中任务才轮询,否则停摆
    refetchInterval: (q) =>
      (q.state.data ?? []).some(
        (j) => j.status === 'pending' || j.status === 'running',
      )
        ? 3000
        : false,
  })
  // F16:回测任务 running→终态 迁移 → 全局通知(跨页也能收到)
  useJobTransitionNotify(jobs, 'backtest')

  const openDetail = (id: string) => {
    setDetailJobId(id)
    setActiveTab('detail')
  }

  // 点击「查看结果」/ 深链进入详情时滚动到结果区。
  // 结果区(Card)常驻渲染,不依赖报告加载完成;依赖 detailJobId/activeTab 在 render 提交后触发。
  const resultRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (detailJobId && activeTab === 'detail') {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [detailJobId, activeTab])

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
      width: 160,
      render: (_: unknown, rec: JobStatus) => (
        <Space size={0}>
          <Button
            type="link"
            disabled={rec.status !== 'done'}
            onClick={() => openDetail(rec.job_id)}
          >
            查看结果
          </Button>
          {(rec.status === 'pending' || rec.status === 'running') && (
            <Popconfirm
              title="取消此任务?"
              onConfirm={() => cancelMut.mutate(rec.job_id)}
            >
              <Button type="link" danger>取消</Button>
            </Popconfirm>
          )}
        </Space>
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
          scroll={{ x: true }}
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
                children: (
                  <ReportCompare
                    jobIds={compareJobIds}
                    titles={Object.fromEntries(
                      (jobs ?? []).map((j) => [j.job_id, j.title || j.job_id]),
                    )}
                  />
                ),
              },
            ]}
          />
        </Card>
      </div>
    </Space>
  )
}

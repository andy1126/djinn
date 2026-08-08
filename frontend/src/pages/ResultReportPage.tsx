import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { Alert, Button, Card, Input, Space, Spin, Tabs, Typography, message } from 'antd'
import { getBacktestReport } from '@/api/client'
import MetricsCards from '@/components/MetricsCards'
import EquityCurveChart from '@/components/charts/EquityCurveChart'
import DrawdownChart from '@/components/charts/DrawdownChart'
import ReturnsHeatmap from '@/components/charts/ReturnsHeatmap'
import PositionAreaChart from '@/components/charts/PositionAreaChart'
import TradesTable from '@/components/TradesTable'
import IndustryPieChart from '@/components/charts/IndustryPieChart'
import FactorDistChart from '@/components/charts/FactorDistChart'
import BrinsonBarChart from '@/components/charts/BrinsonBarChart'
import { exportBacktest } from '@/api/client'

export default function ResultReportPage() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [inputJobId, setInputJobId] = useState(jobId || '')

  const { data: report, isLoading, error } = useQuery({
    queryKey: ['report', jobId],
    queryFn: () => getBacktestReport(jobId!),
    enabled: !!jobId,
    retry: 1,
  })

  const onExport = async (fmt: 'csv' | 'excel') => {
    if (!jobId) return
    try {
      const data: any = await exportBacktest(jobId, fmt)
      if (fmt === 'csv') {
        message.success(`已导出 CSV 到 ${data.path}`)
      } else {
        // blob 下载
        const url = URL.createObjectURL(data)
        const a = document.createElement('a')
        a.href = url
        a.download = `${jobId}.xlsx`
        a.click()
        URL.revokeObjectURL(url)
      }
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '导出失败')
    }
  }

  const loadReport = () => {
    if (inputJobId && inputJobId !== jobId) {
      navigate(`/results/${inputJobId}`)
    }
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="回测结果">
        <Space>
          <Input placeholder="输入任务 ID" value={inputJobId} onChange={(e) => setInputJobId(e.target.value)} style={{ width: 240 }} />
          <Button onClick={loadReport}>加载</Button>
          <Button onClick={() => onExport('csv')} disabled={!report}>导出 CSV</Button>
          <Button onClick={() => onExport('excel')} disabled={!report}>导出 Excel</Button>
        </Space>
      </Card>

      {!jobId && <Alert type="info" message="输入任务 ID 后加载报告" showIcon />}

      {jobId && isLoading && <Spin tip="加载报告中(后端会重新运行回测,请稍候)" size="large"><div style={{ height: 200 }} /></Spin>}

      {jobId && error && <Alert type="error" message="加载失败" description={(error as any)?.message} showIcon />}

      {jobId && report && (
        <Tabs
          items={[
            {
              key: 'overview',
              label: '指标总览',
              children: <MetricsCards metrics={report.metrics} benchmark={report.benchmark_stats} />,
            },
            {
              key: 'curve',
              label: '净值曲线',
              children: (
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                  <Card size="small" title="策略净值 vs 基准">
                    <EquityCurveChart equity={report.equity_curve} benchmark={report.benchmark_curve} />
                  </Card>
                  <Card size="small" title="水下回撤">
                    <DrawdownChart drawdown={report.drawdown_curve} />
                  </Card>
                </Space>
              ),
            },
            {
              key: 'heatmap',
              label: '月度收益热力图',
              children: <ReturnsHeatmap monthly={report.monthly_returns} />,
            },
            {
              key: 'positions',
              label: '持仓变化',
              children: <PositionAreaChart weights={report.weights} />,
            },
            {
              key: 'attribution',
              label: '归因',
              children: report.attribution || report.factor_exposure ? (
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                  {report.attribution && (
                    <Card size="small" title="Brinson 行业归因(配置 / 选股 / 交互)">
                      <Typography.Text>
                        超额收益: {(report.attribution.excess_return ?? 0).toFixed(4)} ·
                        三效应之和: {(report.attribution.total_effect ?? 0).toFixed(4)}
                      </Typography.Text>
                      <BrinsonBarChart brinson={report.attribution} />
                    </Card>
                  )}
                  {report.factor_exposure && (
                    <>
                      <Card size="small" title="因子暴露时序">
                        <FactorDistChart exposures={report.factor_exposure.exposures} />
                      </Card>
                      <Card size="small" title="行业权重分布(最末交易日)">
                        <IndustryPieChart industryDistribution={report.factor_exposure.industry_distribution} />
                      </Card>
                    </>
                  )}
                </Space>
              ) : <Typography.Text type="secondary">无归因数据(因子组合策略之外的回测不计算归因)</Typography.Text>,
            },
            {
              key: 'trades',
              label: '交易明细',
              children: <TradesTable trades={report.trades} />,
            },
            {
              key: 'config',
              label: '回测配置',
              children: (
                <Typography.Text>
                  <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4, maxHeight: 500, overflow: 'auto' }}>
                    策略: {report.symbols.join(', ')}
                  </pre>
                </Typography.Text>
              ),
            },
          ]}
        />
      )}
    </Space>
  )
}
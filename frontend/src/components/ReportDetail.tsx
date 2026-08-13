import { useQuery } from '@tanstack/react-query'
import { Alert, Button, Card, Space, Spin, Tabs, Typography, message } from 'antd'
import { getBacktestReport, exportBacktest } from '@/api/client'
import MetricsCards from '@/components/MetricsCards'
import EquityCurveChart from '@/components/charts/EquityCurveChart'
import DrawdownChart from '@/components/charts/DrawdownChart'
import ReturnsHeatmap from '@/components/charts/ReturnsHeatmap'
import PositionAreaChart from '@/components/charts/PositionAreaChart'
import TradesTable from '@/components/TradesTable'
import IndustryPieChart from '@/components/charts/IndustryPieChart'
import FactorDistChart from '@/components/charts/FactorDistChart'
import BrinsonBarChart from '@/components/charts/BrinsonBarChart'

interface Props {
  jobId: string
}

export default function ReportDetail({ jobId }: Props) {
  const { data: report, isLoading, error } = useQuery({
    queryKey: ['report', jobId],
    queryFn: () => getBacktestReport(jobId),
    retry: 1,
  })

  const onExport = async (fmt: 'csv' | 'excel') => {
    try {
      const data: any = await exportBacktest(jobId, fmt)
      if (fmt === 'csv') {
        message.success(`已导出 CSV 到 ${data.path}`)
      } else {
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

  if (isLoading) {
    return (
      <Spin tip="加载报告中(后端会重新运行回测,请稍候)" size="large">
        <div style={{ height: 200 }} />
      </Spin>
    )
  }

  if (error) {
    return <Alert type="error" message="加载失败" description={(error as any)?.message} showIcon />
  }

  if (!report) return null

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card
        title={`回测结果 ${jobId}`}
        extra={
          <Space>
            <Button onClick={() => onExport('csv')}>导出 CSV</Button>
            <Button onClick={() => onExport('excel')}>导出 Excel</Button>
          </Space>
        }
      >
        <Typography.Text type="secondary">
          标的: {report.symbols.join(', ')}
        </Typography.Text>
      </Card>

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
        ]}
      />
    </Space>
  )
}

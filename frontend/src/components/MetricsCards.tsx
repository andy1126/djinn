import { Card, Col, Row, Statistic } from 'antd'
import type { Metrics, BenchmarkStats } from '@/types'
import { fmtNum, fmtPct } from '@/utils/format'

interface Props {
  metrics: Metrics
  benchmark?: BenchmarkStats | null
}

export default function MetricsCards({ metrics, benchmark }: Props) {
  return (
    <Row gutter={[12, 12]}>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="累计收益" value={fmtPct(metrics.total_return)} valueStyle={{ color: metrics.total_return >= 0 ? '#3f8600' : '#cf1322' }} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="年化收益" value={fmtPct(metrics.annual_return)} valueStyle={{ color: metrics.annual_return >= 0 ? '#3f8600' : '#cf1322' }} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="夏普比率" value={fmtNum(metrics.sharpe)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="索提诺" value={fmtNum(metrics.sortino)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="最大回撤" value={fmtPct(metrics.max_drawdown)} valueStyle={{ color: '#cf1322' }} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="卡玛比率" value={fmtNum(metrics.calmar)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="年化波动" value={fmtPct(metrics.annual_volatility)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="交易次数" value={Number(metrics.n_trades)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="胜率" value={fmtPct(metrics.win_rate, 1)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="换手率" value={fmtPct(metrics.turnover)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="VaR 95% (日)" value={fmtPct(metrics.var_95 ?? 0)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="CVaR 95% (日)" value={fmtPct(metrics.cvar_95 ?? 0)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="最长回撤时长" value={`${metrics.max_drawdown_duration ?? 0} 天`} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="最长连亏" value={`${metrics.max_losing_streak ?? 0} 天`} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="Jensen α" value={fmtNum(benchmark?.alpha ?? 0)} /></Card></Col>
      <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="Beta" value={fmtNum(benchmark?.beta ?? 0)} /></Card></Col>
      {benchmark?.information_ratio != null && (
        <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="信息比率" value={fmtNum(benchmark.information_ratio)} /></Card></Col>
      )}
      {benchmark?.excess_return != null && (
        <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="超额收益" value={fmtPct(benchmark.excess_return)} valueStyle={{ color: benchmark.excess_return >= 0 ? '#3f8600' : '#cf1322' }} /></Card></Col>
      )}
      {benchmark?.downside_capture != null && (
        <Col xs={12} sm={12} lg={6}><Card size="small"><Statistic title="下行捕获" value={fmtPct(benchmark.downside_capture, 0)} /></Card></Col>
      )}
    </Row>
  )
}
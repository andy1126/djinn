import { Fragment } from 'react'
import type { ReactNode } from 'react'
import { Card, Col, Row, Space, Tag, Typography } from 'antd'
import { useNavigate } from 'react-router-dom'
import {
  ApiOutlined,
  ArrowRightOutlined,
  BarChartOutlined,
  CodeOutlined,
  DatabaseOutlined,
  DesktopOutlined,
  ExperimentOutlined,
  PieChartOutlined,
  SafetyOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'

const { Title, Paragraph, Text } = Typography

interface Module {
  key: string
  icon: ReactNode
  title: string
  desc: string
  path: string
  color: string
}

// 核心链路:数据 → 研究 → 策略 → 回测 → 组合风控 → 分析归因(与 CLAUDE.md 分层架构一致)
const PIPELINE: Module[] = [
  {
    key: 'data',
    icon: <DatabaseOutlined />,
    title: '数据',
    desc: '多市场行情 + 基本面 + 本地缓存',
    path: '/data',
    color: 'blue',
  },
  {
    key: 'research',
    icon: <ExperimentOutlined />,
    title: '研究',
    desc: '因子引擎 / IC 分析 / 多因子诊断 / 选股',
    path: '/factors',
    color: 'cyan',
  },
  {
    key: 'strategy',
    icon: <BarChartOutlined />,
    title: '策略',
    desc: '择时 + 选股 TopN,Python / Pine',
    path: '/strategies',
    color: 'green',
  },
  {
    key: 'engine',
    icon: <ThunderboltOutlined />,
    title: '回测引擎',
    desc: '事件驱动精确撮合 / 滑点 / 费用',
    path: '/backtest',
    color: 'volcano',
  },
  {
    key: 'portfolio',
    icon: <SafetyOutlined />,
    title: '组合风控',
    desc: 'Decimal 账本 / 再平衡 / 风控约束',
    path: '/portfolio',
    color: 'orange',
  },
  {
    key: 'analytics',
    icon: <PieChartOutlined />,
    title: '分析归因',
    desc: '指标 + Brinson / 因子归因 + 导出',
    path: '/results',
    color: 'purple',
  },
]

const ENTRIES = [
  {
    key: 'cli',
    icon: <CodeOutlined />,
    title: 'CLI',
    desc: 'YAML 配置,可复现,可入版本控制',
  },
  {
    key: 'api',
    icon: <ApiOutlined />,
    title: 'FastAPI 后端',
    desc: 'REST + WebSocket,把内核与 alpha 层包成 Web 服务',
  },
  {
    key: 'web',
    icon: <DesktopOutlined />,
    title: 'React Web',
    desc: '唯一可视化交互入口,13 个页面',
  },
]

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <div>
        <Title level={2} style={{ marginBottom: 4 }}>
          Djinn 多市场量化选股平台
        </Title>
        <Paragraph type="secondary" style={{ marginBottom: 0, maxWidth: 720 }}>
          覆盖 A 股 / 港股 / 美股,从数据、因子研究、策略构建到回测与归因的一体化选股工作台。
          一条链路走完「找数据 → 挖因子 → 定策略 → 跑回测 → 看归因」。
        </Paragraph>
      </div>

      <Card title="核心链路" styles={{ body: { padding: 24 } }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'stretch',
            flexWrap: 'wrap',
            gap: 8,
          }}
        >
          {PIPELINE.map((m, i) => (
            <Fragment key={m.key}>
              <Card
                hoverable
                onClick={() => navigate(m.path)}
                style={{ flex: '1 1 160px', minWidth: 150, textAlign: 'center' }}
                styles={{ body: { padding: 16 } }}
              >
                <div style={{ fontSize: 24, color: '#1677ff', marginBottom: 8 }}>{m.icon}</div>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  <Tag color={m.color} style={{ marginRight: 0 }}>
                    {i + 1}
                  </Tag>{' '}
                  {m.title}
                </div>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {m.desc}
                </Text>
              </Card>
              {i < PIPELINE.length - 1 && (
                <div style={{ alignSelf: 'center', color: '#bbb', flex: '0 0 auto', fontSize: 18 }}>
                  <ArrowRightOutlined />
                </div>
              )}
            </Fragment>
          ))}
        </div>
      </Card>

      <Card title="三种使用方式" styles={{ body: { padding: 24 } }}>
        <Row gutter={16}>
          {ENTRIES.map((e) => (
            <Col key={e.key} xs={24} sm={8}>
              <Card styles={{ body: { padding: 20 } }}>
                <Space direction="vertical" size={4}>
                  <Space size={8}>
                    <span style={{ fontSize: 20, color: '#1677ff' }}>{e.icon}</span>
                    <Text strong>{e.title}</Text>
                  </Space>
                  <Text type="secondary" style={{ fontSize: 13 }}>
                    {e.desc}
                  </Text>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>
    </Space>
  )
}

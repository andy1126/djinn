import { ConfigProvider, Layout, Menu, theme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import {
  DashboardOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  ThunderboltOutlined,
  BarsOutlined,
  BarChartOutlined,
  SettingOutlined,
  AppstoreOutlined,
  SwapOutlined,
  FundOutlined,
  FilterOutlined,
  ProfileOutlined,
  EditOutlined,
  LineChartOutlined,
} from '@ant-design/icons'

const { Header, Sider, Content } = Layout

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, retry: 1 },
  },
})

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '仪表盘' },
  { key: '/data', icon: <DatabaseOutlined />, label: '数据管理' },
  { key: '/universe', icon: <FundOutlined />, label: '股票池' },
  { key: '/profiles', icon: <ProfileOutlined />, label: '标的组合' },
  { key: '/factors', icon: <ExperimentOutlined />, label: '因子分析' },
  { key: '/indicators', icon: <LineChartOutlined />, label: '指标库' },
  { key: '/factor-matrix', icon: <ExperimentOutlined />, label: '多因子诊断' },
  { key: '/screener', icon: <FilterOutlined />, label: '选股' },
  { key: '/strategies', icon: <BarChartOutlined />, label: '策略配置' },
  { key: '/strategies/editor', icon: <EditOutlined />, label: '策略编辑器' },
  { key: '/portfolio', icon: <AppstoreOutlined />, label: '组合配置' },
  { key: '/backtest', icon: <ThunderboltOutlined />, label: '运行回测' },
  { key: '/results', icon: <BarChartOutlined />, label: '结果报告' },
  { key: '/compare', icon: <SwapOutlined />, label: '结果对比' },
  { key: '/sweep', icon: <BarsOutlined />, label: '参数扫描' },
  { key: '/settings', icon: <SettingOutlined />, label: '设置' },
]

export default function LayoutShell() {
  const navigate = useNavigate()
  const location = useLocation()
  const { token } = theme.useToken()

  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider locale={zhCN} theme={{ token: { colorPrimary: '#1677ff' } }}>
        <Layout style={{ minHeight: '100vh' }}>
          <Sider width={200} style={{ background: token.colorBgContainer }}>
            <div style={{ height: 56, padding: 16, fontSize: 18, fontWeight: 700, color: token.colorPrimary }}>
              Djinn 量化回测
            </div>
            <Menu
              mode="inline"
              selectedKeys={[location.pathname]}
              items={menuItems}
              onClick={({ key }) => navigate(key)}
              style={{ borderRight: 0 }}
            />
          </Sider>
          <Layout>
            <Header style={{ background: token.colorBgContainer, padding: '0 24px' }}>
              <h2 style={{ margin: 0, lineHeight: '64px' }}>多市场量化回测平台</h2>
            </Header>
            <Content style={{ padding: 24, overflow: 'auto' }}>
              <Outlet />
            </Content>
          </Layout>
        </Layout>
      </ConfigProvider>
    </QueryClientProvider>
  )
}
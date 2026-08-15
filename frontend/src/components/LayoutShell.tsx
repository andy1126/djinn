import { Suspense, useEffect, useState } from 'react'
import { Badge, ConfigProvider, Dropdown, Layout, Menu, Spin, theme } from 'antd'
import type { MenuProps } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import {
  HomeOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
  SettingOutlined,
  BellOutlined,
} from '@ant-design/icons'
import { useUiStore } from '@/store/uiStore'
import { useNotifyStore } from '@/store/notifyStore'

const { Header, Sider, Content } = Layout

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, retry: 1 },
  },
})

const menuItems: MenuProps['items'] = [
  { key: '/', icon: <HomeOutlined />, label: '首页' },
  {
    key: 'data',
    icon: <DatabaseOutlined />,
    label: '数据',
    children: [
      { key: '/data', label: '数据管理' },
      { key: '/universe', label: '股票池' },
    ],
  },
  {
    key: 'research',
    icon: <ExperimentOutlined />,
    label: '研究',
    children: [
      { key: '/factors', label: '因子分析' },
      { key: '/factor-matrix', label: '多因子诊断' },
      { key: '/screener', label: '选股' },
      { key: '/indicators', label: '指标库' },
    ],
  },
  {
    key: 'strategy',
    icon: <BarChartOutlined />,
    label: '策略',
    children: [
      { key: '/strategies', label: '策略' },
      { key: '/portfolio', label: '组合配置' },
    ],
  },
  {
    key: 'backtest',
    icon: <ThunderboltOutlined />,
    label: '回测',
    children: [
      { key: '/backtest', label: '运行回测' },
      { key: '/sweep', label: '参数扫描' },
      { key: '/results', label: '回测结果' },
    ],
  },
  { key: '/settings', icon: <SettingOutlined />, label: '设置' },
]

// 深链 /results/:jobId 归一化到菜单项 /results,保证「回测结果」高亮
function selectedKeyFor(path: string): string {
  if (path.startsWith('/results')) return '/results'
  return path
}

// 根据当前路径确定需要展开的一级分组
function openGroupFor(path: string): string[] {
  if (path.startsWith('/data') || path.startsWith('/universe')) {
    return ['data']
  }
  if (
    path.startsWith('/factors') ||
    path.startsWith('/factor-matrix') ||
    path.startsWith('/screener') ||
    path.startsWith('/indicators')
  ) {
    return ['research']
  }
  if (path.startsWith('/strategies') || path.startsWith('/portfolio')) {
    return ['strategy']
  }
  if (
    path.startsWith('/backtest') ||
    path.startsWith('/sweep') ||
    path.startsWith('/results')
  ) {
    return ['backtest']
  }
  return []
}

export default function LayoutShell() {
  const navigate = useNavigate()
  const location = useLocation()
  const { token } = theme.useToken()
  const dark = useUiStore((s) => s.dark)
  const notifyItems = useNotifyStore((s) => s.items)
  const notifyUnread = useNotifyStore((s) => s.unread)
  const markAllRead = useNotifyStore((s) => s.markAllRead)
  const [openKeys, setOpenKeys] = useState<string[]>(() => openGroupFor(location.pathname))

  useEffect(() => {
    setOpenKeys(openGroupFor(location.pathname))
  }, [location.pathname])

  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider
        locale={zhCN}
        theme={{
          algorithm: dark ? theme.darkAlgorithm : theme.defaultAlgorithm,
          token: { colorPrimary: '#1677ff' },
        }}
      >
        <Layout style={{ minHeight: '100vh' }}>
          <Sider
            width={224}
            breakpoint="lg"
            collapsedWidth="0"
            style={{ background: token.colorBgContainer }}
          >
            <div style={{ height: 56, padding: 16, fontSize: 18, fontWeight: 700, color: token.colorPrimary }}>
              Djinn 量化回测
            </div>
            <Menu
              mode="inline"
              selectedKeys={[selectedKeyFor(location.pathname)]}
              openKeys={openKeys}
              onOpenChange={(keys) => setOpenKeys(keys as string[])}
              items={menuItems}
              onClick={({ key }) => {
                if (key.startsWith('/')) navigate(key)
              }}
              style={{ borderRight: 0 }}
            />
          </Sider>
          <Layout>
            <Header
              style={{
                background: token.colorBgContainer,
                padding: '0 24px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <h2 style={{ margin: 0, lineHeight: '64px' }}>多市场量化回测平台</h2>
              <Dropdown
                trigger={['click']}
                onOpenChange={(open) => { if (open) markAllRead() }}
                menu={{
                  items: notifyItems.length
                    ? notifyItems.map((n) => ({
                        key: n.id,
                        label: `${n.title} — ${n.status}`,
                        onClick: () => {
                          const kindPath: Record<string, string> = {
                            backtest: '/results',
                            sweep: '/sweep',
                            'factor-analysis': '/factors',
                            'factor-matrix': '/factor-matrix',
                            screen: '/screener',
                          }
                          navigate(`${kindPath[n.kind] ?? '/results'}/${n.jobId}`)
                        },
                      }))
                    : [{ key: 'empty', label: '暂无通知', disabled: true }],
                }}
              >
                <Badge count={notifyUnread} size="small">
                  <BellOutlined style={{ fontSize: 18, cursor: 'pointer' }} />
                </Badge>
              </Dropdown>
            </Header>
            <Content style={{ padding: 24, overflow: 'auto' }}>
              <Suspense fallback={<Spin style={{ display: 'block', margin: '80px auto' }} />}>
                <Outlet />
              </Suspense>
            </Content>
          </Layout>
        </Layout>
      </ConfigProvider>
    </QueryClientProvider>
  )
}

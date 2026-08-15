import { lazy } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import LayoutShell from './components/LayoutShell'

// F8:路由级懒加载,拆分包体积,首屏只加载当前页
const HomePage = lazy(() => import('./pages/HomePage'))
const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const DataManagerPage = lazy(() => import('./pages/DataManagerPage'))
const StrategyPage = lazy(() => import('./pages/StrategyPage'))
const PortfolioConfigPage = lazy(() => import('./pages/PortfolioConfigPage'))
const BacktestRunPage = lazy(() => import('./pages/BacktestRunPage'))
const SweepPage = lazy(() => import('./pages/SweepPage'))
const WalkForwardPage = lazy(() => import('./pages/WalkForwardPage'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))
const UniversePage = lazy(() => import('./pages/UniversePage'))
const FactorAnalysisPage = lazy(() => import('./pages/FactorAnalysisPage'))
const FactorMatrixPage = lazy(() => import('./pages/FactorMatrixPage'))
const ScreenerPage = lazy(() => import('./pages/ScreenerPage'))
const IndicatorLibraryPage = lazy(() => import('./pages/IndicatorLibraryPage'))

export const router = createBrowserRouter([
  {
    path: '/',
    element: <LayoutShell />,
    children: [
      { index: true, element: <HomePage /> },
      { path: 'results', element: <DashboardPage /> },
      { path: 'results/:jobId', element: <DashboardPage /> },
      { path: 'data', element: <DataManagerPage /> },
      { path: 'universe', element: <UniversePage /> },
      { path: 'factors', element: <FactorAnalysisPage /> },
      { path: 'indicators', element: <IndicatorLibraryPage /> },
      { path: 'factor-matrix', element: <FactorMatrixPage /> },
      { path: 'screener', element: <ScreenerPage /> },
      { path: 'strategies', element: <StrategyPage /> },
      { path: 'portfolio', element: <PortfolioConfigPage /> },
      { path: 'backtest', element: <BacktestRunPage /> },
      { path: 'sweep', element: <SweepPage /> },
      { path: 'walk', element: <WalkForwardPage /> },
      { path: 'settings', element: <SettingsPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])

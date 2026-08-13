import { createBrowserRouter, Navigate } from 'react-router-dom'
import LayoutShell from './components/LayoutShell'
import HomePage from './pages/HomePage'
import DashboardPage from './pages/DashboardPage'
import DataManagerPage from './pages/DataManagerPage'
import StrategyPage from './pages/StrategyPage'
import PortfolioConfigPage from './pages/PortfolioConfigPage'
import BacktestRunPage from './pages/BacktestRunPage'
import SweepPage from './pages/SweepPage'
import SettingsPage from './pages/SettingsPage'
import UniversePage from './pages/UniversePage'
import FactorAnalysisPage from './pages/FactorAnalysisPage'
import FactorMatrixPage from './pages/FactorMatrixPage'
import ScreenerPage from './pages/ScreenerPage'
import IndicatorLibraryPage from './pages/IndicatorLibraryPage'

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
      { path: 'settings', element: <SettingsPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])

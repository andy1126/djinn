import { createBrowserRouter, Navigate } from 'react-router-dom'
import LayoutShell from './components/LayoutShell'
import DashboardPage from './pages/DashboardPage'
import DataManagerPage from './pages/DataManagerPage'
import StrategyConfigPage from './pages/StrategyConfigPage'
import StrategyEditorPage from './pages/StrategyEditorPage'
import PortfolioConfigPage from './pages/PortfolioConfigPage'
import BacktestRunPage from './pages/BacktestRunPage'
import ResultReportPage from './pages/ResultReportPage'
import ComparePage from './pages/ComparePage'
import SweepPage from './pages/SweepPage'
import SettingsPage from './pages/SettingsPage'
import UniversePage from './pages/UniversePage'
import FactorAnalysisPage from './pages/FactorAnalysisPage'
import FactorMatrixPage from './pages/FactorMatrixPage'
import ScreenerPage from './pages/ScreenerPage'
import ProfilesPage from './pages/ProfilesPage'
import IndicatorLibraryPage from './pages/IndicatorLibraryPage'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <LayoutShell />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'data', element: <DataManagerPage /> },
      { path: 'universe', element: <UniversePage /> },
      { path: 'profiles', element: <ProfilesPage /> },
      { path: 'factors', element: <FactorAnalysisPage /> },
      { path: 'indicators', element: <IndicatorLibraryPage /> },
      { path: 'factor-matrix', element: <FactorMatrixPage /> },
      { path: 'screener', element: <ScreenerPage /> },
      { path: 'strategies', element: <StrategyConfigPage /> },
      { path: 'strategies/editor', element: <StrategyEditorPage /> },
      { path: 'portfolio', element: <PortfolioConfigPage /> },
      { path: 'backtest', element: <BacktestRunPage /> },
      { path: 'results', element: <ResultReportPage /> },
      { path: 'results/:jobId', element: <ResultReportPage /> },
      { path: 'compare', element: <ComparePage /> },
      { path: 'sweep', element: <SweepPage /> },
      { path: 'settings', element: <SettingsPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])
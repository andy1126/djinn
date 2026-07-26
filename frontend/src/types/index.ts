// 与后端 djinn.api.schemas 对应的类型定义

export type Market = 'CN' | 'HK' | 'US'
export type Adjust = 'none' | 'forward' | 'backward'

export interface ParamSchema {
  name: string
  type: string
  default: number | string | boolean | null
  min: number | null
  max: number | null
  choices: (string | number)[] | null
  description: string | null
}

export interface StrategyInfo {
  name: string
  description: string
  params: ParamSchema[]
}

export interface StrategyListResponse {
  strategies: StrategyInfo[]
}

export interface JobCreated {
  job_id: string
  status: string
}

export interface JobStatus {
  job_id: string
  title: string
  status: 'pending' | 'running' | 'done' | 'error'
  progress: number
  stage: string
  error: string | null
  result_path: string | null
}

// BacktestConfig 子配置
export interface UniverseConfig {
  symbols: string[]
  benchmark?: string | null
  market?: Market | null
}

export interface PeriodConfig {
  start: string
  end: string
}

export interface AccountConfig {
  initial_cash: number
  currency: string
  t_plus_1?: boolean | null
}

export interface CommissionConfig {
  type: 'default' | 'china' | 'us' | 'hk'
  rate?: number | null
  min_commission?: number | null
  stamp_duty_rate?: number | null
  transfer_fee_rate?: number | null
}

export interface SlippageConfig {
  type: 'zero' | 'none' | 'fixed_bps' | 'fixed' | 'random' | 'volume_share'
  bps?: number
}

export interface CostsConfig {
  commission: CommissionConfig
  slippage: SlippageConfig
  enforce_price_limit: boolean
  enforce_suspension: boolean
  enforce_lot: boolean
}

export interface StrategyConfig {
  name: string
  params: Record<string, number | string | boolean | null>
}

export interface RebalanceConfig {
  period: 'none' | 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly'
  threshold: number
  min_hold_days: number
}

export interface PortfolioConfig {
  mode: 'single' | 'portfolio'
  allocation: 'equal' | 'market_cap' | 'custom'
  weights?: Record<string, number> | null
  rebalance: RebalanceConfig
}

export interface RiskConfig {
  max_single_weight: number
  max_total_position: number
  max_sector_weight?: number | null
  sector_map?: Record<string, string> | null
}

export interface OutputConfig {
  dir: string
  export: ('csv' | 'excel')[]
  report: 'html' | 'none'
  rolling_window: number
}

export interface BacktestConfig {
  universe: UniverseConfig
  period: PeriodConfig
  account: AccountConfig
  costs: CostsConfig
  strategy: StrategyConfig
  portfolio: PortfolioConfig
  risk: RiskConfig
  output: OutputConfig
  adjust: Adjust
  risk_free_rate: number
}

export interface BacktestRequest {
  config: BacktestConfig
}

export interface SweepRequest {
  config: BacktestConfig
  grid: Record<string, (number | string)[]>
  target: string
  parallel: boolean
}

export interface DataFetchRequest {
  symbols: string[]
  market?: string | null
  start: string
  end: string
  adjust: string
  csv_dir?: string | null
}

export interface CacheEntry {
  file: string
  rows: number
  start?: string | null
  end?: string | null
  error: boolean
}

export interface CacheResponse {
  entries: CacheEntry[]
}

// 回测报告(完整)
export interface SeriesData {
  index: string[]
  values: number[]
}

export interface DataFrameData {
  index: string[]
  columns: string[]
  data: (number | string | null)[][]
}

export interface Metrics {
  total_return: number
  annual_return: number
  annual_volatility: number
  volatility?: number
  cagr: number
  sharpe: number
  sortino: number
  max_drawdown: number
  calmar: number
  win_rate: number
  profit_loss_ratio: number
  turnover: number
  n_trades: number
  n_days: number
  extra?: Record<string, number>
  [key: string]: number | Record<string, number> | undefined
}

export interface BenchmarkStats {
  alpha?: number
  beta?: number
  tracking_error?: number
  information_ratio?: number
  correlation?: number
  benchmark_return?: number
  strategy_return?: number
  excess_return?: number
  [key: string]: number | undefined
}

export interface TradeRecord {
  [key: string]: number | string | boolean | null
}

export interface BacktestReport {
  job_id: string
  symbols: string[]
  metrics: Metrics
  trade_stats: Record<string, number | string | number[]>
  benchmark_stats: BenchmarkStats | null
  equity_curve: SeriesData
  benchmark_curve: SeriesData
  drawdown_curve: SeriesData
  monthly_returns: DataFrameData
  yearly_returns: SeriesData
  rolling_sharpe: SeriesData
  rolling_volatility: SeriesData
  trades: TradeRecord[]
  rejections: Record<string, number | string>[]
  positions: DataFrameData
  weights: DataFrameData
}
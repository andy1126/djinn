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
  result?: Record<string, unknown> | null
  kind?: string
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
  allocation:
    | 'equal'
    | 'market_cap'
    | 'custom'
    | 'score'
    | 'risk_parity'
    | 'min_variance'
    | 'mean_variance'
  weights?: Record<string, number> | null
  rebalance: RebalanceConfig
}

export interface RiskConfig {
  max_single_weight: number
  max_total_position: number
  max_sector_weight?: number | null
  sector_map?: Record<string, string> | null
  max_turnover?: number | null
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

/** sweep 单组合结果行(后端 _run_one 返回)。 */
export interface SweepResultRow {
  params: Record<string, number | string | boolean | null>
  config_summary: {
    strategy: string
    'universe.index': string | null
    n_symbols: number
    'strategy.factor_weights': Record<string, number> | null
    'portfolio.allocation': string
    'strategy.n_stocks': number | null
    'strategy.rebalance_freq': number | null
    'strategy.params': Record<string, number | string | boolean | null>
  }
  [key: string]: unknown
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
  attribution: BrinsonResult | null
  factor_exposure: FactorExposureReport | null
}

// ── 归因(Phase 5)─────────────────────────────────────
export interface BrinsonResult {
  allocation: SeriesData
  selection: SeriesData
  interaction: SeriesData
  excess_return: number
  total_effect: number
}

export interface FactorExposureReport {
  exposures: DataFrameData
  industry_distribution: DataFrameData
}

// ── 因子库 / 因子分析 ────────────────────────────────────
export interface FactorInfo {
  name: string
  category: string
  description: string
  params: ParamSchema[]
}

export interface FactorListResponse {
  factors: FactorInfo[]
}

export interface FactorAnalysisRequest {
  factor: string
  params?: Record<string, number | string | boolean | null>
  index?: string | null
  symbols?: string[] | null
  market?: string | null
  start: string
  end: string
  adjust?: string
  ic_method?: string
  n_quantiles?: number
  periods?: number[]
}

export interface ICSummary {
  ic_mean: number
  ic_std: number
  icir: number
  ic_pos_ratio: number
  [key: string]: number
}

export interface FactorReport {
  factor_name: string
  ic: SeriesData
  ic_summary: ICSummary
  ic_decay: Record<string, SeriesData>
  quantile_returns: DataFrameData
  quantile_cumulative: Record<string, SeriesData>
  long_short: SeriesData
  monotonicity: number
  turnover: number
  ic_by_group: SeriesData
}

// ── 选股 ───────────────────────────────────────────────
export type ScreenOp = 'gt' | 'lt' | 'ge' | 'le' | 'eq' | 'between' | 'in'

export interface ScreenCondition {
  field: string
  op: ScreenOp
  value: number | string | boolean | (number | string)[]
}

export interface FactorScore {
  factor: string
  weight: number
  direction: 1 | -1
}

export interface ScreenRequest {
  conditions?: ScreenCondition[]
  scores?: FactorScore[]
  top_n?: number | null
  index?: string | null
  symbols?: string[] | null
  market?: string | null
  when?: string | null
  lookback_days?: number
}

export interface ScreenResultRow {
  symbol: string
  score: number | null
  [key: string]: number | string | boolean | null
}

// ── 股票池 ─────────────────────────────────────────────
export interface UniverseStock {
  symbol: string
  name: string
  market: string
}

export interface UniverseStockListResponse {
  market: string | null
  count: number
  stocks: UniverseStock[]
}

export interface IndexInfo {
  key: string
  name: string
  market: string
}

export interface IndexListResponse {
  indexes: IndexInfo[]
}

export interface IndexComponentsResponse {
  index: string
  count: number
  symbols: string[]
}

export interface IndustryCount {
  name: string
  count: number
}

export interface IndustryListResponse {
  industries: IndustryCount[]
}

// ── 多因子诊断 ─────────────────────────────────────────
export interface FactorMatrixPoint {
  factor: string
  weight: number
  direction: 1 | -1
  params?: Record<string, number | string | boolean | null>
}

export interface FactorMatrixRequest {
  factors: FactorMatrixPoint[]
  index?: string | null
  symbols?: string[] | null
  market?: string | null
  start: string
  end: string
  adjust?: string
  ic_method?: string
  periods?: number[]
}

export interface FactorMatrixReport {
  factors: string[]
  correlation: DataFrameData
  ic_summary: Record<string, Record<string, ICSummary>>
  turnover: Record<string, number>
}
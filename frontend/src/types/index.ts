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
  required: boolean
}

export interface StrategyInfo {
  name: string
  description: string
  params: ParamSchema[]
}

export interface StrategyListResponse {
  strategies: StrategyInfo[]
}

export interface UserStrategy {
  strategy_id: string
  name: string
  kind: string
  source_code: string
  description: string
  created_at: string
  updated_at: string
  params: ParamSchema[]
}

export interface UserStrategyCreate {
  name: string
  source_code: string
  kind: string
  description?: string
}

export interface UserStrategyUpdate {
  name?: string
  source_code?: string
  kind?: string
  description?: string
}

export interface UserStrategyValidateResponse {
  valid: boolean
  error: string | null
  params: ParamSchema[]
}

export interface IndicatorParam {
  name: string
  default: number | string | boolean | null
}

export interface IndicatorInfo {
  name: string
  category: string
  description: string
  doc: string
  signature: string
  params: IndicatorParam[]
  source: string
  origin: 'builtin' | 'user'
}

export interface IndicatorListResponse {
  indicators: IndicatorInfo[]
}

export interface UserIndicator {
  indicator_id: string
  name: string
  source_code: string
  description: string
  created_at: string
  updated_at: string
  signature: string
}

export interface UserIndicatorCreate {
  name: string
  source_code: string
  description?: string
}

export interface UserIndicatorUpdate {
  name?: string
  source_code?: string
  description?: string
}

export interface UserIndicatorValidateResponse {
  valid: boolean
  error: string | null
  signature: string
}

export interface JobCreated {
  job_id: string
  status: string
}

export interface JobStatus {
  job_id: string
  title: string
  status: 'pending' | 'running' | 'done' | 'error' | 'cancelled'
  progress: number
  stage: string
  error: string | null
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
  currency: string | null
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
  type: 'zero' | 'fixed_bps' | 'fixed' | 'random' | 'volume_share'
  bps?: number
}

export interface CostsConfig {
  commission: CommissionConfig
  slippage: SlippageConfig
  enforce_price_limit: boolean
  enforce_suspension: boolean
  enforce_lot: boolean
  fill_ref: 'open' | 'close' | 'vwap'
}

export interface StrategyConfig {
  name: string
  params: Record<string, number | string | boolean | null>
  selection?: {
    min_amount?: number | null
    min_list_days?: number | null
    exclude_st?: boolean
    neutralize?: boolean
    industry_neutral?: boolean
    max_sector_weight?: number | null
    min_score_diff?: number
  } | null
  timing?: {
    market_filter?: Record<string, unknown> | null
    exit_rule?: Record<string, unknown> | null
    entry_confirm?: Record<string, unknown> | null
    cooldown_days?: number
  } | null
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
  walk_forward?: WalkForwardConfig | null
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
    'strategy.weighting': string
    'strategy.min_score_diff': number | null
    'portfolio.allocation': string
    'strategy.n_stocks': number | null
    'strategy.rebalance_freq': number | null
    'strategy.params': Record<string, number | string | boolean | null>
  }
  sharpe?: number
  sortino?: number
  calmar?: number
  total_return?: number
  max_drawdown?: number
  n_trades?: number
  [key: string]: unknown
}

// ── Walk-Forward(H 计划)────────────────────────────────
export interface WalkForwardConfig {
  is_days: number
  oos_days: number
  step?: number | null
  n_windows?: number | null
  target: string
  grid: Record<string, (number | string)[]>
  top_k?: number
  min_is_sharpe?: number | null
  warmup_days?: number
}

export interface WalkForwardRequest {
  config: BacktestConfig
  grid?: Record<string, (number | string)[]> | null
  target?: string | null
  parallel: boolean
}

/** WFO 单窗口:IS 最优参数 + OOS 评估。deployed=false 表示 IS 未达标未部署。 */
export interface WFWindow {
  no: number
  is_start: string
  is_end: string
  oos_start: string
  oos_end: string
  deployed: boolean
  best_params?: Record<string, number | string | boolean | null> | null
  is_metrics?: Record<string, unknown> | null
  oos_metrics?: Record<string, unknown> | null
  oos_equity?: SeriesData | null
}

/** Walk-Forward 完整结果:逐窗口 + 拼接样本外净值 + 整体指标。 */
export interface WalkForwardReport {
  target: string
  full_start: string | null
  full_end: string | null
  windows: WFWindow[]
  equity_curve: SeriesData | null
  metrics: Record<string, unknown> | null
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

export interface CacheColumn {
  name: string
  dtype: string
}

export interface CacheContent {
  file: string
  rows: number
  index_type: string
  columns: CacheColumn[]
  head: Record<string, string | number | boolean | null>[]
  tail: Record<string, string | number | boolean | null>[]
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
  turnover_annual?: number
  n_trades: number
  n_round_trips?: number
  n_days: number
  var_95?: number
  cvar_95?: number
  max_drawdown_duration?: number
  max_losing_streak?: number
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
  downside_capture?: number
  upside_capture?: number
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
  prices: DataFrameData
  attribution: BrinsonResult | null
  factor_exposure: FactorExposureReport | null
  meta?: {
    data_caveats?: string[]
    selection_log?: Array<{ date: string; selected: string[]; scores: Record<string, number> }>
  }
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
  ic_t: number
  ic_pvalue: number
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
  recommended_rebalance: string | null
  data_caveats: string[]
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

export interface ScreenField {
  name: string
  label: string
  kind: 'number' | 'string'
  group: 'valuation' | 'financial'
  description: string
}

export interface ScreenFieldsResponse {
  fields: ScreenField[]
}

export interface ScreenMarket {
  market: string
  label: string
  available: boolean
  reason: string
}

export interface ScreenMarketsResponse {
  markets: ScreenMarket[]
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
  names: string[]
}

export interface IndustryCount {
  name: string
  count: number
}

export interface IndustryListResponse {
  industries: IndustryCount[]
}

// ── 标的 profile ───────────────────────────────────────
export interface Profile {
  profile_id: string
  name: string
  symbols: string[]
  market?: Market | null
  created_at: string
  updated_at: string
}

export interface ProfileCreate {
  name: string
  symbols: string[]
  market?: Market | null
}

export interface ProfileUpdate {
  name?: string
  symbols?: string[]
  market?: Market | null
}

// ── 股票搜索 / 详情 ─────────────────────────────────────
export interface SymbolSearchResult {
  symbol: string
  market: string
  name: string
}

export interface SymbolSearchResponse {
  query: string
  results: SymbolSearchResult[]
}

export interface StockDetail {
  symbol: string
  market: string
  name: string
  price: number | null
  pe: number | null
  pb: number | null
  ps: number | null
  market_cap: number | null
  float_cap: number | null
  roe: number | null
  gross_margin: number | null
  revenue: number | null
  net_profit: number | null
  revenue_yoy: number | null
  profit_yoy: number | null
  profile: StockProfile | null
}

export interface StockProfile {
  forward_pe: number | null
  eps_ttm: number | null
  forward_eps: number | null
  peg_ratio: number | null
  book_value: number | null
  enterprise_value: number | null
  ev_to_ebitda: number | null
  beta: number | null
  operating_margin: number | null
  profit_margin: number | null
  return_on_assets: number | null
  current_ratio: number | null
  quick_ratio: number | null
  debt_to_equity: number | null
  total_cash: number | null
  total_debt: number | null
  free_cashflow: number | null
  fifty_two_week_high: number | null
  fifty_two_week_low: number | null
  fifty_day_avg: number | null
  two_hundred_day_avg: number | null
  target_mean_price: number | null
  target_high_price: number | null
  target_low_price: number | null
  number_of_analysts: number | null
  dividend_rate: number | null
  dividend_yield: number | null
  sector: string | null
  industry: string | null
  recommendation: string | null
  website: string | null
  summary: string | null
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
  orthogonalized?: boolean
}

export interface FMBLambda {
  lambda_mean: number
  lambda_t: number
  lambda_pvalue: number
  pos_ratio: number
}

export interface FMBReport {
  n_days: number
  lambdas: Record<string, FMBLambda>
}

export interface FactorMatrixReport {
  factors: string[]
  correlation: DataFrameData
  ic_summary: Record<string, Record<string, ICSummary>>
  turnover: Record<string, number>
  fmb: FMBReport | null
}
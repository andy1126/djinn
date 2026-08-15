import axios from 'axios'
import type {
  BacktestRequest,
  BacktestReport,
  CacheContent,
  CacheResponse,
  DataFetchRequest,
  FactorAnalysisRequest,
  FactorInfo,
  FactorListResponse,
  FactorMatrixRequest,
  FactorMatrixReport,
  FactorReport,
  IndexComponentsResponse,
  IndexListResponse,
  IndicatorInfo,
  IndicatorListResponse,
  IndustryListResponse,
  JobCreated,
  JobStatus,
  Profile,
  ProfileCreate,
  ProfileUpdate,
  ScreenFieldsResponse,
  ScreenMarketsResponse,
  ScreenRequest,
  StockDetail,
  StrategyListResponse,
  StrategyInfo,
  SweepRequest,
  UserIndicator,
  UserIndicatorCreate,
  UserIndicatorUpdate,
  UserIndicatorValidateResponse,
  UserStrategy,
  UserStrategyCreate,
  UserStrategyUpdate,
  UserStrategyValidateResponse,
  SymbolSearchResponse,
  UniverseStockListResponse,
} from '@/types'

// 通过 vite proxy,前缀 /api 转发到 http://localhost:8000
const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'
// WebSocket 直连后端(默认同主机 8000;跨域部署用 VITE_WS_BASE 覆盖)
const WS_BASE =
  import.meta.env.VITE_WS_BASE ??
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:8000`

const http = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
})

// F7:全局响应拦截器——在错误对象上挂友好消息(列表页内联展示,不全局弹窗)
http.interceptors.response.use(
  (resp) => resp,
  (err) => {
    err.friendlyMessage =
      err?.response?.data?.detail ?? err?.message ?? '请求失败'
    return Promise.reject(err)
  },
)

/** 从错误对象提取友好文案(拦截器已挂 friendlyMessage,兜底 message)。 */
export const errDetail = (e: unknown): string =>
  (e as { friendlyMessage?: string })?.friendlyMessage
  ?? (e as { message?: string })?.message
  ?? '请求失败'

// ── 策略 ─────────────────────────────────────────────
export const listStrategies = async (): Promise<StrategyListResponse> =>
  (await http.get('/strategies')).data

export const getStrategy = async (name: string): Promise<StrategyInfo> =>
  (await http.get(`/strategies/${name}`)).data

export const listUserStrategies = async (): Promise<UserStrategy[]> =>
  (await http.get('/strategies/user')).data

export const createUserStrategy = async (
  req: UserStrategyCreate,
): Promise<UserStrategy> => (await http.post('/strategies/user', req)).data

export const updateUserStrategy = async (
  id: string,
  req: UserStrategyUpdate,
): Promise<UserStrategy> => (await http.put(`/strategies/user/${id}`, req)).data

export const deleteUserStrategy = async (id: string): Promise<void> => {
  await http.delete(`/strategies/user/${id}`)
}

export const validateUserStrategy = async (
  req: UserStrategyCreate,
): Promise<UserStrategyValidateResponse> =>
  (await http.post('/strategies/user/validate', req)).data

// ── 指标库 ─────────────────────────────────────────────
export const listIndicators = async (): Promise<IndicatorListResponse> =>
  (await http.get('/indicators')).data

export const getIndicator = async (name: string): Promise<IndicatorInfo> =>
  (await http.get(`/indicators/${name}`)).data

export const listUserIndicators = async (): Promise<UserIndicator[]> =>
  (await http.get('/indicators/user')).data

export const createUserIndicator = async (
  req: UserIndicatorCreate,
): Promise<UserIndicator> => (await http.post('/indicators/user', req)).data

export const updateUserIndicator = async (
  id: string,
  req: UserIndicatorUpdate,
): Promise<UserIndicator> => (await http.put(`/indicators/user/${id}`, req)).data

export const deleteUserIndicator = async (id: string): Promise<void> => {
  await http.delete(`/indicators/user/${id}`)
}

export const validateUserIndicator = async (
  req: UserIndicatorCreate,
): Promise<UserIndicatorValidateResponse> =>
  (await http.post('/indicators/user/validate', req)).data

// ── 回测 ─────────────────────────────────────────────
export const createBacktest = async (req: BacktestRequest): Promise<JobCreated> =>
  (await http.post('/backtests', req)).data

export const listBacktests = async (limit = 50): Promise<JobStatus[]> =>
  (await http.get(`/backtests?limit=${limit}`)).data

export const getBacktest = async (jobId: string): Promise<JobStatus> =>
  (await http.get(`/backtests/${jobId}`)).data

export const getBacktestReport = async (jobId: string): Promise<BacktestReport> =>
  (await http.get(`/backtests/${jobId}/report`)).data

export const exportBacktest = async (
  jobId: string,
  fmt: 'csv' | 'excel',
): Promise<unknown> => {
  const res = await http.get(`/backtests/${jobId}/export/${fmt}`, {
    responseType: fmt === 'excel' ? 'blob' : 'json',
  })
  return res.data
}

// ── 取消任务(E4/F17:统一端点)─────────────────────────────
export const cancelJob = async (jobId: string): Promise<{ status: string }> =>
  (await http.post(`/jobs/${jobId}/cancel`)).data

// ── 扫描 ─────────────────────────────────────────────
export const createSweep = async (req: SweepRequest): Promise<JobCreated> =>
  (await http.post('/sweeps', req)).data

export const listSweeps = async (limit = 50): Promise<JobStatus[]> =>
  (await http.get(`/sweeps?limit=${limit}`)).data

export const getSweep = async (jobId: string): Promise<JobStatus> =>
  (await http.get(`/sweeps/${jobId}`)).data

// ── 数据 ─────────────────────────────────────────────
export const fetchData = async (req: DataFetchRequest): Promise<unknown> =>
  (await http.post('/data/fetch', req)).data

export const listCache = async (): Promise<CacheResponse> =>
  (await http.get('/data/cache')).data

export const clearCache = async (): Promise<{ status: string }> =>
  (await http.delete('/data/cache')).data

export const getCacheContent = async (file: string): Promise<CacheContent> =>
  (await http.get('/data/cache/content', { params: { file } })).data

// ── 健康检查 ─────────────────────────────────────────
export const healthCheck = async (): Promise<{ status: string }> =>
  (await http.get('/health')).data

// ── 因子库 / 因子分析 ───────────────────────────────────
export const listFactors = async (): Promise<FactorListResponse> =>
  (await http.get('/factors')).data

export const getFactor = async (name: string): Promise<FactorInfo> =>
  (await http.get(`/factors/${name}`)).data

export const createFactorAnalysis = async (
  req: FactorAnalysisRequest,
): Promise<JobCreated> => (await http.post('/factor-analysis', req)).data

export const listFactorAnalyses = async (limit = 50): Promise<JobStatus[]> =>
  (await http.get(`/factor-analysis?limit=${limit}`)).data

export const getFactorAnalysisJob = async (jobId: string): Promise<JobStatus> =>
  (await http.get(`/factor-analysis/${jobId}`)).data

export const getFactorAnalysisReport = async (jobId: string): Promise<FactorReport> =>
  (await http.get(`/factor-analysis/${jobId}/report`)).data

// ── 多因子诊断 ─────────────────────────────────────────
export const createFactorMatrix = async (
  req: FactorMatrixRequest,
): Promise<JobCreated> => (await http.post('/factor-matrix', req)).data

export const listFactorMatrices = async (limit = 50): Promise<JobStatus[]> =>
  (await http.get(`/factor-matrix?limit=${limit}`)).data

export const getFactorMatrixJob = async (jobId: string): Promise<JobStatus> =>
  (await http.get(`/factor-matrix/${jobId}`)).data

export const getFactorMatrixReport = async (jobId: string): Promise<FactorMatrixReport> =>
  (await http.get(`/factor-matrix/${jobId}/report`)).data

// ── 选股 ───────────────────────────────────────────────
export const listScreenFields = async (): Promise<ScreenFieldsResponse> =>
  (await http.get('/screens/fields')).data

export const listScreenMarkets = async (): Promise<ScreenMarketsResponse> =>
  (await http.get('/screens/markets')).data

export const listScreenJobs = async (limit = 20): Promise<JobStatus[]> =>
  (await http.get(`/screens?limit=${limit}`)).data

export const createScreen = async (req: ScreenRequest): Promise<JobCreated> =>
  (await http.post('/screens', req)).data

export const getScreenJob = async (jobId: string): Promise<JobStatus> =>
  (await http.get(`/screens/${jobId}`)).data

// ── 股票池 ─────────────────────────────────────────────
export const getStockList = async (
  market?: string,
): Promise<UniverseStockListResponse> =>
  (await http.get('/universe/stock-list', { params: { market } })).data

export const listIndexes = async (): Promise<IndexListResponse> =>
  (await http.get('/universe/indexes')).data

export const getIndexComponents = async (
  index: string,
): Promise<IndexComponentsResponse> =>
  (await http.get(`/universe/index-components/${index}`)).data

export const getIndustries = async (
  index?: string,
  symbols?: string[],
): Promise<IndustryListResponse> =>
  (await http.get('/universe/industries', {
    params: { index: index ?? undefined, symbols },
  })).data

// ── 股票搜索 / 详情 ────────────────────────────────────
export const searchStocks = async (
  q: string,
  market?: string,
): Promise<SymbolSearchResponse> =>
  (await http.get('/stocks/search', { params: { q, market } })).data

export const getStockDetail = async (
  symbol: string,
  market?: string,
): Promise<StockDetail> =>
  (await http.get(`/stocks/${symbol}`, { params: { market } })).data

// ── 标的 profile ───────────────────────────────────────
export const listProfiles = async (): Promise<Profile[]> =>
  (await http.get('/profiles')).data

export const createProfile = async (req: ProfileCreate): Promise<Profile> =>
  (await http.post('/profiles', req)).data

export const updateProfile = async (
  id: string,
  req: ProfileUpdate,
): Promise<Profile> => (await http.put(`/profiles/${id}`, req)).data

export const deleteProfile = async (id: string): Promise<void> => {
  await http.delete(`/profiles/${id}`)
}

// ── WebSocket 进度 ───────────────────────────────────
export const subscribeProgress = (
  jobId: string,
  onUpdate: (job: JobStatus) => void,
  onClose?: () => void,
): WebSocket => {
  const url = `${WS_BASE}/backtests/${jobId}/progress`
  const ws = new WebSocket(url)
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data)
      if (data.type !== 'heartbeat') {
        onUpdate(data as JobStatus)
      }
    } catch {
      // ignore non-json
    }
  }
  ws.onclose = () => onClose?.()
  return ws
}
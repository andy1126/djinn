import axios from 'axios'
import type {
  BacktestRequest,
  BacktestReport,
  CacheResponse,
  DataFetchRequest,
  JobCreated,
  JobStatus,
  StrategyListResponse,
  StrategyInfo,
  SweepRequest,
} from '@/types'

// 通过 vite proxy,前缀 /api 转发到 http://localhost:8000
const http = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

// ── 策略 ─────────────────────────────────────────────
export const listStrategies = async (): Promise<StrategyListResponse> =>
  (await http.get('/strategies')).data

export const getStrategy = async (name: string): Promise<StrategyInfo> =>
  (await http.get(`/strategies/${name}`)).data

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

// ── 健康检查 ─────────────────────────────────────────
export const healthCheck = async (): Promise<{ status: string }> =>
  (await http.get('/health')).data

// ── WebSocket 进度 ───────────────────────────────────
export const subscribeProgress = (
  jobId: string,
  onUpdate: (job: JobStatus) => void,
  onClose?: () => void,
): WebSocket => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  // 通过 vite proxy /ws 转发到后端,但 vite ws proxy 不重写路径
  // 这里直接用后端地址(开发环境)
  const url = `${proto}://localhost:8000/backtests/${jobId}/progress`
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
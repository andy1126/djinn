import { create } from 'zustand'
import type { BacktestConfig } from '@/types'

// 默认回测配置(美股 NVDA, MACrossover)
const defaultConfig: BacktestConfig = {
  universe: { symbols: ['NVDA'], benchmark: '^GSPC', market: 'US' },
  period: { start: '2024-01-01', end: '2024-12-31' },
  account: { initial_cash: 100000, currency: 'USD' },
  costs: {
    commission: { type: 'us' },
    slippage: { type: 'fixed_bps', bps: 5 },
    enforce_price_limit: true,
    enforce_suspension: true,
    enforce_lot: true,
  },
  strategy: { name: 'MACrossover', params: { fast: 10, slow: 30 } },
  portfolio: {
    mode: 'single',
    allocation: 'equal',
    rebalance: { period: 'none', threshold: 0, min_hold_days: 0 },
  },
  risk: { max_single_weight: 1.0, max_total_position: 1.0 },
  output: { dir: './results', export: [], report: 'none', rolling_window: 63 },
  adjust: 'backward',
  risk_free_rate: 0.0,
}

interface ConfigStore {
  config: BacktestConfig
  setConfig: (cfg: BacktestConfig) => void
  updateConfig: <K extends keyof BacktestConfig>(key: K, value: BacktestConfig[K]) => void
  reset: () => void
}

export const useConfigStore = create<ConfigStore>((set) => ({
  config: defaultConfig,
  setConfig: (cfg) => set({ config: cfg }),
  updateConfig: (key, value) =>
    set((state) => ({ config: { ...state.config, [key]: value } })),
  reset: () => set({ config: defaultConfig }),
}))
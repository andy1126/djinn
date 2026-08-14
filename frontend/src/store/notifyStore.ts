import { create } from 'zustand'

export interface NotifyItem {
  id: string
  kind: string // backtest / sweep / factor-analysis / ...
  title: string
  status: string
  ts: number
  jobId: string
}

interface NotifyState {
  items: NotifyItem[]
  unread: number
  push: (item: Omit<NotifyItem, 'id' | 'ts'>) => void
  markAllRead: () => void
}

let _seq = 0

// F16:全局通知中心(任务完成提醒)
export const useNotifyStore = create<NotifyState>((set) => ({
  items: [],
  unread: 0,
  push: (item) =>
    set((s) => ({
      items: [{ ...item, id: `n${_seq++}`, ts: Date.now() }, ...s.items].slice(0, 50),
      unread: s.unread + 1,
    })),
  markAllRead: () => set({ unread: 0 }),
}))

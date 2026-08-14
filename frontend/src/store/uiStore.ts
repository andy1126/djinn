import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

interface UiState {
  dark: boolean
  toggle: () => void
}

// F18:暗色模式,persist 到 localStorage
export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      dark: false,
      toggle: () => set((s) => ({ dark: !s.dark })),
    }),
    {
      name: 'djinn-ui',
      storage: createJSONStorage(() => localStorage),
      version: 1,
    },
  ),
)

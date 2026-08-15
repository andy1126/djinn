import { useEffect, useState } from 'react'

/** 返回防抖后的值:输入变化后延迟 ``delay`` ms 才更新(F11 搜索防抖)。 */
export function useDebouncedValue<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return debounced
}

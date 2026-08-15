import { useEffect, useRef } from 'react'
import { useNotifyStore } from '@/store/notifyStore'
import type { JobStatus } from '@/types'

/**
 * F16:任务列表页检测 running/pending → done/error/cancelled 迁移时推送全局通知。
 * 供各任务列表页(回测/扫描/因子分析/矩阵/选股)复用;跨页也能收到(通知入全局 store)。
 */
export function useJobTransitionNotify(
  jobs: JobStatus[] | undefined,
  kind: string,
): void {
  const push = useNotifyStore((s) => s.push)
  const prev = useRef<Record<string, string>>({})
  useEffect(() => {
    const current: Record<string, string> = {}
    for (const j of jobs ?? []) {
      current[j.job_id] = j.status
      const p = prev.current[j.job_id]
      if (
        p &&
        (p === 'running' || p === 'pending') &&
        (j.status === 'done' || j.status === 'error' || j.status === 'cancelled')
      ) {
        push({ kind, title: j.title || j.job_id, status: j.status, jobId: j.job_id })
      }
    }
    prev.current = current
  }, [jobs, kind, push])
}

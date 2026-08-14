import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getBacktest, subscribeProgress } from '@/api/client'
import type { JobStatus } from '@/types'

export interface JobProgress {
  job: JobStatus | null
  /** 进度来源:ws(实时 WebSocket)/ poll(断连后轮询降级)。 */
  via: 'ws' | 'poll'
}

/**
 * 订阅后台任务进度:优先 WebSocket,断连后自动降级为轮询。
 *
 * - 组件卸载 / jobId 变化时关闭旧 WS 连接(避免泄漏);
 * - WS onclose → 标记断连,切换 TanStack Query 轮询(2s 间隔,终态停止)。
 */
export function useJobProgress(jobId: string | null): JobProgress {
  const [job, setJob] = useState<JobStatus | null>(null)
  const [wsDead, setWsDead] = useState(false)

  useEffect(() => {
    if (!jobId) return
    setWsDead(false)
    setJob(null)
    const ws = subscribeProgress(
      jobId,
      (j) => setJob(j),
      () => setWsDead(true),
    )
    return () => ws.close()
  }, [jobId])

  const poll = useQuery({
    queryKey: ['job-poll', jobId],
    queryFn: () => getBacktest(jobId as string),
    enabled: !!jobId && wsDead,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'error'
        ? false
        : 2000,
  })

  return {
    job: wsDead ? (poll.data ?? job) : job,
    via: wsDead ? 'poll' : 'ws',
  }
}

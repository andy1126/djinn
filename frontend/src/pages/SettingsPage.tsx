import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Card, Input, Popconfirm, Space, Switch, Typography, message } from 'antd'
import { healthCheck, listCache, clearCache, purgeJobs, errDetail } from '@/api/client'
import { useUiStore } from '@/store/uiStore'

export default function SettingsPage() {
  const { data: health, error } = useQuery({
    queryKey: ['health'],
    queryFn: healthCheck,
    refetchInterval: 5000,
  })
  const dark = useUiStore((s) => s.dark)
  const toggle = useUiStore((s) => s.toggle)
  const qc = useQueryClient()

  // F5:API Token(存 localStorage,axios 拦截器读取;留空则不用鉴权)
  const [token, setToken] = useState<string>(() => localStorage.getItem('djinn_api_token') ?? '')
  const saveToken = () => {
    localStorage.setItem('djinn_api_token', token.trim())
    message.success('API Token 已保存(刷新后对后续请求生效)')
  }

  // F5:数据维护 —— 缓存大小 + 清理历史任务 + 清空缓存
  const { data: cacheResp } = useQuery({ queryKey: ['cache'], queryFn: listCache })
  const cacheEntries = cacheResp?.entries ?? []
  const cacheRows = cacheEntries.reduce((a, e) => a + (Number(e.rows) || 0), 0)
  const cacheFiles = cacheEntries.filter((e) => !e.error).length

  const doPurge = async () => {
    try {
      const r = await purgeJobs(30)
      message.success(`已清理 ${r.removed} 个过期任务`)
      qc.invalidateQueries({ queryKey: ['backtests'] })
      qc.invalidateQueries({ queryKey: ['sweeps'] })
    } catch (e) {
      message.error(errDetail(e))
    }
  }
  const doClearCache = async () => {
    try {
      await clearCache()
      message.success('缓存已清空')
      qc.invalidateQueries({ queryKey: ['cache'] })
    } catch (e) {
      message.error(errDetail(e))
    }
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="外观">
        <Space>
          <span>暗色模式</span>
          <Switch checked={dark} onChange={toggle} />
        </Space>
      </Card>

      {/* F5:API Token(配合 E8 后端 DJINN_API_TOKEN) */}
      <Card title="API 鉴权">
        <Space direction="vertical" style={{ width: '100%' }}>
          <Typography.Text type="secondary">
            仅当后端设置了 <code>DJINN_API_TOKEN</code> 时需要填写;留空则无需鉴权。
          </Typography.Text>
          <Space.Compact style={{ width: '100%' }}>
            <Input.Password
              placeholder="API Token(可选)"
              value={token}
              onChange={(e) => setToken(e.target.value)}
            />
            <Button type="primary" onClick={saveToken}>保存</Button>
          </Space.Compact>
        </Space>
      </Card>

      {/* F5:数据维护(E6 purge + 缓存) */}
      <Card title="数据维护">
        <Space direction="vertical" style={{ width: '100%' }}>
          <Typography.Text>
            缓存: {cacheFiles} 个文件 / {cacheRows.toLocaleString()} 行
          </Typography.Text>
          <Space>
            <Popconfirm title="清理 30 天前已终态的任务?报告与导出文件将一并删除。" onConfirm={doPurge}>
              <Button>清理历史任务</Button>
            </Popconfirm>
            <Popconfirm title="清空全部本地行情/基本面缓存?下次拉取需重新联网。" onConfirm={doClearCache}>
              <Button danger>清空缓存</Button>
            </Popconfirm>
          </Space>
        </Space>
      </Card>

      <Card title="系统状态">
        {/* F5:错误内联展示,去掉 5s 一次的 message 弹窗刷屏 */}
        {error ? (
          <Alert type="error" showIcon message="后端服务不可达" />
        ) : (
          <p>后端: {health?.status === 'healthy' ? '🟢 正常' : '🔴 不可达'}</p>
        )}
        <p>API 地址: <code>localhost:8000</code></p>
      </Card>
    </Space>
  )
}

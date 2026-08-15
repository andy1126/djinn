import { Alert, Button, Space } from 'antd'

interface Props {
  error: unknown
  retry?: () => void
}

/** 列表查询失败时的内联错误提示(F7),替代静默空表。 */
export default function QueryErrorAlert({ error, retry }: Props) {
  const msg = (error as { friendlyMessage?: string; message?: string })?.friendlyMessage
    || (error as { message?: string })?.message
    || '请求失败'
  return (
    <Alert
      type="error"
      showIcon
      message="加载失败"
      description={
        <Space direction="vertical" size={4}>
          <span>{msg}</span>
          {retry && (
            <Button size="small" onClick={retry}>重试</Button>
          )}
        </Space>
      }
    />
  )
}

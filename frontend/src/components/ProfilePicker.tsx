import { useQuery } from '@tanstack/react-query'
import { Button, Select, Space, Tooltip } from 'antd'
import { SettingOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { listProfiles } from '@/api/client'
import type { Profile } from '@/types'

interface Props {
  onSelect: (profile: Profile) => void
  placeholder?: string
}

/**
 * profile 快速选择下拉:选中后把 profile 的标的(及可选市场)交给调用方。
 * 旁边附「管理」按钮跳转到 /profiles。
 */
export default function ProfilePicker({ onSelect, placeholder = '从 Profile 载入标的' }: Props) {
  const navigate = useNavigate()
  const { data: profiles } = useQuery({ queryKey: ['profiles'], queryFn: listProfiles })

  const options = (profiles || []).map((p) => ({
    value: p.profile_id,
    label: `${p.name} (${p.symbols.length})`,
  }))

  const onChange = (id: string) => {
    const p = (profiles || []).find((x) => x.profile_id === id)
    if (p) onSelect(p)
  }

  return (
    <Space.Compact style={{ width: '100%' }}>
      <Select
        style={{ flex: 1 }}
        placeholder={placeholder}
        options={options}
        onChange={onChange}
        showSearch
        optionFilterProp="label"
        notFoundContent={profiles && profiles.length === 0 ? '暂无 Profile,点右侧管理创建' : undefined}
        allowClear
      />
      <Tooltip title="管理 Profile">
        <Button icon={<SettingOutlined />} onClick={() => navigate('/profiles')} />
      </Tooltip>
    </Space.Compact>
  )
}

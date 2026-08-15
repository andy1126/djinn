import { Form, InputNumber, Select, Typography } from 'antd'
import type { ParamSchema } from '@/types'

/** 因子/策略参数的动态表单控件(F9 共享):int/float/str/bool/select。 */
export default function ParamField({
  p,
  value,
  onSet,
}: {
  p: ParamSchema
  value: unknown
  onSet: (v: unknown) => void
}) {
  const label = (
    <span>
      <b>{p.name}</b> <Typography.Text type="secondary">{p.description || ''}</Typography.Text>
    </span>
  )
  if (p.choices && p.choices.length > 0) {
    return (
      <Form.Item key={p.name} label={label}>
        <Select
          value={(value ?? p.default) as string | number | boolean}
          onChange={onSet}
          options={p.choices.map((c) => ({ label: String(c), value: c }))}
          style={{ width: '100%' }}
        />
      </Form.Item>
    )
  }
  if (p.type === 'bool' || p.type === 'boolean') {
    return (
      <Form.Item key={p.name} label={label}>
        <Select
          value={(value ?? p.default) as boolean}
          onChange={onSet}
          options={[{ label: 'true', value: true }, { label: 'false', value: false }]}
          style={{ width: '100%' }}
        />
      </Form.Item>
    )
  }
  if (p.type === 'str' || p.type === 'string' || p.type === 'NoneType') {
    return (
      <Form.Item key={p.name} label={label}>
        <Select
          value={(value ?? p.default) as string | undefined}
          onChange={onSet}
          style={{ width: '100%' }}
          placeholder="请输入"
          showSearch
          options={[]}
        />
      </Form.Item>
    )
  }
  return (
    <Form.Item key={p.name} label={label}>
      <InputNumber
        value={value != null ? Number(value) : Number(p.default)}
        onChange={(v) => onSet(v ?? 0)}
        min={p.min != null ? Number(p.min) : undefined}
        max={p.max != null ? Number(p.max) : undefined}
        style={{ width: '100%' }}
      />
    </Form.Item>
  )
}

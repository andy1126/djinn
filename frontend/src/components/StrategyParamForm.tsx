import { Form, InputNumber, Select, Input, Slider, Tooltip } from 'antd'
import type { ParamSchema } from '@/types'

interface Props {
  schema: ParamSchema[]
  value: Record<string, number | string | boolean | null>
  onChange: (value: Record<string, number | string | boolean | null>) => void
}

export default function StrategyParamForm({ schema, value, onChange }: Props) {
  const update = (name: string, v: any) => onChange({ ...value, [name]: v })

  const renderField = (p: ParamSchema) => {
    const label = p.description ? <Tooltip title={p.description}>{p.name}</Tooltip> : p.name
    const v = value[p.name] ?? p.default

    if (p.choices && p.choices.length > 0) {
      return (
        <Select
          value={v as any}
          onChange={(val) => update(p.name, val)}
          options={p.choices.map((c) => ({ label: String(c), value: c }))}
          style={{ width: '100%' }}
        />
      )
    }
    if (p.type === 'bool' || p.type === 'boolean') {
      return (
        <Select
          value={v as any}
          onChange={(val) => update(p.name, val)}
          options={[{ label: 'true', value: true }, { label: 'false', value: false }]}
          style={{ width: '100%' }}
        />
      )
    }
    if (p.type === 'str' || p.type === 'string' || p.type === 'NoneType') {
      return <Input value={v != null ? String(v) : ''} onChange={(e) => update(p.name, e.target.value)} />
    }
    // 数值类型:若有 min/max 且范围合理,用 Slider;否则用 InputNumber
    if (p.min != null && p.max != null && p.max <= p.min + 200) {
      return (
        <div>
          <Slider
            min={p.min as number}
            max={p.max as number}
            step={1}
            value={Number(v ?? p.default ?? p.min)}
            onChange={(val) => update(p.name, val)}
            marks={{ [p.min as number]: String(p.min), [p.max as number]: String(p.max) }}
          />
        </div>
      )
    }
    return (
      <InputNumber
        value={v != null ? Number(v) : null}
        onChange={(val) => update(p.name, val ?? 0)}
        min={p.min != null ? (p.min as number) : undefined}
        max={p.max != null ? (p.max as number) : undefined}
        style={{ width: '100%' }}
      />
    )
  }

  return (
    <Form layout="vertical" size="small">
      {schema.map((p) => (
        <Form.Item key={p.name} label={p.name} tooltip={p.description}>
          {renderField(p)}
        </Form.Item>
      ))}
    </Form>
  )
}
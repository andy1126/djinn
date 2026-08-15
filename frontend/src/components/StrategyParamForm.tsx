import { Form, InputNumber, Select, Input, Slider } from 'antd'
import type { ParamSchema } from '@/types'

/** 取一个数的小数位数(用于 Slider 拖拽后消除浮点漂移)。 */
function decimalPlaces(n: number): number {
  const s = String(n)
  const i = s.indexOf('.')
  return i === -1 ? 0 : s.length - i - 1
}

interface Props {
  schema: ParamSchema[]
  value: Record<string, number | string | boolean | null>
  onChange: (value: Record<string, number | string | boolean | null>) => void
}

export default function StrategyParamForm({ schema, value, onChange }: Props) {
  const update = (name: string, v: number | string | boolean) => onChange({ ...value, [name]: v })

  const renderField = (p: ParamSchema) => {
    const v = value[p.name] ?? p.default

    if (p.choices && p.choices.length > 0) {
      return (
        <Select
          value={v as string | number | boolean}
          onChange={(val) => update(p.name, val)}
          options={p.choices.map((c) => ({ label: String(c), value: c }))}
          style={{ width: '100%' }}
        />
      )
    }
    if (p.type === 'bool' || p.type === 'boolean') {
      return (
        <Select
          value={v as string | number | boolean}
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
      const min = p.min as number
      const max = p.max as number
      const range = max - min
      const isInt = p.type === 'int'
      // int 用整数步长;float 按区间均分 100 份取小数步长,避免 step=1 跳穿小数区间
      const step = isInt ? 1 : range / 100
      // float 拖拽后按参数最小粒度取整,消除 0.0500…01 这类浮点漂移
      const decimals = isInt
        ? 0
        : Math.min(
            6,
            Math.max(
              decimalPlaces(min),
              decimalPlaces(max),
              decimalPlaces(typeof p.default === 'number' ? p.default : 0)
            )
          )
      return (
        <Slider
          min={min}
          max={max}
          step={step}
          value={Number(v ?? p.default ?? min)}
          onChange={(val) =>
            update(p.name, isInt ? (val as number) : Number((val as number).toFixed(decimals)))
          }
        />
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
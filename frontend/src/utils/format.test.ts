import { describe, expect, it } from 'vitest'
import { fmtNum, fmtPct, formatCompact } from './format'

describe('format 工具(F13)', () => {
  it('fmtPct 小数转百分比', () => {
    expect(fmtPct(0.123)).toBe('12.30%')
    expect(fmtPct(0.05, 1)).toBe('5.0%')
    expect(fmtPct(-0.2)).toBe('-20.00%')
  })

  it('fmtNum 保留小数', () => {
    expect(fmtNum(3.14159)).toBe('3.14')
    expect(fmtNum(1, 3)).toBe('1.000')
  })

  it('formatCompact 大额缩写', () => {
    expect(formatCompact(2.5e8)).toBe('2.50 亿')
    expect(formatCompact(3.6e4)).toBe('3.60 万')
    expect(formatCompact(42)).toBe('42.00')
    expect(formatCompact(null)).toBe('—')
    expect(formatCompact(NaN)).toBe('—')
  })
})

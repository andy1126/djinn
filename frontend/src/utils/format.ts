/** 数值格式化工具(F9 共享):百分比 / 数值 / 大额缩写。 */

/** 小数转百分数字符串(0.123 → "12.30%")。 */
export const fmtPct = (v: number, digits = 2): string =>
  (v * 100).toFixed(digits) + '%'

/** 数值保留 ``digits`` 位小数。 */
export const fmtNum = (v: number, digits = 2): string => v.toFixed(digits)

/** 大额数值缩写(≥1e8 → "x.xx 亿",≥1e4 → "x.xx 万",A 股习惯)。 */
export function formatCompact(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  const abs = Math.abs(v)
  if (abs >= 1e8) return (v / 1e8).toFixed(2) + ' 亿'
  if (abs >= 1e4) return (v / 1e4).toFixed(2) + ' 万'
  return v.toFixed(2)
}

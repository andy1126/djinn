/** 指标对比方向表(F15):与后端 REVERSE_MIN_TARGETS 同步。 */

/** 越大越好的指标(含 max_drawdown —— 后端存为 ≤0 负值,越接近 0 越好)。 */
export const METRIC_BETTER_HIGHER = new Set([
  'total_return',
  'annual_return',
  'sharpe',
  'sortino',
  'calmar',
  'win_rate',
  'max_drawdown',
])

/** 越小越好的指标(仅波动类,升序最优)。 */
export const METRIC_BETTER_LOWER = new Set(['volatility', 'annual_volatility'])

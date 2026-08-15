import { theme } from 'antd'

/**
 * F18:暗色模式下的 ECharts 配色(读 antd token,随主题切换自适应)。
 * ECharts 默认文字为深色,暗色布局下不可读;各图表把返回值并入 option。
 */
export function useChartTheme() {
  const { token } = theme.useToken()
  return {
    // 全局文字(legend / title / tooltip 之外)
    textStyle: { color: token.colorText },
    legend: { textStyle: { color: token.colorText }, inactiveColor: token.colorTextTertiary },
    tooltip: { backgroundColor: token.colorBgElevated, textStyle: { color: token.colorText }, borderColor: token.colorSplit },
    xAxis: {
      axisLabel: { color: token.colorTextSecondary },
      axisLine: { lineStyle: { color: token.colorBorder } },
    },
    yAxis: {
      axisLabel: { color: token.colorTextSecondary },
      axisLine: { lineStyle: { color: token.colorBorder } },
      splitLine: { lineStyle: { color: token.colorSplit } },
    },
  }
}

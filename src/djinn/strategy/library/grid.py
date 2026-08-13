"""网格策略:价格每下跌一格加仓,回升后按档位减仓。"""

from __future__ import annotations

from djinn.strategy.base import Context, Strategy, param


class Grid(Strategy):
    """网格交易(简化版:按价格相对基准价的档位定目标仓位)。

    以首根收盘价为基准,价格每跌破 ``step`` 一档即加一格仓位(最多 ``num_levels``
    格),回升到基准上方则逐档减仓直至清仓。适合震荡行情。
    """

    step = param(0.05, min=0.005, max=0.2, description="每格幅度(比例)")
    num_levels = param(5, min=1, max=20, description="最大档位数")
    unit_weight = param(0.1, min=0.01, max=0.5, description="每格仓位权重")

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self._base: dict[str, float] = {}
        self._level: dict[str, int] = {}

    def on_bar(self, ctx: Context) -> None:
        for s in ctx.data.symbols:
            df = ctx.data[s]
            if len(df) == 0:
                continue
            price = float(df["close"].iloc[-1])
            base = self._base.get(s)
            if base is None:
                self._base[s] = price
                self._level[s] = 0
                continue  # 首根定基准,不交易
            # 相对基准下跌的档位数(向下为正)
            k = int((base - price) / (base * float(self.step)))
            k = max(0, min(int(self.num_levels), k))
            if k != self._level.get(s):
                self._level[s] = k
                ctx.order_target_percent(s, min(1.0, k * float(self.unit_weight)))

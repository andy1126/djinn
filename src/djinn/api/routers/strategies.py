"""策略路由:列表/schema。"""

from fastapi import APIRouter, HTTPException

from djinn.api.schemas import StrategyInfo, StrategyListResponse
from djinn.strategy import (
    DCA,
    MACrossover,
    Momentum,
    RSIReversal,
    param_schema,
)

router = APIRouter(prefix="/strategies", tags=["strategies"])

STRATEGIES = {
    "MACrossover": (MACrossover, "双均线交叉(fast/slow)"),
    "RSIReversal": (RSIReversal, "RSI 超买超卖反转"),
    "Momentum": (Momentum, "N 日通道突破"),
    "DCA": (DCA, "定期定额买入"),
}


@router.get("", response_model=StrategyListResponse)
async def list_strategies() -> StrategyListResponse:
    """列出所有内置策略及其参数 schema。"""
    items = []
    for name, (cls, desc) in STRATEGIES.items():
        params = param_schema(cls)
        items.append(
            StrategyInfo(
                name=name,
                description=desc,
                params=[p.to_dict() for p in params],
            )
        )
    return StrategyListResponse(strategies=items)


@router.get("/{name}", response_model=StrategyInfo)
async def get_strategy(name: str) -> StrategyInfo:
    """获取单个策略的详细信息。"""
    if name not in STRATEGIES:
        raise HTTPException(status_code=404, detail=f"策略 {name} 不存在")
    cls, desc = STRATEGIES[name]
    params = param_schema(cls)
    return StrategyInfo(
        name=name,
        description=desc,
        params=[p.to_dict() for p in params],
    )

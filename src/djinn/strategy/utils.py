"""策略层内部工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["state_from_signals"]


def state_from_signals(sig: pd.Series) -> pd.Series:
    """把稀疏 +1/-1 信号序列转成持仓状态(前向填充,0 起始)。

    例如 ``[0,0,1,0,0,-1,0]`` → ``[0,0,1,1,1,-1,-1]``。
    金叉后维持多头直到死叉;无 pandas downcast 警告。
    """
    f = sig.astype(float)
    f = f.where(f != 0, np.nan)
    return f.ffill().fillna(0.0).astype(int)

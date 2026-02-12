"""
Djinn 市场数据模型。

这个模块定义了市场数据相关的数据模型和数据结构。
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

from pydantic import BaseModel, Field, validator, root_validator

from ..utils.validation import Validator
from ..utils.date_utils import DateUtils
from ..utils.exceptions import ValidationError


class MarketDataType(str, Enum):
    """市场数据类型枚举。"""

    OHLCV = "ohlcv"  # 开高低收成交量
    FUNDAMENTAL = "fundamental"  # 基本面数据
    INTRADAY = "intraday"  # 日内数据
    OPTIONS = "options"  # 期权数据
    FUTURES = "futures"  # 期货数据


class AdjustmentType(str, Enum):
    """价格调整类型枚举。"""

    RAW = "raw"  # 原始价格
    ADJ = "adj"  # 调整后价格（复权）
    SPLIT = "split"  # 仅考虑拆股
    DIVIDEND = "dividend"  # 仅考虑股息


class MarketStatus(str, Enum):
    """市场状态枚举。"""

    OPEN = "open"  # 开市
    CLOSED = "closed"  # 闭市
    PRE_MARKET = "pre_market"  # 盘前
    AFTER_HOURS = "after_hours"  # 盘后
    HOLIDAY = "holiday"  # 节假日


@dataclass
class OHLCV:
    """OHLCV 数据点。"""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: Optional[float] = None
    dividends: Optional[float] = None
    stock_splits: Optional[float] = None

    def __post_init__(self):
        """数据验证。"""
        # 验证价格数据的合理性
        if self.high < self.low:
            raise ValidationError(
                "最高价不能低于最低价",
                field="high/low",
                value=f"high={self.high}, low={self.low}",
                expected="high >= low",
            )

        if self.close > self.high:
            raise ValidationError(
                "收盘价不能高于最高价",
                field="close",
                value=self.close,
                expected=f"<= {self.high}",
            )

        if self.close < self.low:
            raise ValidationError(
                "收盘价不能低于最低价",
                field="close",
                value=self.close,
                expected=f">= {self.low}",
            )

        if self.open > self.high:
            raise ValidationError(
                "开盘价不能高于最高价",
                field="open",
                value=self.open,
                expected=f"<= {self.high}",
            )

        if self.open < self.low:
            raise ValidationError(
                "开盘价不能低于最低价",
                field="open",
                value=self.open,
                expected=f">= {self.low}",
            )

        if self.volume < 0:
            raise ValidationError(
                "成交量不能为负数",
                field="volume",
                value=self.volume,
                expected=">= 0",
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCV":
        """从字典创建。"""
        return cls(**data)

    @classmethod
    def from_series(cls, series: pd.Series) -> "OHLCV":
        """从 pandas Series 创建。"""
        return cls(
            timestamp=series.name if isinstance(series.name, datetime) else pd.to_datetime(series.name),
            open=series.get("open", series.get("Open", 0.0)),
            high=series.get("high", series.get("High", 0.0)),
            low=series.get("low", series.get("Low", 0.0)),
            close=series.get("close", series.get("Close", 0.0)),
            volume=series.get("volume", series.get("Volume", 0.0)),
            adj_close=series.get("adj_close", series.get("Adj Close")),
            dividends=series.get("dividends", 0.0),
            stock_splits=series.get("stock_splits", 0.0),
        )


@dataclass
class FundamentalData:
    """基本面数据。"""

    symbol: str
    date: date
    # 财务指标
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    # 盈利能力
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    # 财务健康
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    # 运营效率
    roe: Optional[float] = None  # 净资产收益率
    roa: Optional[float] = None  # 总资产收益率
    roi: Optional[float] = None  # 投资回报率

    def __post_init__(self):
        """数据验证。"""
        # 验证股票代码
        self.symbol = Validator.validate_stock_symbol(self.symbol, "symbol")

        # 验证日期
        if isinstance(self.date, str):
            self.date = DateUtils.parse_date(self.date)
        elif isinstance(self.date, datetime):
            self.date = self.date.date()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FundamentalData":
        """从字典创建。"""
        return cls(**data)


@dataclass
class MarketData:
    """市场数据容器。"""

    symbol: str
    data_type: MarketDataType
    data: Union[pd.DataFrame, List[OHLCV], List[FundamentalData]]
    interval: str = "1d"
    adjustment: AdjustmentType = AdjustmentType.ADJ
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """数据验证。"""
        # 验证股票代码
        self.symbol = Validator.validate_stock_symbol(self.symbol, "symbol")

        # 验证数据
        if isinstance(self.data, pd.DataFrame):
            # 验证 DataFrame 包含必要的列
            if self.data_type == MarketDataType.OHLCV:
                required_columns = ["open", "high", "low", "close", "volume"]
                missing = [col for col in required_columns if col not in self.data.columns]
                if missing:
                    raise ValidationError(
                        f"OHLCV 数据缺少必需的列",
                        field="data.columns",
                        value=list(self.data.columns),
                        expected=f"包含列: {', '.join(required_columns)}",
                        details={"missing_columns": missing},
                    )
        elif isinstance(self.data, list):
            # 验证列表元素类型
            if self.data_type == MarketDataType.OHLCV:
                if not all(isinstance(item, OHLCV) for item in self.data):
                    raise ValidationError(
                        "OHLCV 数据列表包含非 OHLCV 对象",
                        field="data",
                        value=[type(item).__name__ for item in self.data],
                        expected="List[OHLCV]",
                    )
            elif self.data_type == MarketDataType.FUNDAMENTAL:
                if not all(isinstance(item, FundamentalData) for item in self.data):
                    raise ValidationError(
                        "基本面数据列表包含非 FundamentalData 对象",
                        field="data",
                        value=[type(item).__name__ for item in self.data],
                        expected="List[FundamentalData]",
                    )

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 pandas DataFrame。"""
        if isinstance(self.data, pd.DataFrame):
            return self.data.copy()

        elif isinstance(self.data, list):
            if self.data_type == MarketDataType.OHLCV:
                records = [item.to_dict() for item in self.data]
                df = pd.DataFrame(records)
                if not df.empty:
                    df.set_index("timestamp", inplace=True)
                return df

            elif self.data_type == MarketDataType.FUNDAMENTAL:
                records = [item.to_dict() for item in self.data]
                df = pd.DataFrame(records)
                if not df.empty:
                    df.set_index("date", inplace=True)
                return df

        raise ValidationError(
            f"无法将 {self.data_type} 数据转换为 DataFrame",
            field="data",
            value=type(self.data).__name__,
            expected="pd.DataFrame 或可转换的列表",
        )

    def get_ohlcv_list(self) -> List[OHLCV]:
        """获取 OHLCV 数据列表。"""
        if self.data_type != MarketDataType.OHLCV:
            raise ValidationError(
                f"数据类型不是 OHLCV",
                field="data_type",
                value=self.data_type.value,
                expected=MarketDataType.OHLCV.value,
            )

        if isinstance(self.data, list):
            return self.data
        elif isinstance(self.data, pd.DataFrame):
            return [
                OHLCV.from_series(row)
                for _, row in self.data.iterrows()
            ]
        else:
            raise ValidationError(
                "无法获取 OHLCV 列表",
                field="data",
                value=type(self.data).__name__,
                expected="List[OHLCV] 或 pd.DataFrame",
            )

    def get_fundamental_list(self) -> List[FundamentalData]:
        """获取基本面数据列表。"""
        if self.data_type != MarketDataType.FUNDAMENTAL:
            raise ValidationError(
                f"数据类型不是 FUNDAMENTAL",
                field="data_type",
                value=self.data_type.value,
                expected=MarketDataType.FUNDAMENTAL.value,
            )

        if isinstance(self.data, list):
            return self.data
        elif isinstance(self.data, pd.DataFrame):
            # 从 DataFrame 转换
            records = self.data.to_dict("records")
            return [
                FundamentalData.from_dict(record)
                for record in records
            ]
        else:
            raise ValidationError(
                "无法获取基本面数据列表",
                field="data",
                value=type(self.data).__name__,
                expected="List[FundamentalData] 或 pd.DataFrame",
            )

    @property
    def start_date(self) -> Optional[datetime]:
        """获取数据开始日期。"""
        if isinstance(self.data, pd.DataFrame):
            if not self.data.empty:
                return self.data.index[0] if hasattr(self.data.index[0], "timestamp") else pd.to_datetime(self.data.index[0])
        elif isinstance(self.data, list) and self.data:
            if self.data_type == MarketDataType.OHLCV:
                return self.data[0].timestamp
            elif self.data_type == MarketDataType.FUNDAMENTAL:
                return datetime.combine(self.data[0].date, datetime.min.time())
        return None

    @property
    def end_date(self) -> Optional[datetime]:
        """获取数据结束日期。"""
        if isinstance(self.data, pd.DataFrame):
            if not self.data.empty:
                return self.data.index[-1] if hasattr(self.data.index[-1], "timestamp") else pd.to_datetime(self.data.index[-1])
        elif isinstance(self.data, list) and self.data:
            if self.data_type == MarketDataType.OHLCV:
                return self.data[-1].timestamp
            elif self.data_type == MarketDataType.FUNDAMENTAL:
                return datetime.combine(self.data[-1].date, datetime.min.time())
        return None

    @property
    def length(self) -> int:
        """获取数据长度。"""
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        elif isinstance(self.data, list):
            return len(self.data)
        return 0

    def filter_by_date(
        self,
        start_date: Union[datetime, date, str],
        end_date: Union[datetime, date, str],
    ) -> "MarketData":
        """按日期过滤数据。"""
        # 解析日期
        if isinstance(start_date, str):
            start_date = DateUtils.parse_date(start_date)
        if isinstance(end_date, str):
            end_date = DateUtils.parse_date(end_date)

        # 转换为 datetime 以便比较
        if isinstance(start_date, date):
            start_dt = datetime.combine(start_date, datetime.min.time())
        else:
            start_dt = start_date

        if isinstance(end_date, date):
            end_dt = datetime.combine(end_date, datetime.max.time())
        else:
            end_dt = end_date

        if isinstance(self.data, pd.DataFrame):
            # 过滤 DataFrame
            mask = (self.data.index >= start_dt) & (self.data.index <= end_dt)
            filtered_data = self.data[mask].copy()
        elif isinstance(self.data, list):
            # 过滤列表
            if self.data_type == MarketDataType.OHLCV:
                filtered_data = [
                    item for item in self.data
                    if start_dt <= item.timestamp <= end_dt
                ]
            elif self.data_type == MarketDataType.FUNDAMENTAL:
                filtered_data = [
                    item for item in self.data
                    if start_dt <= datetime.combine(item.date, datetime.min.time()) <= end_dt
                ]
            else:
                filtered_data = self.data
        else:
            filtered_data = self.data

        return MarketData(
            symbol=self.symbol,
            data_type=self.data_type,
            data=filtered_data,
            interval=self.interval,
            adjustment=self.adjustment,
            metadata=self.metadata.copy(),
        )


class MarketDataRequest(BaseModel):
    """市场数据请求模型。"""

    symbol: str = Field(..., description="股票代码")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)")
    interval: str = Field("1d", description="数据间隔")
    adjustment: AdjustmentType = Field(AdjustmentType.ADJ, description="价格调整类型")
    data_type: MarketDataType = Field(MarketDataType.OHLCV, description="数据类型")

    @validator("symbol")
    def validate_symbol(cls, v):
        """验证股票代码。"""
        return Validator.validate_stock_symbol(v, "symbol")

    @validator("start_date", "end_date")
    def validate_date(cls, v):
        """验证日期格式。"""
        try:
            DateUtils.parse_date(v)
            return v
        except Exception as e:
            raise ValidationError(
                f"无效的日期格式: {v}",
                field="date",
                value=v,
                expected="YYYY-MM-DD 格式",
                details={"error": str(e)},
            )

    @validator("interval")
    def validate_interval(cls, v):
        """验证数据间隔。"""
        valid_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
        if v not in valid_intervals:
            raise ValidationError(
                f"无效的数据间隔: {v}",
                field="interval",
                value=v,
                expected=f"以下值之一: {', '.join(valid_intervals)}",
            )
        return v

    @root_validator(skip_on_failure=True)
    def validate_date_range(cls, values):
        """验证日期范围。"""
        start_date = values.get("start_date")
        end_date = values.get("end_date")

        if start_date and end_date:
            start = DateUtils.parse_date(start_date)
            end = DateUtils.parse_date(end_date)

            if end <= start:
                raise ValidationError(
                    "结束日期必须晚于开始日期",
                    field="end_date",
                    value=end_date,
                    expected=f"> {start_date}",
                )

            # 检查日期范围是否合理
            days_diff = (end - start).days
            if days_diff > 365 * 20:  # 20年
                raise ValidationError(
                    "日期范围超过20年，请缩小范围",
                    field="date_range",
                    value=f"{start_date} 到 {end_date}",
                    expected="<= 20年",
                )

        return values


# 导出
__all__ = [
    "MarketDataType",
    "AdjustmentType",
    "MarketStatus",
    "OHLCV",
    "FundamentalData",
    "MarketData",
    "MarketDataRequest",
]
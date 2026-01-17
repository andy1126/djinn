"""
Djinn 数据验证模块。

这个模块提供了数据验证和清洗相关的功能。
"""

import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, date
from decimal import Decimal, InvalidOperation

from .logger import logger
from .exceptions import ValidationError
from .date_utils import DateUtils


class Validator:
    """数据验证器。"""

    @staticmethod
    def validate_not_none(value: Any, field: str = "value") -> None:
        """
        验证值不为 None。

        Args:
            value: 要验证的值
            field: 字段名称

        Raises:
            ValidationError: 如果值为 None
        """
        if value is None:
            raise ValidationError(
                f"字段 '{field}' 不能为 None",
                field=field,
                value=value,
                expected="非 None 值",
            )

    @staticmethod
    def validate_not_empty(value: Any, field: str = "value") -> None:
        """
        验证值不为空。

        Args:
            value: 要验证的值
            field: 字段名称

        Raises:
            ValidationError: 如果值为空
        """
        Validator.validate_not_none(value, field)

        if isinstance(value, str) and not value.strip():
            raise ValidationError(
                f"字段 '{field}' 不能为空字符串",
                field=field,
                value=value,
                expected="非空字符串",
            )
        elif hasattr(value, "__len__") and len(value) == 0:
            raise ValidationError(
                f"字段 '{field}' 不能为空集合",
                field=field,
                value=value,
                expected="非空集合",
            )

    @staticmethod
    def validate_type(
        value: Any,
        expected_type: Union[type, Tuple[type, ...]],
        field: str = "value",
    ) -> None:
        """
        验证值的类型。

        Args:
            value: 要验证的值
            expected_type: 期望的类型或类型元组
            field: 字段名称

        Raises:
            ValidationError: 如果类型不匹配
        """
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                type_names = [t.__name__ for t in expected_type]
                expected_str = f"以下类型之一: {', '.join(type_names)}"
            else:
                expected_str = expected_type.__name__

            raise ValidationError(
                f"字段 '{field}' 的类型不匹配",
                field=field,
                value=type(value).__name__,
                expected=expected_str,
            )

    @staticmethod
    def validate_numeric(
        value: Any,
        field: str = "value",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
    ) -> float:
        """
        验证数值。

        Args:
            value: 要验证的值
            field: 字段名称
            min_value: 最小值（包含）
            max_value: 最大值（包含）
            allow_nan: 是否允许 NaN
            allow_inf: 是否允许无穷大

        Returns:
            float: 验证后的数值

        Raises:
            ValidationError: 如果验证失败
        """
        # 尝试转换为数值
        try:
            if isinstance(value, (int, float, Decimal)):
                num = float(value)
            elif isinstance(value, str):
                num = float(value)
            else:
                raise ValueError(f"无法转换为数值: {value}")
        except (ValueError, TypeError, InvalidOperation) as e:
            raise ValidationError(
                f"字段 '{field}' 不是有效的数值",
                field=field,
                value=value,
                expected="有效的数值",
                details={"error": str(e)},
            )

        # 检查 NaN
        if np.isnan(num) and not allow_nan:
            raise ValidationError(
                f"字段 '{field}' 不能为 NaN",
                field=field,
                value=value,
                expected="非 NaN 数值",
            )

        # 检查无穷大
        if np.isinf(num) and not allow_inf:
            raise ValidationError(
                f"字段 '{field}' 不能为无穷大",
                field=field,
                value=value,
                expected="有限数值",
            )

        # 检查范围
        if min_value is not None and num < min_value:
            raise ValidationError(
                f"字段 '{field}' 必须大于等于 {min_value}",
                field=field,
                value=num,
                expected=f">= {min_value}",
            )

        if max_value is not None and num > max_value:
            raise ValidationError(
                f"字段 '{field}' 必须小于等于 {max_value}",
                field=field,
                value=num,
                expected=f"<= {max_value}",
            )

        return num

    @staticmethod
    def validate_string(
        value: Any,
        field: str = "value",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_values: Optional[List[str]] = None,
    ) -> str:
        """
        验证字符串。

        Args:
            value: 要验证的值
            field: 字段名称
            min_length: 最小长度
            max_length: 最大长度
            pattern: 正则表达式模式
            allowed_values: 允许的值列表

        Returns:
            str: 验证后的字符串

        Raises:
            ValidationError: 如果验证失败
        """
        # 转换为字符串
        if not isinstance(value, str):
            try:
                str_value = str(value)
            except Exception as e:
                raise ValidationError(
                    f"字段 '{field}' 无法转换为字符串",
                    field=field,
                    value=value,
                    expected="可转换为字符串的值",
                    details={"error": str(e)},
                )
        else:
            str_value = value

        # 检查长度
        if min_length is not None and len(str_value) < min_length:
            raise ValidationError(
                f"字段 '{field}' 的长度必须至少为 {min_length}",
                field=field,
                value=str_value,
                expected=f"长度 >= {min_length}",
            )

        if max_length is not None and len(str_value) > max_length:
            raise ValidationError(
                f"字段 '{field}' 的长度不能超过 {max_length}",
                field=field,
                value=str_value,
                expected=f"长度 <= {max_length}",
            )

        # 检查正则表达式
        if pattern is not None:
            if not re.match(pattern, str_value):
                raise ValidationError(
                    f"字段 '{field}' 不匹配模式: {pattern}",
                    field=field,
                    value=str_value,
                    expected=f"匹配模式: {pattern}",
                )

        # 检查允许的值
        if allowed_values is not None and str_value not in allowed_values:
            raise ValidationError(
                f"字段 '{field}' 的值不在允许的列表中",
                field=field,
                value=str_value,
                expected=f"以下值之一: {', '.join(allowed_values)}",
            )

        return str_value

    @staticmethod
    def validate_date(
        value: Any,
        field: str = "value",
        min_date: Optional[Union[date, datetime, str]] = None,
        max_date: Optional[Union[date, datetime, str]] = None,
        format_str: Optional[str] = None,
    ) -> date:
        """
        验证日期。

        Args:
            value: 要验证的值
            field: 字段名称
            min_date: 最小日期
            max_date: 最大日期
            format_str: 日期格式字符串

        Returns:
            date: 验证后的日期

        Raises:
            ValidationError: 如果验证失败
        """
        # 解析日期
        try:
            if isinstance(value, (date, datetime)):
                date_obj = value if isinstance(value, date) else value.date()
            else:
                date_obj = DateUtils.parse_date(value, format_str, raise_error=True)
        except Exception as e:
            raise ValidationError(
                f"字段 '{field}' 不是有效的日期",
                field=field,
                value=value,
                expected="有效的日期",
                details={"error": str(e)},
            )

        # 解析最小日期
        if min_date is not None:
            try:
                if isinstance(min_date, (date, datetime)):
                    min_date_obj = (
                        min_date if isinstance(min_date, date) else min_date.date()
                    )
                else:
                    min_date_obj = DateUtils.parse_date(min_date, format_str)
            except Exception as e:
                raise ValidationError(
                    f"最小日期参数无效",
                    field="min_date",
                    value=min_date,
                    expected="有效的日期",
                    details={"error": str(e)},
                )

            if date_obj < min_date_obj:
                raise ValidationError(
                    f"字段 '{field}' 必须晚于或等于 {min_date_obj}",
                    field=field,
                    value=date_obj,
                    expected=f">= {min_date_obj}",
                )

        # 解析最大日期
        if max_date is not None:
            try:
                if isinstance(max_date, (date, datetime)):
                    max_date_obj = (
                        max_date if isinstance(max_date, date) else max_date.date()
                    )
                else:
                    max_date_obj = DateUtils.parse_date(max_date, format_str)
            except Exception as e:
                raise ValidationError(
                    f"最大日期参数无效",
                    field="max_date",
                    value=max_date,
                    expected="有效的日期",
                    details={"error": str(e)},
                )

            if date_obj > max_date_obj:
                raise ValidationError(
                    f"字段 '{field}' 必须早于或等于 {max_date_obj}",
                    field=field,
                    value=date_obj,
                    expected=f"<= {max_date_obj}",
                )

        return date_obj

    @staticmethod
    def validate_dataframe(
        df: Any,
        field: str = "dataframe",
        required_columns: Optional[List[str]] = None,
        column_types: Optional[Dict[str, type]] = None,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        allow_na: bool = True,
    ) -> pd.DataFrame:
        """
        验证数据框。

        Args:
            df: 要验证的数据框
            field: 字段名称
            required_columns: 必需的列名列表
            column_types: 列名到类型的映射
            min_rows: 最小行数
            max_rows: 最大行数
            allow_na: 是否允许缺失值

        Returns:
            pd.DataFrame: 验证后的数据框

        Raises:
            ValidationError: 如果验证失败
        """
        # 检查是否为 DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(
                f"字段 '{field}' 不是 pandas DataFrame",
                field=field,
                value=type(df).__name__,
                expected="pandas.DataFrame",
            )

        # 检查必需列
        if required_columns is not None:
            missing_columns = [
                col for col in required_columns if col not in df.columns
            ]
            if missing_columns:
                raise ValidationError(
                    f"字段 '{field}' 缺少必需的列",
                    field=field,
                    value=df.columns.tolist(),
                    expected=f"包含列: {', '.join(required_columns)}",
                    details={"missing_columns": missing_columns},
                )

        # 检查列类型
        if column_types is not None:
            for col_name, expected_type in column_types.items():
                if col_name in df.columns:
                    col_dtype = df[col_name].dtype

                    # 简单的类型检查
                    if expected_type == float and not pd.api.types.is_float_dtype(
                        col_dtype
                    ):
                        raise ValidationError(
                            f"列 '{col_name}' 的类型不匹配",
                            field=f"{field}.{col_name}",
                            value=str(col_dtype),
                            expected="float",
                        )
                    elif expected_type == int and not pd.api.types.is_integer_dtype(
                        col_dtype
                    ):
                        raise ValidationError(
                            f"列 '{col_name}' 的类型不匹配",
                            field=f"{field}.{col_name}",
                            value=str(col_dtype),
                            expected="int",
                        )
                    elif expected_type == str and not pd.api.types.is_string_dtype(
                        col_dtype
                    ):
                        raise ValidationError(
                            f"列 '{col_name}' 的类型不匹配",
                            field=f"{field}.{col_name}",
                            value=str(col_dtype),
                            expected="str",
                        )
                    elif expected_type == bool and not pd.api.types.is_bool_dtype(
                        col_dtype
                    ):
                        raise ValidationError(
                            f"列 '{col_name}' 的类型不匹配",
                            field=f"{field}.{col_name}",
                            value=str(col_dtype),
                            expected="bool",
                        )

        # 检查行数
        num_rows = len(df)
        if min_rows is not None and num_rows < min_rows:
            raise ValidationError(
                f"字段 '{field}' 的行数太少",
                field=field,
                value=num_rows,
                expected=f">= {min_rows} 行",
            )

        if max_rows is not None and num_rows > max_rows:
            raise ValidationError(
                f"字段 '{field}' 的行数太多",
                field=field,
                value=num_rows,
                expected=f"<= {max_rows} 行",
            )

        # 检查缺失值
        if not allow_na:
            na_columns = df.columns[df.isna().any()].tolist()
            if na_columns:
                raise ValidationError(
                    f"字段 '{field}' 包含缺失值",
                    field=field,
                    value=df.shape,
                    expected="无缺失值的数据框",
                    details={"columns_with_na": na_columns},
                )

        return df

    @staticmethod
    def validate_stock_symbol(
        symbol: Any,
        field: str = "symbol",
        market: Optional[str] = None,
    ) -> str:
        """
        验证股票代码。

        Args:
            symbol: 股票代码
            field: 字段名称
            market: 市场类型 (US, HK, CN)

        Returns:
            str: 验证后的股票代码

        Raises:
            ValidationError: 如果验证失败
        """
        # 转换为字符串
        symbol_str = Validator.validate_string(symbol, field, min_length=1)

        # 根据市场进行验证
        if market == "US":
            # 美股代码: 1-5个大写字母
            if not re.match(r"^[A-Z]{1,5}$", symbol_str):
                raise ValidationError(
                    f"字段 '{field}' 不是有效的美股代码",
                    field=field,
                    value=symbol_str,
                    expected="1-5个大写字母",
                )
        elif market == "HK":
            # 港股代码: 4-5位数字
            if not re.match(r"^\d{4,5}$", symbol_str):
                raise ValidationError(
                    f"字段 '{field}' 不是有效的港股代码",
                    field=field,
                    value=symbol_str,
                    expected="4-5位数字",
                )
        elif market == "CN":
            # A股代码: 6位数字，上海以6开头，深圳以0或3开头
            if not re.match(r"^[0-9]{6}$", symbol_str):
                raise ValidationError(
                    f"字段 '{field}' 不是有效的A股代码",
                    field=field,
                    value=symbol_str,
                    expected="6位数字",
                )
            # 进一步验证交易所
            if symbol_str.startswith("6"):
                # 上海证券交易所
                pass
            elif symbol_str.startswith("0") or symbol_str.startswith("3"):
                # 深圳证券交易所
                pass
            else:
                raise ValidationError(
                    f"字段 '{field}' 不是有效的A股代码",
                    field=field,
                    value=symbol_str,
                    expected="以6（上海）、0或3（深圳）开头的6位数字",
                )
        else:
            # 通用验证: 只允许字母、数字和点号
            if not re.match(r"^[A-Za-z0-9.]{1,10}$", symbol_str):
                raise ValidationError(
                    f"字段 '{field}' 不是有效的股票代码",
                    field=field,
                    value=symbol_str,
                    expected="1-10个字母、数字或点号",
                )

        return symbol_str.upper() if market == "US" else symbol_str


class DataCleaner:
    """数据清洗器。"""

    @staticmethod
    def clean_dataframe(
        df: pd.DataFrame,
        fill_na: bool = True,
        remove_duplicates: bool = True,
        sort_index: bool = True,
        validate_ohlcv: bool = True,
    ) -> pd.DataFrame:
        """
        清洗数据框。

        Args:
            df: 要清洗的数据框
            fill_na: 是否填充缺失值
            remove_duplicates: 是否移除重复行
            sort_index: 是否按索引排序
            validate_ohlcv: 是否验证OHLCV数据

        Returns:
            pd.DataFrame: 清洗后的数据框
        """
        df_clean = df.copy()

        # 移除重复行
        if remove_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean[~df_clean.index.duplicated(keep="first")]
            removed = initial_rows - len(df_clean)
            if removed > 0:
                logger.debug(f"移除了 {removed} 个重复行")

        # 按索引排序
        if sort_index and df_clean.index.is_monotonic_increasing is False:
            df_clean = df_clean.sort_index()
            logger.debug("按索引排序数据框")

        # 填充缺失值
        if fill_na:
            # 对于OHLCV数据，使用前向填充
            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            existing_columns = [col for col in ohlcv_columns if col in df_clean.columns]

            if existing_columns:
                na_before = df_clean[existing_columns].isna().sum().sum()
                df_clean[existing_columns] = df_clean[existing_columns].ffill()
                na_after = df_clean[existing_columns].isna().sum().sum()
                filled = na_before - na_after
                if filled > 0:
                    logger.debug(f"前向填充了 {filled} 个缺失值")

        # 验证OHLCV数据
        if validate_ohlcv and all(
            col in df_clean.columns for col in ["open", "high", "low", "close"]
        ):
            # 检查价格数据的合理性
            mask_invalid = (
                (df_clean["high"] < df_clean["low"])
                | (df_clean["close"] > df_clean["high"])
                | (df_clean["close"] < df_clean["low"])
                | (df_clean["open"] > df_clean["high"])
                | (df_clean["open"] < df_clean["low"])
            )

            invalid_count = mask_invalid.sum()
            if invalid_count > 0:
                logger.warning(f"发现 {invalid_count} 行无效的OHLC数据")
                # 修正无效数据：使用前一行数据
                df_clean.loc[mask_invalid, ["open", "high", "low", "close"]] = np.nan
                df_clean[["open", "high", "low", "close"]] = df_clean[
                    ["open", "high", "low", "close"]
                ].ffill()

        # 检查并记录清洗结果
        na_count = df_clean.isna().sum().sum()
        if na_count > 0:
            logger.warning(f"清洗后仍有 {na_count} 个缺失值")

        logger.debug(
            f"数据清洗完成: 原始形状 {df.shape}, 清洗后形状 {df_clean.shape}"
        )

        return df_clean

    @staticmethod
    def normalize_prices(
        df: pd.DataFrame,
        adjust_column: str = "close",
        base_date: Optional[Union[date, datetime, str]] = None,
        base_value: float = 100.0,
    ) -> pd.DataFrame:
        """
        标准化价格数据。

        Args:
            df: 包含价格数据的数据框
            adjust_column: 要标准化的列名
            base_date: 基准日期，如果为 None 则使用第一行
            base_value: 基准值

        Returns:
            pd.DataFrame: 标准化后的数据框
        """
        if adjust_column not in df.columns:
            raise ValidationError(
                f"列 '{adjust_column}' 不存在",
                field="adjust_column",
                value=adjust_column,
                expected=f"数据框中的列名",
            )

        df_normalized = df.copy()

        # 确定基准价格
        if base_date is not None:
            # 解析基准日期
            if isinstance(base_date, (date, datetime)):
                base_date_obj = (
                    base_date if isinstance(base_date, date) else base_date.date()
                )
            else:
                base_date_obj = DateUtils.parse_date(base_date)

            # 找到最接近的日期
            if df_normalized.index.is_all_dates:
                date_index = pd.DatetimeIndex(df_normalized.index)
                base_idx = date_index.get_indexer([pd.Timestamp(base_date_obj)], method="nearest")[0]
                if base_idx == -1:
                    raise ValidationError(
                        f"找不到基准日期 {base_date_obj} 附近的数据",
                        field="base_date",
                        value=base_date_obj,
                        expected="数据范围内的日期",
                    )
                base_price = df_normalized.iloc[base_idx][adjust_column]
            else:
                raise ValidationError(
                    "数据框索引不是日期类型",
                    field="df.index",
                    value=type(df_normalized.index).__name__,
                    expected="日期类型索引",
                )
        else:
            # 使用第一行作为基准
            base_price = df_normalized.iloc[0][adjust_column]

        # 标准化价格
        if base_price != 0:
            df_normalized[adjust_column] = (
                df_normalized[adjust_column] / base_price * base_value
            )
        else:
            logger.warning(f"基准价格为0，无法标准化")

        logger.debug(
            f"价格标准化完成: 基准价格 {base_price:.4f}, 基准值 {base_value}"
        )

        return df_normalized

    @staticmethod
    def calculate_returns(
        df: pd.DataFrame,
        price_column: str = "close",
        return_type: str = "simple",
        log_returns: bool = False,
    ) -> pd.DataFrame:
        """
        计算收益率。

        Args:
            df: 包含价格数据的数据框
            price_column: 价格列名
            return_type: 收益率类型 ('simple' 或 'log')
            log_returns: 是否计算对数收益率（已弃用，使用 return_type）

        Returns:
            pd.DataFrame: 包含收益率的数据框
        """
        if price_column not in df.columns:
            raise ValidationError(
                f"列 '{price_column}' 不存在",
                field="price_column",
                value=price_column,
                expected=f"数据框中的列名",
            )

        df_returns = df.copy()

        # 向后兼容
        if log_returns:
            return_type = "log"

        # 计算收益率
        if return_type == "simple":
            df_returns["returns"] = df_returns[price_column].pct_change()
        elif return_type == "log":
            df_returns["returns"] = np.log(
                df_returns[price_column] / df_returns[price_column].shift(1)
            )
        else:
            raise ValidationError(
                f"不支持的收益率类型: {return_type}",
                field="return_type",
                value=return_type,
                expected="'simple' 或 'log'",
            )

        # 移除第一行的 NaN
        df_returns["returns"] = df_returns["returns"].fillna(0)

        logger.debug(f"计算收益率完成: 类型={return_type}")

        return df_returns


# 导出
__all__ = [
    "Validator",
    "DataCleaner",
]
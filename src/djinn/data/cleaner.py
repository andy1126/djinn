"""
Djinn 数据清洗器。

这个模块提供了数据清洗和预处理功能。
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, date
import warnings

from .market_data import MarketData, MarketDataType, OHLCV
from ..utils.logger import logger
from ..utils.exceptions import DataError
from ..utils.validation import Validator, DataCleaner as BaseDataCleaner
from ..utils.date_utils import DateUtils


class DataCleaner:
    """数据清洗器。"""

    @staticmethod
    def clean_market_data(
        market_data: MarketData,
        fill_na: bool = True,
        remove_duplicates: bool = True,
        sort_index: bool = True,
        validate_ohlcv: bool = True,
        handle_outliers: bool = True,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
    ) -> MarketData:
        """
        清洗市场数据。

        Args:
            market_data: 市场数据对象
            fill_na: 是否填充缺失值
            remove_duplicates: 是否移除重复行
            sort_index: 是否按索引排序
            validate_ohlcv: 是否验证OHLCV数据
            handle_outliers: 是否处理异常值
            outlier_method: 异常值检测方法 ('iqr', 'zscore', 'percentile')
            outlier_threshold: 异常值阈值

        Returns:
            MarketData: 清洗后的市场数据对象
        """
        if market_data.data_type != MarketDataType.OHLCV:
            logger.warning(f"数据清洗仅支持 OHLCV 数据类型，当前类型: {market_data.data_type}")
            return market_data

        # 转换为 DataFrame
        df = market_data.to_dataframe()

        # 记录原始信息
        original_shape = df.shape
        original_na_count = df.isna().sum().sum()

        # 基本清洗
        df_clean = BaseDataCleaner.clean_dataframe(
            df,
            fill_na=fill_na,
            remove_duplicates=remove_duplicates,
            sort_index=sort_index,
            validate_ohlcv=validate_ohlcv,
        )

        # 处理异常值
        if handle_outliers:
            df_clean = DataCleaner._handle_outliers(
                df_clean,
                method=outlier_method,
                threshold=outlier_threshold,
            )

        # 记录清洗结果
        cleaned_shape = df_clean.shape
        cleaned_na_count = df_clean.isna().sum().sum()

        removed_rows = original_shape[0] - cleaned_shape[0]
        filled_na = original_na_count - cleaned_na_count

        if removed_rows > 0:
            logger.info(f"移除了 {removed_rows} 行数据")
        if filled_na > 0:
            logger.info(f"填充了 {filled_na} 个缺失值")

        # 创建清洗后的市场数据对象
        cleaned_data = MarketData(
            symbol=market_data.symbol,
            data_type=market_data.data_type,
            data=df_clean,
            interval=market_data.interval,
            adjustment=market_data.adjustment,
            metadata={
                **market_data.metadata,
                "cleaned": True,
                "original_rows": original_shape[0],
                "cleaned_rows": cleaned_shape[0],
                "removed_rows": removed_rows,
                "filled_na": filled_na,
                "outlier_method": outlier_method if handle_outliers else None,
                "cleaned_at": datetime.now().isoformat(),
            },
        )

        logger.debug(
            f"数据清洗完成: {market_data.symbol}, "
            f"原始: {original_shape}, 清洗后: {cleaned_shape}"
        )

        return cleaned_data

    @staticmethod
    def resample_data(
        market_data: MarketData,
        new_interval: str,
        method: str = "ohlc",
        volume_agg: str = "sum",
    ) -> MarketData:
        """
        重采样数据到新的时间间隔。

        Args:
            market_data: 市场数据对象
            new_interval: 新的时间间隔
            method: 重采样方法 ('ohlc', 'close', 'mean')
            volume_agg: 成交量聚合方法 ('sum', 'mean', 'last')

        Returns:
            MarketData: 重采样后的市场数据对象
        """
        if market_data.data_type != MarketDataType.OHLCV:
            raise DataError(
                f"重采样仅支持 OHLCV 数据类型",
                details={"data_type": market_data.data_type.value},
            )

        df = market_data.to_dataframe()

        if df.empty:
            return market_data

        # 确保索引是 DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise DataError(
                    f"无法将索引转换为时间戳",
                    details={"error": str(e)},
                )

        # 定义重采样规则
        try:
            if method == "ohlc":
                # OHLC 重采样
                ohlc_dict = {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                }

                if "adj_close" in df.columns:
                    ohlc_dict["adj_close"] = "last"

                resampled = df.resample(new_interval).agg(ohlc_dict)

                # 成交量聚合
                if "volume" in df.columns:
                    if volume_agg == "sum":
                        resampled["volume"] = df["volume"].resample(new_interval).sum()
                    elif volume_agg == "mean":
                        resampled["volume"] = df["volume"].resample(new_interval).mean()
                    elif volume_agg == "last":
                        resampled["volume"] = df["volume"].resample(new_interval).last()
                    else:
                        resampled["volume"] = df["volume"].resample(new_interval).sum()

            elif method == "close":
                # 仅使用收盘价
                resampled = pd.DataFrame()
                resampled["close"] = df["close"].resample(new_interval).last()

                if "adj_close" in df.columns:
                    resampled["adj_close"] = df["adj_close"].resample(new_interval).last()

                if "volume" in df.columns:
                    if volume_agg == "sum":
                        resampled["volume"] = df["volume"].resample(new_interval).sum()
                    elif volume_agg == "mean":
                        resampled["volume"] = df["volume"].resample(new_interval).mean()
                    else:
                        resampled["volume"] = df["volume"].resample(new_interval).sum()

            elif method == "mean":
                # 使用平均值
                resampled = df.resample(new_interval).mean()
            else:
                raise DataError(
                    f"不支持的重采样方法: {method}",
                    details={"method": method, "expected": "ohlc, close, mean"},
                )

        except Exception as e:
            raise DataError(
                f"重采样失败",
                details={
                    "interval": new_interval,
                    "method": method,
                    "error": str(e),
                },
            )

        # 移除空行
        resampled = resampled.dropna()

        # 创建重采样后的市场数据对象
        resampled_data = MarketData(
            symbol=market_data.symbol,
            data_type=market_data.data_type,
            data=resampled,
            interval=new_interval,
            adjustment=market_data.adjustment,
            metadata={
                **market_data.metadata,
                "resampled": True,
                "original_interval": market_data.interval,
                "resampled_interval": new_interval,
                "resample_method": method,
                "volume_agg": volume_agg,
                "original_rows": len(df),
                "resampled_rows": len(resampled),
                "resampled_at": datetime.now().isoformat(),
            },
        )

        logger.info(
            f"数据重采样完成: {market_data.symbol}, "
            f"{market_data.interval} -> {new_interval}, "
            f"行数: {len(df)} -> {len(resampled)}"
        )

        return resampled_data

    @staticmethod
    def align_data(
        data_dict: Dict[str, MarketData],
        method: str = "inner",
        fill_method: str = "ffill",
    ) -> Dict[str, MarketData]:
        """
        对齐多个市场数据的时间索引。

        Args:
            data_dict: 股票代码到市场数据的映射
            method: 对齐方法 ('inner', 'outer', 'left', 'right')
            fill_method: 填充方法 ('ffill', 'bfill', 'zero', 'none')

        Returns:
            Dict[str, MarketData]: 对齐后的市场数据映射
        """
        # 过滤掉 None 值
        valid_data = {k: v for k, v in data_dict.items() if v is not None}

        if len(valid_data) < 2:
            logger.warning("需要至少两个有效的数据集进行对齐")
            return data_dict

        # 转换为 DataFrame
        dfs = {}
        for symbol, market_data in valid_data.items():
            if market_data.data_type == MarketDataType.OHLCV:
                df = market_data.to_dataframe()
                dfs[symbol] = df
            else:
                logger.warning(f"跳过非 OHLCV 数据: {symbol} ({market_data.data_type})")

        if len(dfs) < 2:
            return data_dict

        # 对齐索引
        try:
            # 获取所有索引的并集或交集
            all_indices = [df.index for df in dfs.values()]

            if method == "inner":
                common_index = all_indices[0]
                for idx in all_indices[1:]:
                    common_index = common_index.intersection(idx)
            elif method == "outer":
                common_index = all_indices[0]
                for idx in all_indices[1:]:
                    common_index = common_index.union(idx)
            elif method == "left":
                common_index = all_indices[0]
            elif method == "right":
                common_index = all_indices[-1]
            else:
                raise DataError(
                    f"不支持的对齐方法: {method}",
                    details={"method": method, "expected": "inner, outer, left, right"},
                )

            # 重新索引和填充
            aligned_dfs = {}
            for symbol, df in dfs.items():
                aligned_df = df.reindex(common_index)

                # 填充缺失值
                if fill_method == "ffill":
                    aligned_df = aligned_df.ffill().bfill()  # 前向填充，然后后向填充
                elif fill_method == "bfill":
                    aligned_df = aligned_df.bfill().ffill()  # 后向填充，然后前向填充
                elif fill_method == "zero":
                    aligned_df = aligned_df.fillna(0)
                elif fill_method == "none":
                    pass  # 不填充
                else:
                    aligned_df = aligned_df.ffill().bfill()  # 默认使用前向填充

                aligned_dfs[symbol] = aligned_df

            # 转换回 MarketData
            aligned_data = {}
            for symbol, market_data in valid_data.items():
                if symbol in aligned_dfs:
                    aligned_df = aligned_dfs[symbol]

                    aligned_market_data = MarketData(
                        symbol=market_data.symbol,
                        data_type=market_data.data_type,
                        data=aligned_df,
                        interval=market_data.interval,
                        adjustment=market_data.adjustment,
                        metadata={
                            **market_data.metadata,
                            "aligned": True,
                            "align_method": method,
                            "fill_method": fill_method,
                            "original_rows": len(dfs[symbol]),
                            "aligned_rows": len(aligned_df),
                            "aligned_at": datetime.now().isoformat(),
                        },
                    )

                    aligned_data[symbol] = aligned_market_data
                else:
                    aligned_data[symbol] = market_data

            logger.info(
                f"数据对齐完成: {len(aligned_dfs)} 个数据集, "
                f"方法: {method}, 填充: {fill_method}, "
                f"共同行数: {len(common_index)}"
            )

            return aligned_data

        except Exception as e:
            raise DataError(
                f"数据对齐失败",
                details={
                    "method": method,
                    "fill_method": fill_method,
                    "symbols": list(valid_data.keys()),
                    "error": str(e),
                },
            )

    @staticmethod
    def calculate_returns(
        market_data: MarketData,
        return_type: str = "simple",
        include_original: bool = True,
    ) -> MarketData:
        """
        计算收益率。

        Args:
            market_data: 市场数据对象
            return_type: 收益率类型 ('simple', 'log')
            include_original: 是否包含原始数据

        Returns:
            MarketData: 包含收益率的数据对象
        """
        if market_data.data_type != MarketDataType.OHLCV:
            raise DataError(
                f"收益率计算仅支持 OHLCV 数据类型",
                details={"data_type": market_data.data_type.value},
            )

        df = market_data.to_dataframe()

        if df.empty:
            return market_data

        # 计算收益率
        df_returns = BaseDataCleaner.calculate_returns(
            df,
            price_column="close",
            return_type=return_type,
        )

        # 决定返回的数据
        if include_original:
            return_data = df_returns
        else:
            return_data = df_returns[["returns"]]

        # 创建包含收益率的数据对象
        returns_data = MarketData(
            symbol=market_data.symbol,
            data_type=market_data.data_type,
            data=return_data,
            interval=market_data.interval,
            adjustment=market_data.adjustment,
            metadata={
                **market_data.metadata,
                "returns_calculated": True,
                "return_type": return_type,
                "include_original": include_original,
                "calculated_at": datetime.now().isoformat(),
            },
        )

        logger.debug(
            f"收益率计算完成: {market_data.symbol}, 类型: {return_type}"
        )

        return returns_data

    @staticmethod
    def normalize_prices(
        market_data: MarketData,
        base_date: Optional[Union[date, datetime, str]] = None,
        base_value: float = 100.0,
    ) -> MarketData:
        """
        标准化价格数据。

        Args:
            market_data: 市场数据对象
            base_date: 基准日期，如果为 None 则使用第一行
            base_value: 基准值

        Returns:
            MarketData: 标准化后的市场数据对象
        """
        if market_data.data_type != MarketDataType.OHLCV:
            raise DataError(
                f"价格标准化仅支持 OHLCV 数据类型",
                details={"data_type": market_data.data_type.value},
            )

        df = market_data.to_dataframe()

        if df.empty:
            return market_data

        # 标准化价格
        df_normalized = BaseDataCleaner.normalize_prices(
            df,
            adjust_column="close",
            base_date=base_date,
            base_value=base_value,
        )

        # 创建标准化后的数据对象
        normalized_data = MarketData(
            symbol=market_data.symbol,
            data_type=market_data.data_type,
            data=df_normalized,
            interval=market_data.interval,
            adjustment=market_data.adjustment,
            metadata={
                **market_data.metadata,
                "normalized": True,
                "base_date": base_date.strftime("%Y-%m-%d") if base_date else "first_row",
                "base_value": base_value,
                "normalized_at": datetime.now().isoformat(),
            },
        )

        logger.debug(
            f"价格标准化完成: {market_data.symbol}, 基准值: {base_value}"
        )

        return normalized_data

    @staticmethod
    def _handle_outliers(
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        处理异常值。

        Args:
            df: 数据框
            method: 异常值检测方法
            threshold: 异常值阈值

        Returns:
            pd.DataFrame: 处理异常值后的数据框
        """
        df_clean = df.copy()

        # 只处理数值列
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        # 移除价格和成交量相关的列
        price_columns = ["open", "high", "low", "close", "adj_close"]
        volume_column = "volume"

        # 分别处理价格和成交量
        for col in numeric_columns:
            if col in price_columns:
                # 价格数据：使用温和的方法处理异常值
                df_clean = DataCleaner._handle_price_outliers(df_clean, col, method, threshold)
            elif col == volume_column:
                # 成交量数据：使用不同的方法
                df_clean = DataCleaner._handle_volume_outliers(df_clean, col, method, threshold)

        return df_clean

    @staticmethod
    def _handle_price_outliers(
        df: pd.DataFrame,
        column: str,
        method: str,
        threshold: float,
    ) -> pd.DataFrame:
        """处理价格异常值。"""
        if column not in df.columns:
            return df

        series = df[column].copy()
        original_na = series.isna().sum()

        # 计算异常值边界
        if method == "iqr":
            # IQR 方法
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
        elif method == "zscore":
            # Z-score 方法
            mean = series.mean()
            std = series.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        elif method == "percentile":
            # 百分位方法
            lower_bound = series.quantile(threshold / 100)
            upper_bound = series.quantile(1 - threshold / 100)
        else:
            return df

        # 识别异常值
        outliers = (series < lower_bound) | (series > upper_bound)
        outlier_count = outliers.sum()

        if outlier_count > 0:
            logger.debug(f"在 {column} 列中发现 {outlier_count} 个异常值")

            # 对于价格数据，使用前向填充处理异常值
            series[outliers] = np.nan
            series = series.ffill().bfill()

            df[column] = series

            filled_count = outlier_count - series.isna().sum() + original_na
            if filled_count > 0:
                logger.debug(f"填充了 {filled_count} 个价格异常值")

        return df

    @staticmethod
    def _handle_volume_outliers(
        df: pd.DataFrame,
        column: str,
        method: str,
        threshold: float,
    ) -> pd.DataFrame:
        """处理成交量异常值。"""
        if column not in df.columns:
            return df

        series = df[column].copy()

        # 对于成交量，我们更关心异常大的值
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + threshold * iqr
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            upper_bound = mean + threshold * std
        elif method == "percentile":
            upper_bound = series.quantile(1 - threshold / 100)
        else:
            return df

        # 只处理异常大的成交量
        outliers = series > upper_bound
        outlier_count = outliers.sum()

        if outlier_count > 0:
            logger.debug(f"在 {column} 列中发现 {outlier_count} 个异常大的成交量")

            # 对于异常大的成交量，使用上限值
            series[outliers] = upper_bound
            df[column] = series

        return df


# 导出
__all__ = [
    "DataCleaner",
]
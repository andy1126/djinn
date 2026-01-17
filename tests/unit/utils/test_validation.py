"""
Unit tests for the validation module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from decimal import Decimal

from src.djinn.utils.validation import Validator, DataCleaner
from src.djinn.utils.exceptions import ValidationError


class TestValidator:
    """Test suite for Validator class."""

    def test_validate_not_none_success(self):
        """Test validate_not_none with valid values."""
        # Should not raise exception
        Validator.validate_not_none("test", "field1")
        Validator.validate_not_none(123, "field2")
        Validator.validate_not_none([], "field3")
        Validator.validate_not_none({}, "field4")

    def test_validate_not_none_failure(self):
        """Test validate_not_none with None value."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_not_none(None, "test_field")

        assert exc_info.value.field == "test_field"
        assert "不能为 None" in str(exc_info.value)

    def test_validate_not_empty_string_success(self):
        """Test validate_not_empty with non-empty string."""
        Validator.validate_not_empty("test", "field1")
        Validator.validate_not_empty("  test  ", "field2")

    def test_validate_not_empty_string_failure(self):
        """Test validate_not_empty with empty string."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_not_empty("", "test_field")

        assert "不能为空字符串" in str(exc_info.value)

    def test_validate_not_empty_list_success(self):
        """Test validate_not_empty with non-empty list."""
        Validator.validate_not_empty([1, 2, 3], "field1")
        Validator.validate_not_empty(["test"], "field2")

    def test_validate_not_empty_list_failure(self):
        """Test validate_not_empty with empty list."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_not_empty([], "test_field")

        assert "不能为空集合" in str(exc_info.value)

    def test_validate_type_success(self):
        """Test validate_type with matching types."""
        Validator.validate_type("test", str, "field1")
        Validator.validate_type(123, int, "field2")
        Validator.validate_type(123.45, float, "field3")
        Validator.validate_type([1, 2], list, "field4")
        Validator.validate_type({"a": 1}, dict, "field5")

    def test_validate_type_failure(self):
        """Test validate_type with mismatched type."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_type("test", int, "test_field")

        assert "类型不匹配" in str(exc_info.value)
        assert exc_info.value.field == "test_field"

    def test_validate_type_multiple_success(self):
        """Test validate_type with multiple allowed types."""
        Validator.validate_type("test", (str, int), "field1")
        Validator.validate_type(123, (str, int), "field2")

    def test_validate_numeric_success(self):
        """Test validate_numeric with valid numeric values."""
        # Test with various numeric types
        assert Validator.validate_numeric(123, "field1") == 123.0
        assert Validator.validate_numeric(123.45, "field2") == 123.45
        assert Validator.validate_numeric("123", "field3") == 123.0
        assert Validator.validate_numeric("123.45", "field4") == 123.45
        assert Validator.validate_numeric(Decimal("123.45"), "field5") == 123.45

    def test_validate_numeric_with_range(self):
        """Test validate_numeric with min/max constraints."""
        # Test within range
        result = Validator.validate_numeric(50, "field1", min_value=0, max_value=100)
        assert result == 50.0

        # Test at boundaries
        Validator.validate_numeric(0, "field2", min_value=0, max_value=100)
        Validator.validate_numeric(100, "field3", min_value=0, max_value=100)

    def test_validate_numeric_out_of_range(self):
        """Test validate_numeric with out-of-range values."""
        # Below min
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_numeric(-10, "test_field", min_value=0)
        assert "必须大于等于" in str(exc_info.value)

        # Above max
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_numeric(150, "test_field", max_value=100)
        assert "必须小于等于" in str(exc_info.value)

    def test_validate_numeric_nan_inf(self):
        """Test validate_numeric with NaN and Inf."""
        # Test NaN not allowed by default
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_numeric(np.nan, "test_field")
        assert "不能为 NaN" in str(exc_info.value)

        # Test Inf not allowed by default
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_numeric(np.inf, "test_field")
        assert "不能为无穷大" in str(exc_info.value)

        # Test NaN allowed when specified
        result = Validator.validate_numeric(np.nan, "field1", allow_nan=True)
        assert np.isnan(result)

        # Test Inf allowed when specified
        result = Validator.validate_numeric(np.inf, "field2", allow_inf=True)
        assert np.isinf(result)

    def test_validate_string_success(self):
        """Test validate_string with valid strings."""
        result = Validator.validate_string("test", "field1")
        assert result == "test"

        result = Validator.validate_string(123, "field2")  # Convertible to string
        assert result == "123"

    def test_validate_string_with_constraints(self):
        """Test validate_string with length constraints."""
        # Test length constraints
        result = Validator.validate_string("test", "field1", min_length=3, max_length=10)
        assert result == "test"

        # Test exact length
        result = Validator.validate_string("test", "field2", min_length=4, max_length=4)
        assert result == "test"

    def test_validate_string_length_violations(self):
        """Test validate_string with length violations."""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_string("ab", "test_field", min_length=3)
        assert "长度必须至少为" in str(exc_info.value)

        # Too long
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_string("abcdefghij", "test_field", max_length=5)
        assert "长度不能超过" in str(exc_info.value)

    def test_validate_string_pattern(self):
        """Test validate_string with regex pattern."""
        # Valid pattern
        result = Validator.validate_string("ABC123", "field1", pattern=r"^[A-Z]{3}\d{3}$")
        assert result == "ABC123"

        # Invalid pattern
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_string("abc123", "test_field", pattern=r"^[A-Z]{3}\d{3}$")
        assert "不匹配模式" in str(exc_info.value)

    def test_validate_string_allowed_values(self):
        """Test validate_string with allowed values list."""
        # Valid value
        result = Validator.validate_string("apple", "field1", allowed_values=["apple", "banana", "cherry"])
        assert result == "apple"

        # Invalid value
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_string("orange", "test_field", allowed_values=["apple", "banana", "cherry"])
        assert "不在允许的列表中" in str(exc_info.value)

    def test_validate_date_success(self):
        """Test validate_date with valid date inputs."""
        # Test with date object
        test_date = date(2023, 1, 1)
        result = Validator.validate_date(test_date, "field1")
        assert result == test_date

        # Test with datetime object
        test_datetime = datetime(2023, 1, 1, 12, 30, 45)
        result = Validator.validate_date(test_datetime, "field2")
        assert result == test_date  # Should return date part only

        # Test with string
        result = Validator.validate_date("2023-01-01", "field3")
        assert result == test_date

    def test_validate_date_with_range(self):
        """Test validate_date with min/max date constraints."""
        test_date = date(2023, 6, 1)

        # Within range
        result = Validator.validate_date(
            test_date,
            "field1",
            min_date=date(2023, 1, 1),
            max_date=date(2023, 12, 31)
        )
        assert result == test_date

        # At boundaries
        Validator.validate_date(date(2023, 1, 1), "field2", min_date=date(2023, 1, 1))
        Validator.validate_date(date(2023, 12, 31), "field3", max_date=date(2023, 12, 31))

    def test_validate_date_range_violations(self):
        """Test validate_date with range violations."""
        # Before min date
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_date(
                date(2022, 12, 31),
                "test_field",
                min_date=date(2023, 1, 1)
            )
        assert "必须晚于或等于" in str(exc_info.value)

        # After max date
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_date(
                date(2024, 1, 1),
                "test_field",
                max_date=date(2023, 12, 31)
            )
        assert "必须早于或等于" in str(excisc_info.value)

    def test_validate_dataframe_success(self):
        """Test validate_dataframe with valid DataFrame."""
        # Create test DataFrame
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'volume': [1000, 2000, 3000]
        })

        result = Validator.validate_dataframe(df, "test_df")
        pd.testing.assert_frame_equal(result, df)

    def test_validate_dataframe_not_dataframe(self):
        """Test validate_dataframe with non-DataFrame input."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_dataframe("not a dataframe", "test_field")

        assert "不是 pandas DataFrame" in str(exc_info.value)

    def test_validate_dataframe_required_columns(self):
        """Test validate_dataframe with required columns."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

        # All required columns present
        result = Validator.validate_dataframe(df, "test_df", required_columns=['A', 'B'])
        pd.testing.assert_frame_equal(result, df)

        # Missing required column
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_dataframe(df, "test_field", required_columns=['A', 'B', 'C'])

        assert "缺少必需的列" in str(exc_info.value)
        assert 'C' in str(exc_info.value.details['missing_columns'])

    def test_validate_dataframe_row_count(self):
        """Test validate_dataframe with row count constraints."""
        df = pd.DataFrame({'A': range(10)})

        # Within limits
        result = Validator.validate_dataframe(df, "test_df", min_rows=5, max_rows=15)
        pd.testing.assert_frame_equal(result, df)

        # Too few rows
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_dataframe(df, "test_field", min_rows=15)
        assert "行数太少" in str(exc_info.value)

        # Too many rows
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_dataframe(df, "test_field", max_rows=5)
        assert "行数太多" in str(exc_info.value)

    def test_validate_stock_symbol_us(self):
        """Test validate_stock_symbol for US market."""
        # Valid US symbols
        assert Validator.validate_stock_symbol("AAPL", market="US") == "AAPL"
        assert Validator.validate_stock_symbol("GOOGL", market="US") == "GOOGL"
        assert Validator.validate_stock_symbol("BRK.B", market="US") == "BRK.B"

        # Invalid US symbols
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_stock_symbol("123", market="US")
        assert "不是有效的美股代码" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_stock_symbol("ABCDEF", market="US")  # Too long
        assert "不是有效的美股代码" in str(exc_info.value)

    def test_validate_stock_symbol_hk(self):
        """Test validate_stock_symbol for HK market."""
        # Valid HK symbols
        assert Validator.validate_stock_symbol("0700", market="HK") == "0700"
        assert Validator.validate_stock_symbol("00001", market="HK") == "00001"

        # Invalid HK symbols
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_stock_symbol("ABC", market="HK")
        assert "不是有效的港股代码" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_stock_symbol("123456", market="HK")  # Too long
        assert "不是有效的港股代码" in str(exc_info.value)

    def test_validate_stock_symbol_cn(self):
        """Test validate_stock_symbol for CN market."""
        # Valid CN symbols
        assert Validator.validate_stock_symbol("600000", market="CN") == "600000"  # Shanghai
        assert Validator.validate_stock_symbol("000001", market="CN") == "000001"  # Shenzhen
        assert Validator.validate_stock_symbol("300001", market="CN") == "300001"  # Shenzhen ChiNext

        # Invalid CN symbols
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_stock_symbol("12345", market="CN")  # Too short
        assert "不是有效的A股代码" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_stock_symbol("1234567", market="CN")  # Too long
        assert "不是有效的A股代码" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_stock_symbol("200000", market="CN")  # Invalid exchange
        assert "不是有效的A股代码" in str(exc_info.value)

    def test_validate_stock_symbol_generic(self):
        """Test validate_stock_symbol without market specification."""
        # Valid generic symbols
        assert Validator.validate_stock_symbol("BTC-USD") == "BTC-USD"
        assert Validator.validate_stock_symbol("ETH.USD") == "ETH.USD"
        assert Validator.validate_stock_symbol("123") == "123"

        # Invalid generic symbols
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_stock_symbol("", market=None)  # Too short
        assert "不是有效的股票代码" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_stock_symbol("ABCDEFGHIJK", market=None)  # Too long
        assert "不是有效的股票代码" in str(exc_info.value)


class TestDataCleaner:
    """Test suite for DataCleaner class."""

    def test_clean_dataframe_basic(self):
        """Test basic dataframe cleaning."""
        # Create test DataFrame with some issues
        df = pd.DataFrame({
            'open': [100.0, np.nan, 102.0, 102.0],
            'high': [105.0, 106.0, 107.0, 107.0],
            'low': [95.0, 96.0, np.nan, 96.0],
            'close': [102.0, 103.0, 104.0, 104.0],
        }, index=[0, 1, 2, 2])  # Duplicate index at position 2

        cleaned_df = DataCleaner.clean_dataframe(df)

        # Check duplicate removed (should have 3 rows instead of 4)
        assert len(cleaned_df) == 3

        # Check NaN filled (forward fill)
        assert not cleaned_df['open'].isna().any()
        assert not cleaned_df['low'].isna().any()

        # Check index is sorted
        assert cleaned_df.index.is_monotonic_increasing

    def test_clean_dataframe_without_fill_na(self):
        """Test dataframe cleaning without filling NaN."""
        df = pd.DataFrame({
            'open': [100.0, np.nan, 102.0],
            'high': [105.0, 106.0, 107.0],
        })

        cleaned_df = DataCleaner.clean_dataframe(df, fill_na=False)

        # NaN should remain
        assert cleaned_df['open'].isna().sum() == 1

    def test_clean_dataframe_without_remove_duplicates(self):
        """Test dataframe cleaning without removing duplicates."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 100.0],
        }, index=[0, 1, 0])  # Duplicate index

        cleaned_df = DataCleaner.clean_dataframe(df, remove_duplicates=False)

        # Duplicates should remain
        assert len(cleaned_df) == 3

    def test_normalize_prices_basic(self):
        """Test basic price normalization."""
        df = pd.DataFrame({
            'close': [100.0, 110.0, 121.0, 133.1],
        })

        normalized_df = DataCleaner.normalize_prices(df, adjust_column='close', base_value=100.0)

        # First value should be 100.0
        assert normalized_df['close'].iloc[0] == 100.0

        # Check normalization formula
        expected_values = [100.0, 110.0, 121.0, 133.1]  # Already relative to first value
        pd.testing.assert_series_equal(
            normalized_df['close'],
            pd.Series(expected_values),
            check_exact=False,
            rtol=1e-10
        )

    def test_normalize_prices_with_base_date(self):
        """Test price normalization with specific base date."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'close': [100.0, 110.0, 121.0, 133.1, 146.41],
        }, index=dates)

        # Normalize to third date (2023-01-03)
        normalized_df = DataCleaner.normalize_prices(
            df,
            adjust_column='close',
            base_date='2023-01-03',
            base_value=100.0
        )

        # Value at base date should be 100.0
        assert normalized_df.loc['2023-01-03', 'close'] == 100.0

        # Check first value is normalized correctly
        expected_first = 100.0 / 121.0 * 100.0  # (100/121)*100
        assert abs(normalized_df['close'].iloc[0] - expected_first) < 1e-10

    def test_normalize_prices_missing_column(self):
        """Test price normalization with missing column."""
        df = pd.DataFrame({'open': [100.0, 110.0]})

        with pytest.raises(ValidationError) as exc_info:
            DataCleaner.normalize_prices(df, adjust_column='close')

        assert "不存在" in str(exc_info.value)

    def test_calculate_returns_simple(self):
        """Test simple returns calculation."""
        df = pd.DataFrame({
            'close': [100.0, 110.0, 121.0, 133.1],
        })

        returns_df = DataCleaner.calculate_returns(df, price_column='close', return_type='simple')

        # Check returns column exists
        assert 'returns' in returns_df.columns

        # Check first return is 0 (filled NaN)
        assert returns_df['returns'].iloc[0] == 0.0

        # Check second return: (110-100)/100 = 0.1
        assert abs(returns_df['returns'].iloc[1] - 0.1) < 1e-10

        # Check third return: (121-110)/110 ≈ 0.1
        assert abs(returns_df['returns'].iloc[2] - 0.1) < 1e-10

    def test_calculate_returns_log(self):
        """Test log returns calculation."""
        df = pd.DataFrame({
            'close': [100.0, 110.0, 121.0],
        })

        returns_df = DataCleaner.calculate_returns(df, price_column='close', return_type='log')

        # Check log returns
        expected_log_return = np.log(110.0 / 100.0)
        assert abs(returns_df['returns'].iloc[1] - expected_log_return) < 1e-10

    def test_calculate_returns_invalid_type(self):
        """Test returns calculation with invalid return type."""
        df = pd.DataFrame({'close': [100.0, 110.0]})

        with pytest.raises(ValidationError) as exc_info:
            DataCleaner.calculate_returns(df, price_column='close', return_type='invalid')

        assert "不支持的收益率类型" in str(exc_info.value)

    def test_calculate_returns_missing_column(self):
        """Test returns calculation with missing column."""
        df = pd.DataFrame({'open': [100.0, 110.0]})

        with pytest.raises(ValidationError) as exc_info:
            DataCleaner.calculate_returns(df, price_column='close')

        assert "不存在" in str(exc_info.value)
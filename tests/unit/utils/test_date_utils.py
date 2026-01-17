"""
Unit tests for the date_utils module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

from src.djinn.utils.date_utils import (
    DateUtils,
    MarketCalendarType,
    DateFormat,
)
from src.djinn.utils.exceptions import ValidationError


class TestDateFormat:
    """Test DateFormat enumeration."""

    def test_date_format_values(self):
        """Test DateFormat enum values."""
        assert DateFormat.ISO == "%Y-%m-%d"
        assert DateFormat.US == "%m/%d/%Y"
        assert DateFormat.CN == "%Y年%m月%d日"
        assert DateFormat.COMPACT == "%Y%m%d"


class TestMarketCalendarType:
    """Test MarketCalendarType enumeration."""

    def test_market_calendar_type_values(self):
        """Test MarketCalendarType enum values."""
        assert MarketCalendarType.US == "XNYS"
        assert MarketCalendarType.HK == "XHKG"
        assert MarketCalendarType.CN == "XSHG"


class TestDateUtils:
    """Test DateUtils class."""

    def test_parse_date_with_format(self):
        """Test parse_date with specified format."""
        # Test ISO format
        result = DateUtils.parse_date("2023-12-31", "%Y-%m-%d")
        assert result == date(2023, 12, 31)

        # Test US format
        result = DateUtils.parse_date("12/31/2023", "%m/%d/%Y")
        assert result == date(2023, 12, 31)

        # Test compact format
        result = DateUtils.parse_date("20231231", "%Y%m%d")
        assert result == date(2023, 12, 31)

    def test_parse_date_auto_detect(self):
        """Test parse_date with automatic format detection."""
        # Should detect ISO format
        result = DateUtils.parse_date("2023-12-31")
        assert result == date(2023, 12, 31)

        # Should detect slash format
        result = DateUtils.parse_date("2023/12/31")
        assert result == date(2023, 12, 31)

        # Should detect compact format
        result = DateUtils.parse_date("20231231")
        assert result == date(2023, 12, 31)

        # Should detect US format
        result = DateUtils.parse_date("12/31/2023")
        assert result == date(2023, 12, 31)

        # Should detect datetime string
        result = DateUtils.parse_date("2023-12-31 14:30:45")
        assert result == date(2023, 12, 31)

    def test_parse_date_invalid_string(self):
        """Test parse_date with invalid date string."""
        with pytest.raises(ValidationError) as exc_info:
            DateUtils.parse_date("invalid-date")

        assert "无法解析日期字符串" in str(exc_info.value)

    def test_parse_date_no_error(self):
        """Test parse_date with raise_error=False."""
        # Valid date should return date object
        result = DateUtils.parse_date("2023-12-31", raise_error=False)
        assert result == date(2023, 12, 31)

        # Invalid date should return None
        result = DateUtils.parse_date("invalid", raise_error=False)
        assert result is None

    def test_format_date_from_date_object(self):
        """Test format_date with date object."""
        test_date = date(2023, 12, 31)

        # Test ISO format
        result = DateUtils.format_date(test_date, DateFormat.ISO)
        assert result == "2023-12-31"

        # Test US format
        result = DateUtils.format_date(test_date, DateFormat.US)
        assert result == "12/31/2023"

        # Test CN format
        result = DateUtils.format_date(test_date, DateFormat.CN)
        assert result == "2023年12月31日"

        # Test compact format
        result = DateUtils.format_date(test_date, DateFormat.COMPACT)
        assert result == "20231231"

    def test_format_date_from_datetime(self):
        """Test format_date with datetime object."""
        test_datetime = datetime(2023, 12, 31, 14, 30, 45)

        result = DateUtils.format_date(test_datetime, DateFormat.ISO)
        assert result == "2023-12-31"  # Should only include date part

    def test_format_date_from_string(self):
        """Test format_date with string input."""
        # Should parse and format
        result = DateUtils.format_date("2023-12-31", DateFormat.ISO)
        assert result == "2023-12-31"

    def test_format_date_from_pandas_timestamp(self):
        """Test format_date with pandas Timestamp."""
        test_timestamp = pd.Timestamp("2023-12-31")

        result = DateUtils.format_date(test_timestamp, DateFormat.ISO)
        assert result == "2023-12-31"

    def test_get_trading_days_basic(self):
        """Test basic get_trading_days functionality."""
        with patch('pandas_market_calendars.get_calendar') as mock_get_calendar:
            # Mock calendar
            mock_calendar = MagicMock()
            mock_calendar.schedule.return_value = pd.DataFrame(
                index=pd.date_range("2023-01-01", "2023-01-10", freq='D')
            )
            mock_get_calendar.return_value = mock_calendar

            # Call method
            result = DateUtils.get_trading_days(
                start_date="2023-01-01",
                end_date="2023-01-10",
                market=MarketCalendarType.US
            )

            # Verify result
            assert isinstance(result, pd.DatetimeIndex)
            assert len(result) > 0

            # Verify calendar was retrieved from cache or created
            mock_get_calendar.assert_called_once_with(MarketCalendarType.US)

    def test_is_trading_day_with_mocked_calendar(self):
        """Test is_trading_day with mocked calendar."""
        with patch.object(DateUtils, 'get_trading_days') as mock_get_trading_days:
            # Mock trading days
            mock_trading_days = pd.DatetimeIndex([
                pd.Timestamp("2023-12-27"),
                pd.Timestamp("2023-12-28"),
                pd.Timestamp("2023-12-29"),
            ])
            mock_get_trading_days.return_value = mock_trading_days

            # Test trading day
            result = DateUtils.is_trading_day("2023-12-28", MarketCalendarType.US)
            assert result is True

            # Test non-trading day (weekend)
            result = DateUtils.is_trading_day("2023-12-30", MarketCalendarType.US)
            assert result is False

    def test_add_trading_days_basic(self):
        """Test add_trading_days basic functionality."""
        with patch.object(DateUtils, 'get_trading_days') as mock_get_trading_days:
            # Mock trading days
            trading_days = pd.DatetimeIndex(pd.date_range("2023-12-25", "2024-01-05", freq='D'))
            # Remove weekends for simplicity
            trading_days = trading_days[trading_days.dayofweek < 5]
            mock_get_trading_days.return_value = trading_days

            # Add 3 trading days from Dec 27
            start_date = "2023-12-27"
            result = DateUtils.add_trading_days(start_date, 3, MarketCalendarType.US)

            # Should be Dec 28, 29, Jan 2 (skip weekend)
            expected_date = date(2024, 1, 2)
            assert result == expected_date

    def test_count_trading_days_between(self):
        """Test count_trading_days_between."""
        with patch.object(DateUtils, 'get_trading_days') as mock_get_trading_days:
            # Mock trading days
            trading_days = pd.DatetimeIndex([
                pd.Timestamp("2023-12-27"),
                pd.Timestamp("2023-12-28"),
                pd.Timestamp("2023-12-29"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ])
            mock_get_trading_days.return_value = trading_days

            # Count between Dec 27 and Jan 3
            start_date = "2023-12-27"
            end_date = "2024-01-03"
            result = DateUtils.count_trading_days_between(
                start_date, end_date, MarketCalendarType.US
            )

            # Should be 5 days (inclusive of both endpoints)
            assert result == 5

    def test_get_next_trading_day(self):
        """Test get_next_trading_day."""
        with patch.object(DateUtils, 'get_trading_days') as mock_get_trading_days:
            # Mock trading days
            trading_days = pd.DatetimeIndex([
                pd.Timestamp("2023-12-27"),
                pd.Timestamp("2023-12-28"),
                pd.Timestamp("2023-12-29"),
                pd.Timestamp("2024-01-02"),
            ])
            mock_get_trading_days.return_value = trading_days

            # Next trading day after Dec 28 should be Dec 29
            result = DateUtils.get_next_trading_day("2023-12-28", MarketCalendarType.US)
            assert result == date(2023, 12, 29)

            # Next trading day after Dec 30 (weekend) should be Jan 2
            result = DateUtils.get_next_trading_day("2023-12-30", MarketCalendarType.US)
            assert result == date(2024, 1, 2)

    def test_get_previous_trading_day(self):
        """Test get_previous_trading_day."""
        with patch.object(DateUtils, 'get_trading_days') as mock_get_trading_days:
            # Mock trading days
            trading_days = pd.DatetimeIndex([
                pd.Timestamp("2023-12-27"),
                pd.Timestamp("2023-12-28"),
                pd.Timestamp("2023-12-29"),
                pd.Timestamp("2024-01-02"),
            ])
            mock_get_trading_days.return_value = trading_days

            # Previous trading day before Jan 2 should be Dec 29
            result = DateUtils.get_previous_trading_day("2024-01-02", MarketCalendarType.US)
            assert result == date(2023, 12, 29)

            # Previous trading day before Dec 30 (weekend) should be Dec 29
            result = DateUtils.get_previous_trading_day("2023-12-30", MarketCalendarType.US)
            assert result == date(2023, 12, 29)

    def test_is_holiday(self):
        """Test is_holiday method."""
        # Mock holidays module
        with patch('holidays.US') as mock_us_holidays:
            # Set up mock to treat Jan 1, 2023 as holiday
            mock_us_holidays.return_value = {
                date(2023, 1, 1): "New Year's Day",
                date(2023, 12, 25): "Christmas Day"
            }

            # Test holiday
            result = DateUtils.is_holiday("2023-01-01", "US")
            assert result is True

            # Test non-holiday
            result = DateUtils.is_holiday("2023-01-02", "US")
            assert result is False

    def test_get_business_days_between(self):
        """Test get_business_days_between."""
        # Test with simple date range (excluding weekends)
        start_date = "2023-12-27"  # Wednesday
        end_date = "2024-01-03"    # Wednesday next week

        result = DateUtils.get_business_days_between(start_date, end_date)

        # Should exclude weekends: Dec 27, 28, 29 (Wed-Fri) and Jan 2, 3 (Tue-Wed) = 5 days
        assert result == 5

    def test_get_date_range(self):
        """Test get_date_range."""
        # Test daily frequency
        result = DateUtils.get_date_range("2023-12-01", "2023-12-05", freq="D")
        assert len(result) == 5
        assert result[0] == date(2023, 12, 1)
        assert result[-1] == date(2023, 12, 5)

        # Test business day frequency
        result = DateUtils.get_date_range("2023-12-01", "2023-12-08", freq="B")
        # Should exclude weekends (Dec 2-3 and Dec 9-10)
        assert len(result) == 6  # Dec 1, 4, 5, 6, 7, 8

    def test_convert_timezone(self):
        """Test convert_timezone."""
        # Create a datetime in UTC
        utc_dt = datetime(2023, 12, 31, 12, 0, 0)

        # Convert to US/Eastern
        result = DateUtils.convert_timezone(utc_dt, "UTC", "US/Eastern")

        # Should be 7 hours earlier (for standard time)
        assert result.hour == 7 or result.hour == 8  # Depending on DST

    def test_get_quarter(self):
        """Test get_quarter method."""
        # Q1
        assert DateUtils.get_quarter("2023-01-15") == 1
        assert DateUtils.get_quarter("2023-03-31") == 1

        # Q2
        assert DateUtils.get_quarter("2023-04-01") == 2
        assert DateUtils.get_quarter("2023-06-30") == 2

        # Q3
        assert DateUtils.get_quarter("2023-07-01") == 3
        assert DateUtils.get_quarter("2023-09-30") == 3

        # Q4
        assert DateUtils.get_quarter("2023-10-01") == 4
        assert DateUtils.get_quarter("2023-12-31") == 4

    def test_get_week_number(self):
        """Test get_week_number method."""
        # Test known dates
        # Jan 1, 2023 is week 52 of 2022 (ISO week)
        assert DateUtils.get_week_number("2023-01-01") == 52

        # Jan 2, 2023 is week 1 of 2023
        assert DateUtils.get_week_number("2023-01-02") == 1

    def test_date_diff(self):
        """Test date_diff method."""
        # Test day difference
        result = DateUtils.date_diff("2023-12-31", "2023-12-01", unit="D")
        assert result == 30

        # Test business day difference
        result = DateUtils.date_diff("2023-12-08", "2023-12-01", unit="B")
        # Dec 1-8 excluding weekends (Dec 2-3) = 6 business days
        assert result == 6

    def test_get_month_start_end(self):
        """Test get_month_start_end method."""
        start, end = DateUtils.get_month_start_end("2023-12-15")

        assert start == date(2023, 12, 1)
        assert end == date(2023, 12, 31)

    def test_get_year_start_end(self):
        """Test get_year_start_end method."""
        start, end = DateUtils.get_year_start_end("2023-06-15")

        assert start == date(2023, 1, 1)
        assert end == date(2023, 12, 31)
"""
Unit tests for the file_utils module.
"""

import pytest
import pandas as pd
import numpy as np
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock

from src.djinn.utils.file_utils import (
    FileUtils,
    FileFormat,
    CacheManager,
)
from src.djinn.utils.exceptions import DataError, CacheError


class TestFileFormat:
    """Test FileFormat constants."""

    def test_file_format_values(self):
        """Test FileFormat string values."""
        assert FileFormat.CSV == "csv"
        assert FileFormat.PARQUET == "parquet"
        assert FileFormat.JSON == "json"
        assert FileFormat.PICKLE == "pickle"
        assert FileFormat.HDF5 == "hdf5"
        assert FileFormat.FEATHER == "feather"


class TestFileUtils:
    """Test FileUtils class."""

    def test_ensure_dir_existing(self):
        """Test ensure_dir with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Directory already exists
            result = FileUtils.ensure_dir(temp_dir)

            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_dir()

    def test_ensure_dir_new(self):
        """Test ensure_dir with new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "subdir" / "nested"

            # Directory doesn't exist yet
            assert not new_dir.exists()

            result = FileUtils.ensure_dir(new_dir)

            assert result.exists()
            assert result.is_dir()

    def test_get_file_hash_success(self):
        """Test get_file_hash with existing file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            # Test MD5 hash
            md5_hash = FileUtils.get_file_hash(temp_file, algorithm="md5")
            assert len(md5_hash) == 32  # MD5 produces 32 hex characters

            # Test SHA256 hash
            sha256_hash = FileUtils.get_file_hash(temp_file, algorithm="sha256")
            assert len(sha256_hash) == 64  # SHA256 produces 64 hex characters

        finally:
            os.unlink(temp_file)

    def test_get_file_hash_file_not_found(self):
        """Test get_file_hash with non-existent file."""
        non_existent = "/tmp/non_existent_file_12345.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            FileUtils.get_file_hash(non_existent)

        assert "文件不存在" in str(exc_info.value)

    def test_read_dataframe_csv(self):
        """Test read_dataframe with CSV file."""
        # Create CSV content
        csv_content = """col1,col2,col3
1,2,3
4,5,6
7,8,9"""

        mock_file = mock_open(read_data=csv_content)
        with patch('builtins.open', mock_file):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_df = pd.DataFrame({'col1': [1, 4, 7], 'col2': [2, 5, 8], 'col3': [3, 6, 9]})
                mock_read_csv.return_value = mock_df

                result = FileUtils.read_dataframe("/tmp/test.csv")

                mock_read_csv.assert_called_once()
                pd.testing.assert_frame_equal(result, mock_df)

    def test_read_dataframe_parquet(self):
        """Test read_dataframe with Parquet file."""
        mock_df = pd.DataFrame({'A': [1, 2, 3]})

        with patch('pandas.read_parquet') as mock_read_parquet:
            mock_read_parquet.return_value = mock_df

            result = FileUtils.read_dataframe("/tmp/test.parquet")

            mock_read_parquet.assert_called_once()
            pd.testing.assert_frame_equal(result, mock_df)

    def test_read_dataframe_json(self):
        """Test read_dataframe with JSON file."""
        mock_df = pd.DataFrame({'A': [1, 2, 3]})

        with patch('pandas.read_json') as mock_read_json:
            mock_read_json.return_value = mock_df

            result = FileUtils.read_dataframe("/tmp/test.json")

            mock_read_json.assert_called_once()
            pd.testing.assert_frame_equal(result, mock_df)

    def test_write_dataframe_csv(self):
        """Test write_dataframe with CSV format."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                FileUtils.write_dataframe(df, "/tmp/test.csv")

                mock_to_csv.assert_called_once()

    def test_write_dataframe_parquet(self):
        """Test write_dataframe with Parquet format."""
        df = pd.DataFrame({'A': [1, 2, 3]})

        with patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            FileUtils.write_dataframe(df, "/tmp/test.parquet")

            mock_to_parquet.assert_called_once()

    def test_safe_write_atomic(self):
        """Test safe_write with atomic replacement."""
        content = "test content"

        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "test.txt"

            # Write file atomically
            FileUtils.safe_write(target_file, content)

            # Verify content
            with open(target_file, 'r') as f:
                assert f.read() == content

            # Verify backup was created (temporary file during write)
            # The implementation should use a temporary file first

    def test_read_json_success(self):
        """Test read_json with valid JSON file."""
        json_content = {"key": "value", "number": 123, "list": [1, 2, 3]}

        mock_file = mock_open(read_data=json.dumps(json_content))
        with patch('builtins.open', mock_file):
            result = FileUtils.read_json("/tmp/test.json")

            assert result == json_content

    def test_read_json_invalid(self):
        """Test read_json with invalid JSON."""
        invalid_json = "{invalid json"

        mock_file = mock_open(read_data=invalid_json)
        with patch('builtins.open', mock_file):
            with pytest.raises(DataError) as exc_info:
                FileUtils.read_json("/tmp/test.json")

            assert "JSON解析失败" in str(exc_info.value)

    def test_write_json(self):
        """Test write_json."""
        data = {"key": "value", "nested": {"item": 123}}

        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch('json.dump') as mock_json_dump:
                FileUtils.write_json(data, "/tmp/test.json")

                mock_json_dump.assert_called_once_with(
                    data,
                    mock_file(),
                    indent=2,
                    ensure_ascii=False
                )

    def test_read_pickle(self):
        """Test read_pickle."""
        test_data = {"key": "value", "list": [1, 2, 3]}

        mock_file = mock_open(read_data=pickle.dumps(test_data))
        with patch('builtins.open', mock_file):
            with patch('pickle.load') as mock_pickle_load:
                mock_pickle_load.return_value = test_data

                result = FileUtils.read_pickle("/tmp/test.pkl")

                mock_pickle_load.assert_called_once()
                assert result == test_data

    def test_write_pickle(self):
        """Test write_pickle."""
        data = {"key": "value"}

        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch('pickle.dump') as mock_pickle_dump:
                FileUtils.write_pickle(data, "/tmp/test.pkl")

                mock_pickle_dump.assert_called_once_with(
                    data,
                    mock_file(),
                    protocol=pickle.HIGHEST_PROTOCOL
                )

    def test_list_files(self):
        """Test list_files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = ["file1.txt", "file2.csv", "file3.json"]
            for filename in test_files:
                (Path(temp_dir) / filename).touch()

            # Create a subdirectory with more files
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").touch()

            # List all files
            result = FileUtils.list_files(temp_dir, recursive=True)
            assert len(result) == 4  # 3 files + 1 nested

            # List with pattern
            result = FileUtils.list_files(temp_dir, pattern="*.txt")
            assert len(result) == 1  # Only file1.txt at root

            # List non-recursive
            result = FileUtils.list_files(temp_dir, recursive=False)
            assert len(result) == 3  # Only root files

    def test_copy_file(self):
        """Test copy_file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.txt"
            target = Path(temp_dir) / "target.txt"

            # Create source file
            source.write_text("test content")

            # Copy file
            FileUtils.copy_file(source, target)

            # Verify copy
            assert target.exists()
            assert target.read_text() == "test content"

    def test_move_file(self):
        """Test move_file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.txt"
            target = Path(temp_dir) / "target.txt"

            source.write_text("test content")

            # Move file
            FileUtils.move_file(source, target)

            # Verify move
            assert not source.exists()
            assert target.exists()
            assert target.read_text() == "test content"

    def test_delete_file(self):
        """Test delete_file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            file_path.write_text("content")

            assert file_path.exists()

            # Delete file
            FileUtils.delete_file(file_path)

            assert not file_path.exists()

    def test_file_exists(self):
        """Test file_exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = Path(temp_dir) / "exists.txt"
            existing_file.touch()

            non_existing = Path(temp_dir) / "not_exists.txt"

            assert FileUtils.file_exists(existing_file) is True
            assert FileUtils.file_exists(non_existing) is False

    def test_get_file_size(self):
        """Test get_file_size."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            size = FileUtils.get_file_size(temp_file)
            assert size > 0
            assert isinstance(size, int)

        finally:
            os.unlink(temp_file)

    def test_get_file_modified_time(self):
        """Test get_file_modified_time."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        try:
            mtime = FileUtils.get_file_modified_time(temp_file)
            assert isinstance(mtime, datetime)

        finally:
            os.unlink(temp_file)


class TestCacheManager:
    """Test CacheManager class."""

    def test_cache_manager_initialization(self):
        """Test CacheManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"

            manager = CacheManager(cache_dir=cache_dir)

            assert manager.cache_dir == cache_dir
            assert isinstance(manager.memory_cache, dict)
            assert cache_dir.exists()

    def test_generate_cache_key(self):
        """Test generate_cache_key."""
        manager = CacheManager()

        # Test with simple data
        key1 = manager.generate_cache_key("test", {"param": "value"})
        key2 = manager.generate_cache_key("test", {"param": "value"})

        # Same inputs should generate same key
        assert key1 == key2

        # Different inputs should generate different keys
        key3 = manager.generate_cache_key("test", {"param": "different"})
        assert key1 != key3

        # Different function names should generate different keys
        key4 = manager.generate_cache_key("other_func", {"param": "value"})
        assert key1 != key4

    def test_is_cached_memory(self):
        """Test is_cached with memory cache."""
        manager = CacheManager()

        # Not in cache initially
        key = "test_key"
        assert manager.is_cached(key) is False

        # Add to memory cache
        manager.memory_cache[key] = {"data": "value", "timestamp": datetime.now()}
        assert manager.is_cached(key) is True

    def test_is_cached_disk(self):
        """Test is_cached with disk cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=Path(temp_dir))

            key = "test_key"
            cache_file = manager.cache_dir / f"{key}.pkl"

            # Not on disk initially
            assert manager.is_cached(key, check_disk=True) is False

            # Create cache file
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({"data": "value", "timestamp": datetime.now()}, f)

            # Should be found on disk
            assert manager.is_cached(key, check_disk=True) is True

    def test_get_cached_memory(self):
        """Test get_cached from memory."""
        manager = CacheManager()

        key = "test_key"
        test_data = {"result": "data", "metadata": "info"}

        # Add to memory cache
        manager.memory_cache[key] = {
            "data": test_data,
            "timestamp": datetime.now(),
            "metadata": {}
        }

        # Retrieve from cache
        result = manager.get_cached(key)

        assert result == test_data

    def test_get_cached_not_found(self):
        """Test get_cached with non-existent key."""
        manager = CacheManager()

        result = manager.get_cached("non_existent_key")
        assert result is None

    def test_cache_data_memory(self):
        """Test cache_data to memory."""
        manager = CacheManager()

        key = "test_key"
        data = {"result": "test data"}

        # Cache data
        manager.cache_data(key, data, ttl=3600)

        # Verify in memory cache
        assert key in manager.memory_cache
        cached_item = manager.memory_cache[key]
        assert cached_item["data"] == data
        assert isinstance(cached_item["timestamp"], datetime)

    def test_cache_data_disk(self):
        """Test cache_data to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=Path(temp_dir))

            key = "test_key"
            data = {"result": "test data"}

            # Cache to disk
            manager.cache_data(key, data, ttl=3600, persist=True)

            # Verify disk cache file exists
            cache_file = manager.cache_dir / f"{key}.pkl"
            assert cache_file.exists()

            # Verify content
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                assert cached_data["data"] == data

    def test_invalidate_cache_memory(self):
        """Test invalidate_cache for memory."""
        manager = CacheManager()

        key = "test_key"
        manager.memory_cache[key] = {"data": "value", "timestamp": datetime.now()}

        # Verify in cache
        assert key in manager.memory_cache

        # Invalidate
        manager.invalidate_cache(key)

        # Should be removed
        assert key not in manager.memory_cache

    def test_invalidate_cache_disk(self):
        """Test invalidate_cache for disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=Path(temp_dir))

            key = "test_key"
            cache_file = manager.cache_dir / f"{key}.pkl"

            # Create cache file
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.touch()

            assert cache_file.exists()

            # Invalidate
            manager.invalidate_cache(key, persist=True)

            # File should be deleted
            assert not cache_file.exists()

    def test_clear_cache(self):
        """Test clear_cache."""
        manager = CacheManager()

        # Add some cache entries
        manager.memory_cache["key1"] = {"data": "value1", "timestamp": datetime.now()}
        manager.memory_cache["key2"] = {"data": "value2", "timestamp": datetime.now()}

        assert len(manager.memory_cache) == 2

        # Clear cache
        manager.clear_cache()

        assert len(manager.memory_cache) == 0

    def test_cache_ttl_expired(self):
        """Test TTL expiration in cache."""
        manager = CacheManager()

        key = "test_key"
        data = {"result": "data"}

        # Cache with short TTL
        past_time = datetime.now() - timedelta(seconds=100)  # 100 seconds ago
        manager.memory_cache[key] = {
            "data": data,
            "timestamp": past_time,
            "metadata": {"ttl": 60}  # 60 second TTL (expired)
        }

        # Should not be considered cached (expired)
        assert manager.is_cached(key) is False

        # Should not be retrievable
        result = manager.get_cached(key)
        assert result is None

    def test_cache_ttl_valid(self):
        """Test TTL still valid in cache."""
        manager = CacheManager()

        key = "test_key"
        data = {"result": "data"}

        # Cache with long TTL
        recent_time = datetime.now() - timedelta(seconds=30)  # 30 seconds ago
        manager.memory_cache[key] = {
            "data": data,
            "timestamp": recent_time,
            "metadata": {"ttl": 3600}  # 1 hour TTL (still valid)
        }

        # Should be considered cached
        assert manager.is_cached(key) is True

        # Should be retrievable
        result = manager.get_cached(key)
        assert result == data
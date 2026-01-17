"""
Djinn 文件工具模块。

这个模块提供了文件操作和数据处理相关的功能。
"""

import os
import json
import pickle
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .logger import logger
from .exceptions import DataError, CacheError


class FileFormat(str):
    """文件格式类。"""

    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    PICKLE = "pickle"
    HDF5 = "hdf5"
    FEATHER = "feather"


class FileUtils:
    """文件工具类。"""

    @staticmethod
    def ensure_dir(directory: Union[str, Path]) -> Path:
        """
        确保目录存在，如果不存在则创建。

        Args:
            directory: 目录路径

        Returns:
            Path: 目录的 Path 对象
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @staticmethod
    def get_file_hash(
        file_path: Union[str, Path],
        algorithm: str = "md5",
        chunk_size: int = 8192,
    ) -> str:
        """
        计算文件的哈希值。

        Args:
            file_path: 文件路径
            algorithm: 哈希算法 (md5, sha1, sha256)
            chunk_size: 读取块大小

        Returns:
            str: 文件的哈希值

        Raises:
            FileNotFoundError: 如果文件不存在
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    @staticmethod
    def read_dataframe(
        file_path: Union[str, Path],
        file_format: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        读取数据框文件。

        Args:
            file_path: 文件路径
            file_format: 文件格式，如果为 None 则从扩展名推断
            **kwargs: 传递给读取函数的额外参数

        Returns:
            pd.DataFrame: 读取的数据框

        Raises:
            DataError: 如果读取失败
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 推断文件格式
        if file_format is None:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                file_format = FileFormat.CSV
            elif suffix == ".parquet":
                file_format = FileFormat.PARQUET
            elif suffix == ".json":
                file_format = FileFormat.JSON
            elif suffix in [".pkl", ".pickle"]:
                file_format = FileFormat.PICKLE
            elif suffix in [".h5", ".hdf5"]:
                file_format = FileFormat.HDF5
            elif suffix == ".feather":
                file_format = FileFormat.FEATHER
            else:
                raise DataError(
                    f"无法推断文件格式: {file_path}",
                    details={"suffix": suffix},
                )

        try:
            if file_format == FileFormat.CSV:
                df = pd.read_csv(file_path, **kwargs)
            elif file_format == FileFormat.PARQUET:
                df = pd.read_parquet(file_path, **kwargs)
            elif file_format == FileFormat.JSON:
                df = pd.read_json(file_path, **kwargs)
            elif file_format == FileFormat.PICKLE:
                with open(file_path, "rb") as f:
                    df = pickle.load(f)
            elif file_format == FileFormat.HDF5:
                df = pd.read_hdf(file_path, **kwargs)
            elif file_format == FileFormat.FEATHER:
                df = pd.read_feather(file_path, **kwargs)
            else:
                raise DataError(
                    f"不支持的文件格式: {file_format}",
                    details={"file_path": str(file_path)},
                )

            logger.debug(f"成功读取文件: {file_path}, 形状: {df.shape}")
            return df

        except Exception as e:
            raise DataError(
                f"读取文件失败: {file_path}",
                details={
                    "file_format": file_format,
                    "error": str(e),
                },
            )

    @staticmethod
    def write_dataframe(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        file_format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        写入数据框到文件。

        Args:
            df: 要写入的数据框
            file_path: 文件路径
            file_format: 文件格式，如果为 None 则从扩展名推断
            **kwargs: 传递给写入函数的额外参数

        Raises:
            DataError: 如果写入失败
        """
        file_path = Path(file_path)

        # 确保目录存在
        FileUtils.ensure_dir(file_path.parent)

        # 推断文件格式
        if file_format is None:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                file_format = FileFormat.CSV
            elif suffix == ".parquet":
                file_format = FileFormat.PARQUET
            elif suffix == ".json":
                file_format = FileFormat.JSON
            elif suffix in [".pkl", ".pickle"]:
                file_format = FileFormat.PICKLE
            elif suffix in [".h5", ".hdf5"]:
                file_format = FileFormat.HDF5
            elif suffix == ".feather":
                file_format = FileFormat.FEATHER
            else:
                # 默认使用 parquet
                file_format = FileFormat.PARQUET
                file_path = file_path.with_suffix(".parquet")

        try:
            if file_format == FileFormat.CSV:
                df.to_csv(file_path, **kwargs)
            elif file_format == FileFormat.PARQUET:
                df.to_parquet(file_path, **kwargs)
            elif file_format == FileFormat.JSON:
                df.to_json(file_path, **kwargs)
            elif file_format == FileFormat.PICKLE:
                with open(file_path, "wb") as f:
                    pickle.dump(df, f, **kwargs)
            elif file_format == FileFormat.HDF5:
                df.to_hdf(file_path, key="data", **kwargs)
            elif file_format == FileFormat.FEATHER:
                df.to_feather(file_path, **kwargs)
            else:
                raise DataError(
                    f"不支持的文件格式: {file_format}",
                    details={"file_path": str(file_path)},
                )

            logger.debug(f"成功写入文件: {file_path}, 形状: {df.shape}")

        except Exception as e:
            raise DataError(
                f"写入文件失败: {file_path}",
                details={
                    "file_format": file_format,
                    "error": str(e),
                },
            )

    @staticmethod
    def read_json(file_path: Union[str, Path], **kwargs) -> Any:
        """
        读取 JSON 文件。

        Args:
            file_path: 文件路径
            **kwargs: 传递给 json.load 的额外参数

        Returns:
            Any: JSON 数据
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f, **kwargs)
            logger.debug(f"成功读取 JSON 文件: {file_path}")
            return data
        except Exception as e:
            raise DataError(
                f"读取 JSON 文件失败: {file_path}",
                details={"error": str(e)},
            )

    @staticmethod
    def write_json(
        data: Any,
        file_path: Union[str, Path],
        indent: int = 2,
        **kwargs,
    ) -> None:
        """
        写入 JSON 文件。

        Args:
            data: 要写入的数据
            file_path: 文件路径
            indent: 缩进空格数
            **kwargs: 传递给 json.dump 的额外参数
        """
        file_path = Path(file_path)
        FileUtils.ensure_dir(file_path.parent)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, **kwargs)
            logger.debug(f"成功写入 JSON 文件: {file_path}")
        except Exception as e:
            raise DataError(
                f"写入 JSON 文件失败: {file_path}",
                details={"error": str(e)},
            )

    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取文件信息。

        Args:
            file_path: 文件路径

        Returns:
            Dict[str, Any]: 文件信息字典
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        stat = file_path.stat()

        return {
            "path": str(file_path),
            "name": file_path.name,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "accessed": datetime.fromtimestamp(stat.st_atime),
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
            "extension": file_path.suffix,
        }

    @staticmethod
    def find_files(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True,
    ) -> List[Path]:
        """
        查找匹配模式的文件。

        Args:
            directory: 目录路径
            pattern: 文件模式
            recursive: 是否递归查找

        Returns:
            List[Path]: 匹配的文件路径列表
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        # 只返回文件，不包括目录
        files = [f for f in files if f.is_file()]

        logger.debug(f"在 {directory} 中找到 {len(files)} 个匹配 {pattern} 的文件")
        return files

    @staticmethod
    def cleanup_old_files(
        directory: Union[str, Path],
        pattern: str = "*",
        max_age_days: int = 30,
        dry_run: bool = False,
    ) -> List[Path]:
        """
        清理旧文件。

        Args:
            directory: 目录路径
            pattern: 文件模式
            max_age_days: 最大保留天数
            dry_run: 是否只显示而不实际删除

        Returns:
            List[Path]: 被删除的文件列表
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        old_files = []

        for file_path in directory.rglob(pattern):
            if not file_path.is_file():
                continue

            file_info = FileUtils.get_file_info(file_path)
            if file_info["modified"] < cutoff_date:
                old_files.append(file_path)

                if not dry_run:
                    try:
                        file_path.unlink()
                        logger.debug(f"删除旧文件: {file_path}")
                    except Exception as e:
                        logger.error(f"删除文件失败: {file_path}, 错误: {e}")

        if dry_run:
            logger.info(f"找到 {len(old_files)} 个超过 {max_age_days} 天的旧文件")
        else:
            logger.info(f"已删除 {len(old_files)} 个超过 {max_age_days} 天的旧文件")

        return old_files

    @staticmethod
    def create_temp_file(
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[Union[str, Path]] = None,
        delete_on_close: bool = False,
    ) -> Path:
        """
        创建临时文件。

        Args:
            suffix: 文件后缀
            prefix: 文件前缀
            dir: 临时目录
            delete_on_close: 是否在关闭时删除

        Returns:
            Path: 临时文件路径
        """
        if dir is not None:
            dir = str(dir)

        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=dir,
            delete=delete_on_close,
        )
        temp_file.close()  # 关闭文件但不删除

        file_path = Path(temp_file.name)
        logger.debug(f"创建临时文件: {file_path}")

        return file_path

    @staticmethod
    def create_temp_dir(
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        创建临时目录。

        Args:
            suffix: 目录后缀
            prefix: 目录前缀
            dir: 父目录

        Returns:
            Path: 临时目录路径
        """
        if dir is not None:
            dir = str(dir)

        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        dir_path = Path(temp_dir)

        logger.debug(f"创建临时目录: {dir_path}")
        return dir_path

    @staticmethod
    def safe_delete(
        path: Union[str, Path],
        recursive: bool = False,
        force: bool = False,
    ) -> bool:
        """
        安全删除文件或目录。

        Args:
            path: 路径
            recursive: 是否递归删除目录
            force: 是否强制删除（忽略错误）

        Returns:
            bool: 是否成功删除
        """
        path = Path(path)

        if not path.exists():
            logger.warning(f"路径不存在: {path}")
            return False

        try:
            if path.is_file():
                path.unlink()
                logger.debug(f"删除文件: {path}")
                return True

            elif path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                    logger.debug(f"递归删除目录: {path}")
                else:
                    # 只删除空目录
                    try:
                        path.rmdir()
                        logger.debug(f"删除目录: {path}")
                    except OSError as e:
                        if force:
                            shutil.rmtree(path)
                            logger.debug(f"强制删除目录: {path}")
                        else:
                            raise e
                return True

            else:
                logger.warning(f"无法识别的路径类型: {path}")
                return False

        except Exception as e:
            if force:
                logger.warning(f"删除失败但忽略错误: {path}, 错误: {e}")
                return False
            else:
                logger.error(f"删除失败: {path}, 错误: {e}")
                raise


class CacheManager:
    """缓存管理器。"""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        default_ttl: int = 3600,  # 默认1小时
        cleanup_interval: int = 86400,  # 清理间隔24小时
    ):
        """
        初始化缓存管理器。

        Args:
            cache_dir: 缓存目录
            default_ttl: 默认缓存过期时间（秒）
            cleanup_interval: 自动清理间隔（秒）
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        # 确保缓存目录存在
        FileUtils.ensure_dir(self.cache_dir)

        # 元数据文件
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

        # 上次清理时间
        self.last_cleanup = self.metadata.get("last_cleanup", 0)

        logger.info(f"缓存管理器初始化完成，缓存目录: {self.cache_dir}")

    def get(
        self,
        key: str,
        default: Any = None,
        check_expiry: bool = True,
    ) -> Any:
        """
        从缓存获取数据。

        Args:
            key: 缓存键
            default: 默认值
            check_expiry: 是否检查过期

        Returns:
            Any: 缓存数据或默认值
        """
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            return default

        # 检查是否过期
        if check_expiry and self._is_expired(key):
            logger.debug(f"缓存已过期: {key}")
            self.delete(key)
            return default

        try:
            data = FileUtils.read_dataframe(cache_file, FileFormat.PICKLE)
            logger.debug(f"从缓存读取: {key}")
            return data
        except Exception as e:
            logger.error(f"读取缓存失败: {key}, 错误: {e}")
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        设置缓存数据。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 缓存过期时间（秒），如果为 None 使用默认值
        """
        if ttl is None:
            ttl = self.default_ttl

        cache_file = self._get_cache_file(key)

        try:
            # 写入数据
            FileUtils.write_dataframe(value, cache_file, FileFormat.PICKLE)

            # 更新元数据
            expiry_time = datetime.now().timestamp() + ttl
            self.metadata["items"][key] = {
                "expiry_time": expiry_time,
                "created_time": datetime.now().timestamp(),
                "size": cache_file.stat().st_size if cache_file.exists() else 0,
            }

            self._save_metadata()
            logger.debug(f"设置缓存: {key}, TTL: {ttl}秒")

            # 检查是否需要清理
            self._auto_cleanup()

        except Exception as e:
            raise CacheError(
                f"设置缓存失败: {key}",
                cache_key=key,
                details={"error": str(e)},
            )

    def delete(self, key: str) -> bool:
        """
        删除缓存数据。

        Args:
            key: 缓存键

        Returns:
            bool: 是否成功删除
        """
        cache_file = self._get_cache_file(key)

        try:
            if cache_file.exists():
                cache_file.unlink()

            # 从元数据中移除
            if key in self.metadata["items"]:
                del self.metadata["items"][key]
                self._save_metadata()

            logger.debug(f"删除缓存: {key}")
            return True

        except Exception as e:
            logger.error(f"删除缓存失败: {key}, 错误: {e}")
            return False

    def clear(self) -> None:
        """清除所有缓存。"""
        try:
            # 删除所有缓存文件
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

            # 重置元数据
            self.metadata = {"items": {}, "last_cleanup": 0}
            self._save_metadata()

            logger.info("清除所有缓存")
        except Exception as e:
            raise CacheError(
                "清除缓存失败",
                details={"error": str(e)},
            )

    def cleanup_expired(self) -> List[str]:
        """
        清理过期的缓存。

        Returns:
            List[str]: 被清理的缓存键列表
        """
        expired_keys = []
        current_time = datetime.now().timestamp()

        for key, item_info in list(self.metadata["items"].items()):
            if current_time > item_info["expiry_time"]:
                if self.delete(key):
                    expired_keys.append(key)

        # 更新清理时间
        self.metadata["last_cleanup"] = current_time
        self._save_metadata()

        logger.info(f"清理了 {len(expired_keys)} 个过期的缓存项")
        return expired_keys

    def get_info(self) -> Dict[str, Any]:
        """
        获取缓存信息。

        Returns:
            Dict[str, Any]: 缓存信息
        """
        total_size = 0
        current_time = datetime.now().timestamp()

        for item_info in self.metadata["items"].values():
            total_size += item_info.get("size", 0)

        expired_count = sum(
            1 for item_info in self.metadata["items"].values()
            if current_time > item_info["expiry_time"]
        )

        return {
            "total_items": len(self.metadata["items"]),
            "expired_items": expired_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "last_cleanup": datetime.fromtimestamp(
                self.metadata.get("last_cleanup", 0)
            ).isoformat(),
        }

    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径。"""
        # 使用哈希作为文件名，避免特殊字符问题
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def _is_expired(self, key: str) -> bool:
        """检查缓存是否过期。"""
        if key not in self.metadata["items"]:
            return True

        item_info = self.metadata["items"][key]
        current_time = datetime.now().timestamp()

        return current_time > item_info["expiry_time"]

    def _load_metadata(self) -> Dict[str, Any]:
        """加载元数据。"""
        if self.metadata_file.exists():
            try:
                return FileUtils.read_json(self.metadata_file)
            except Exception as e:
                logger.error(f"加载缓存元数据失败: {e}")
                return {"items": {}, "last_cleanup": 0}
        else:
            return {"items": {}, "last_cleanup": 0}

    def _save_metadata(self) -> None:
        """保存元数据。"""
        try:
            FileUtils.write_json(self.metadata, self.metadata_file)
        except Exception as e:
            logger.error(f"保存缓存元数据失败: {e}")

    def _auto_cleanup(self) -> None:
        """自动清理过期缓存。"""
        current_time = datetime.now().timestamp()

        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_expired()
            self.last_cleanup = current_time


# 导出
__all__ = [
    "FileFormat",
    "FileUtils",
    "CacheManager",
]
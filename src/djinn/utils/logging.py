"""日志工具:统一命名空间与格式,供全模块复用。

设计:
- 以 ``djinn`` 为根命名空间,子模块用 ``get_logger(__name__)`` 自动分层。
- 默认 WARNING,通过 ``DJINN_LOG_LEVEL`` env 或 :func:`set_log_level` 调整。
- 测试时可注入 :class:`logging.NullHandler` 避免污染输出。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_ROOT_NAME: Final[str] = "djinn"
_CONFIGURED = False


def _configure_root() -> None:
    """配置根 logger(仅一次)。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger(_ROOT_NAME)
    level_name = os.environ.get("DJINN_LOG_LEVEL", "WARNING").upper()
    root.setLevel(getattr(logging, level_name, logging.WARNING))
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s")
        )
        root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """获取子 logger。``name`` 通常传 ``__name__``。"""
    _configure_root()
    if name is None or name == "__main__":
        return logging.getLogger(_ROOT_NAME)
    if not name.startswith(_ROOT_NAME):
        name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(name)


def set_log_level(level: str | int) -> None:
    """运行时调整日志级别(如 CLI ``--log-level DEBUG``)。"""
    _configure_root()
    root = logging.getLogger(_ROOT_NAME)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)
    root.setLevel(level)

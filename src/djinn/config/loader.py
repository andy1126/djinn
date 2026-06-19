"""配置加载:YAML → BacktestConfig,支持 env 覆盖。

env 覆盖规则:``DJINN_<SECTION>_<FIELD>`` 覆盖对应字段(优先级 env > yaml > 默认)。
例如 ``DJINN_ACCOUNT_INITIAL_CASH=200000``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from djinn.config.models import BacktestConfig
from djinn.utils.exceptions import ConfigError
from djinn.utils.logging import get_logger

_log = get_logger(__name__)

_ENV_PREFIX = "DJINN_"


def _coerce(value: str) -> str | int | float | bool | None:
    """把 env 字符串转为合适的标量类型。"""
    s = value.strip()
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False
    if s.lower() in ("none", "null", ""):
        return None
    # int
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """把 DJINN_ 前缀的 env 变量合并进配置 dict。

    路径按 ``_`` 拆分,但优先匹配已知(含下划线的)字段名,避免
    ``INITIAL_CASH`` 被误拆成 ``initial`` + ``cash``。
    """
    out = dict(data)
    # 收集顶层 section 的已知字段,用于消歧
    from djinn.config.models import BacktestConfig

    section_fields: dict[str, set[str]] = {}
    for name, field in BacktestConfig.model_fields.items():
        ann = field.annotation
        if ann is not None and hasattr(ann, "model_fields"):
            section_fields[name] = set(ann.model_fields.keys())

    for key, val in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        path = key[len(_ENV_PREFIX) :].lower().split("_")
        if len(path) < 2:
            continue
        # 若首段是已知 section,在剩余部分里贪心匹配其字段名
        resolved: list[str] = []
        if path[0] in section_fields:
            resolved.append(path[0])
            rest = path[1:]
            fields = section_fields[path[0]]
            # 贪心:从最长可能字段名匹配
            i = 0
            while i < len(rest):
                matched = None
                for j in range(len(rest), i, -1):
                    candidate = "_".join(rest[i:j])
                    if candidate in fields:
                        matched = candidate
                        break
                if matched:
                    resolved.append(matched)
                    i += len(matched.split("_"))
                else:
                    resolved.append(rest[i])
                    i += 1
        else:
            resolved = path
        if len(resolved) < 2:
            continue
        cur = out
        for p in resolved[:-1]:
            if not isinstance(cur.get(p), dict):
                cur[p] = {}
            cur = cur[p]
        cur[resolved[-1]] = _coerce(val)
    return out


def load_config(
    path: str | Path | None = None, *, data: dict[str, Any] | None = None
) -> BacktestConfig:
    """加载并校验配置。

    Args:
        path: YAML 文件路径(与 data 二选一)。
        data: 已解析的配置 dict。
    """
    if data is None:
        if path is None:
            raise ConfigError("需提供 path 或 data")
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"配置文件不存在: {p}")
        try:
            with p.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML 解析失败: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError("配置根必须是映射(dict)")
    else:
        raw = dict(data)
    raw = _apply_env_overrides(raw)
    try:
        return BacktestConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"配置校验失败: {e}") from e


def dump_config(cfg: BacktestConfig, path: str | Path) -> Path:
    """把 BacktestConfig 导出为 YAML(供 CLI / 前端互导)。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(mode="json")
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    _log.info("配置已导出: %s", p)
    return p

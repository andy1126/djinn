"""
技术指标抽象基类和注册表模块。

本模块定义技术指标的抽象接口和注册管理机制，支持指标的统一管理和参数化配置。

目的：
1. 提供技术指标的标准化抽象接口，确保指标实现的一致性
2. 实现指标注册表，支持动态发现和参数管理
3. 提供缓存装饰器，透明加速指标计算
4. 支持复杂状态化指标（如需要内部状态维护的指标）

实现方案：
1. Indicator抽象基类：定义calculate、get_params、get_type等标准方法
2. IndicatorRegistry：单例注册表，管理指标名称、类和参数模板
3. CachedIndicator装饰器：为任何指标函数提供透明缓存
4. ParameterValidator：指标参数验证和标准化

使用方法：
1. 继承Indicator基类实现自定义指标：class MyIndicator(Indicator): ...
2. 使用注册表注册指标：IndicatorRegistry.register('my_indicator', MyIndicator)
3. 通过注册表创建指标实例：indicator = IndicatorRegistry.create('my_indicator', **params)
4. 使用缓存装饰器：@CachedIndicator(maxsize=128)
"""

import abc
import functools
import hashlib
import pickle
from typing import Dict, Any, Callable, Optional, Type, Union, List
from enum import Enum
import pandas as pd
import numpy as np

from ...utils.logger import get_logger
from ...utils.exceptions import IndicatorError, ValidationError

logger = get_logger(__name__)


class IndicatorType(Enum):
    """
    指标类型枚举。

    继承自str和Enum，支持字符串比较和序列化。
    """
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    CUSTOM = "custom"


class Indicator(abc.ABC):
    """
    技术指标抽象基类。

    所有技术指标的统一接口，支持参数化配置和状态管理。

    目的：
    1. 标准化指标接口，确保实现一致性
    2. 支持参数验证和类型提示
    3. 提供指标元数据和状态管理
    4. 便于集成到策略框架中

    实现方案：
    1. 抽象方法calculate：执行指标计算，返回pd.Series或pd.DataFrame
    2. 具体方法get_params：返回指标参数字典
    3. 具体方法get_type：返回指标类型枚举
    4. 可选状态管理：支持需要维护内部状态的复杂指标

    使用方法：
    1. 继承Indicator并实现calculate方法
    2. 在__init__中初始化参数和状态
    3. 使用ParameterValidator验证参数
    4. 注册到IndicatorRegistry供全局使用
    """

    def __init__(self, **params):
        """
        初始化指标实例。

        参数：
            **params: 指标参数字典，具体参数由子类定义

        实现方案：
        1. 存储原始参数
        2. 调用_validate_params验证参数有效性
        3. 初始化内部状态变量
        4. 设置指标名称和元数据
        """
        self._params = self._validate_params(params)
        self._state: Dict[str, Any] = {}
        self._initialized = False

    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        抽象方法：计算指标值。

        目的：
        1. 执行核心指标计算逻辑
        2. 处理输入数据，生成指标值序列
        3. 维护内部状态（对于状态化指标）

        参数：
            data: 输入数据DataFrame，通常包含OHLCV列

        返回：
            Union[pd.Series, pd.DataFrame]: 指标值序列或数据框
            对于单输出指标返回Series，多输出指标返回DataFrame

        实现方案：
        1. 子类必须实现此方法
        2. 支持向量化计算以提高性能
        3. 正确处理缺失值和边界条件
        4. 更新内部状态（如需要）
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        获取指标参数。

        返回：
            Dict[str, Any]: 指标参数字典
        """
        return self._params.copy()

    def get_type(self) -> IndicatorType:
        """
        获取指标类型。

        返回：
            IndicatorType: 指标类型枚举

        默认实现：
            返回IndicatorType.CUSTOM，子类应重写此方法返回具体类型
        """
        return IndicatorType.CUSTOM

    def reset(self) -> None:
        """
        重置指标状态。

        目的：
        1. 清空内部状态，恢复初始状态
        2. 支持回测中的多次使用
        3. 释放可能的内存占用
        """
        self._state.clear()
        self._initialized = False
        logger.debug(f"指标 {self.__class__.__name__} 状态已重置")

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证指标参数。

        目的：
        1. 确保参数类型和范围有效
        2. 提供默认值处理缺失参数
        3. 转换参数类型（如字符串转数值）

        参数：
            params: 原始参数字典

        返回：
            Dict[str, Any]: 验证后的参数字典

        实现方案：
        1. 子类可重写此方法实现特定验证逻辑
        2. 使用ParameterValidator进行标准验证
        3. 记录验证日志便于调试
        """
        # 基本验证：确保参数为字典
        if not isinstance(params, dict):
            raise ValidationError(
                f"指标参数必须是字典类型，实际类型: {type(params)}",
                field="params",
                value=params
            )

        # 默认实现：直接返回参数，子类应重写
        return params.copy()

    def __repr__(self) -> str:
        """返回指标的可读表示"""
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"{self.__class__.__name__}({params_str})"


class IndicatorRegistry:
    """
    指标注册表。

    单例模式，管理所有注册的指标类和参数模板。

    目的：
    1. 集中管理指标定义，支持动态发现
    2. 提供参数模板和验证规则
    3. 支持指标工厂模式，按需创建实例
    4. 便于指标组合和依赖管理

    实现方案：
    1. 类级字典存储注册信息
    2. 支持别名和版本管理
    3. 参数模板和默认值管理
    4. 工厂方法创建指标实例
    """

    _registry: Dict[str, Dict[str, Any]] = {}
    _instance = None

    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls,
        name: str,
        indicator_class: Type[Indicator],
        default_params: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None
    ) -> None:
        """
        注册指标类。

        参数：
            name: 指标唯一名称
            indicator_class: 指标类（继承自Indicator）
            default_params: 默认参数字典，可选
            aliases: 指标别名列表，可选

        实现方案：
        1. 验证指标类是否继承自Indicator
        2. 存储类引用和默认参数
        3. 注册别名映射
        4. 记录注册日志
        """
        if not issubclass(indicator_class, Indicator):
            raise IndicatorError(
                f"指标类必须继承自Indicator，实际类: {indicator_class}"
            )

        if name in cls._registry:
            logger.warning(f"指标名称 '{name}' 已存在，将被覆盖")

        cls._registry[name] = {
            'class': indicator_class,
            'default_params': default_params or {},
            'aliases': aliases or []
        }

        # 注册别名
        for alias in (aliases or []):
            if alias in cls._registry and alias != name:
                logger.warning(f"指标别名 '{alias}' 已存在，将被覆盖")
            cls._registry[alias] = cls._registry[name]

        logger.info(f"注册指标: {name} ({indicator_class.__name__})")

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        注销指标。

        参数：
            name: 指标名称或别名
        """
        if name in cls._registry:
            entry = cls._registry[name]
            # 如果是主名称，需要同时删除别名
            if 'class' in entry:
                for alias in entry.get('aliases', []):
                    if alias in cls._registry:
                        del cls._registry[alias]
            del cls._registry[name]
            logger.info(f"注销指标: {name}")

    @classmethod
    def create(
        cls,
        name: str,
        **params
    ) -> Indicator:
        """
        创建指标实例。

        参数：
            name: 指标名称或别名
            **params: 指标参数，将覆盖默认参数

        返回：
            Indicator: 指标实例

        实现方案：
        1. 查找注册表，获取指标类和默认参数
        2. 合并默认参数和用户参数（用户参数优先）
        3. 实例化指标类
        4. 返回实例
        """
        if name not in cls._registry:
            raise IndicatorError(f"未注册的指标: {name}")

        entry = cls._registry[name]
        indicator_class = entry['class']
        default_params = entry.get('default_params', {})

        # 合并参数
        merged_params = {**default_params, **params}

        try:
            instance = indicator_class(**merged_params)
            logger.debug(f"创建指标实例: {name} with params: {merged_params}")
            return instance
        except Exception as e:
            raise IndicatorError(
                f"创建指标实例失败: {name}",
                details={"params": merged_params, "error": str(e)}
            )

    @classmethod
    def list_indicators(cls) -> List[str]:
        """
        列出所有注册的指标名称（不包括别名）。

        返回：
            List[str]: 指标名称列表
        """
        # 只返回主名称（包含'class'键的条目）
        return [name for name, entry in cls._registry.items()
                if 'class' in entry and entry.get('class') is not None]

    @classmethod
    def get_indicator_info(cls, name: str) -> Dict[str, Any]:
        """
        获取指标详细信息。

        参数：
            name: 指标名称或别名

        返回：
            Dict[str, Any]: 指标信息字典
        """
        if name not in cls._registry:
            raise IndicatorError(f"未注册的指标: {name}")

        entry = cls._registry[name].copy()
        # 移除类引用（避免序列化问题）
        if 'class' in entry:
            entry['class_name'] = entry['class'].__name__
            del entry['class']
        return entry

    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._registry.clear()
        logger.info("指标注册表已清空")


class CachedIndicator:
    """
    指标缓存装饰器。

    为指标计算函数提供透明缓存，支持内存和可选磁盘缓存。

    目的：
    1. 加速重复指标计算，避免重复计算
    2. 支持LRU缓存策略，防止内存溢出
    3. 提供缓存统计和监控
    4. 支持多级缓存（内存+磁盘）

    实现方案：
    1. 基于函数参数生成缓存键
    2. 使用字典实现内存缓存
    3. 支持最大缓存大小和LRU淘汰
    4. 提供缓存统计信息
    """

    def __init__(self, maxsize: int = 128, enabled: bool = True):
        """
        初始化缓存装饰器。

        参数：
            maxsize: 最大缓存条目数
            enabled: 是否启用缓存
        """
        self.maxsize = maxsize
        self.enabled = enabled
        self._cache: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
        self._cache_order: List[str] = []  # 用于LRU淘汰

    def __call__(self, func: Callable) -> Callable:
        """
        装饰器调用方法。

        参数：
            func: 被装饰的函数

        返回：
            Callable: 包装函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            cache_key = self._generate_key(func.__name__, *args, **kwargs)

            # 检查缓存
            if cache_key in self._cache:
                self._hits += 1
                result = self._cache[cache_key]
                # 更新LRU顺序（移动到末尾）
                self._cache_order.remove(cache_key)
                self._cache_order.append(cache_key)
                logger.debug(f"缓存命中: {func.__name__} [key: {cache_key[:8]}]")
                return result

            # 缓存未命中
            self._misses += 1
            result = func(*args, **kwargs)

            # 存储结果
            self._cache[cache_key] = result
            self._cache_order.append(cache_key)

            # LRU淘汰
            if len(self._cache) > self.maxsize:
                oldest_key = self._cache_order.pop(0)
                del self._cache[oldest_key]
                logger.debug(f"缓存淘汰: {oldest_key[:8]}")

            logger.debug(f"缓存存储: {func.__name__} [key: {cache_key[:8]}]")
            return result

        return wrapper

    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """
        生成缓存键。

        实现方案：
        1. 基于函数名称和参数生成确定性哈希
        2. 处理pandas对象等不可哈希类型
        3. 确保相同输入产生相同键
        """
        def hash_value(obj):
            """哈希值辅助函数"""
            if isinstance(obj, pd.Series):
                # 对Series使用形状和内容哈希
                try:
                    value_hash = hashlib.md5(obj.values.tobytes()).hexdigest()[:16]
                    index_hash = hashlib.md5(pickle.dumps(obj.index)).hexdigest()[:16]
                    return f"series_{obj.shape}_{index_hash}_{value_hash}"
                except Exception:
                    return f"series_{id(obj)}_{obj.shape}"
            elif isinstance(obj, pd.DataFrame):
                try:
                    col_hashes = []
                    for col in obj.columns:
                        col_obj = obj[col]
                        if pd.api.types.is_numeric_dtype(col_obj):
                            col_hash = hashlib.md5(col_obj.values.tobytes()).hexdigest()[:12]
                        else:
                            col_hash = hashlib.md5(pickle.dumps(col_obj.values)).hexdigest()[:12]
                        col_hashes.append(f"{col}:{col_hash}")
                    index_hash = hashlib.md5(pickle.dumps(obj.index)).hexdigest()[:12]
                    return f"df_{obj.shape}_{index_hash}_{'_'.join(col_hashes)}"
                except Exception:
                    return f"df_{id(obj)}_{obj.shape}"
            elif isinstance(obj, np.ndarray):
                return f"array_{obj.shape}_{hashlib.md5(obj.tobytes()).hexdigest()[:16]}"
            else:
                try:
                    return str(hash(obj))
                except TypeError:
                    return str(hash(repr(obj)))

        components = [func_name]
        for arg in args:
            components.append(hash_value(arg))
        for key in sorted(kwargs.keys()):
            components.append(f"{key}={hash_value(kwargs[key])}")

        key_string = "|".join(components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._cache_order.clear()
        self._hits = 0
        self._misses = 0
        logger.info("指标缓存已清空")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "enabled": self.enabled,
            "maxsize": self.maxsize,
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "cache_keys": list(self._cache.keys())[:10]  # 前10个键
        }

    def enable(self) -> None:
        """启用缓存"""
        self.enabled = True
        logger.info("指标缓存已启用")

    def disable(self) -> None:
        """禁用缓存"""
        self.enabled = False
        logger.info("指标缓存已禁用")


# ============================================================================
# 便捷函数和默认注册
# ============================================================================

# 创建全局缓存装饰器实例
default_cached_indicator = CachedIndicator(maxsize=128)

# 便捷装饰器函数
def cached_indicator(maxsize: int = 128):
    """创建缓存装饰器的便捷函数"""
    return CachedIndicator(maxsize=maxsize)


# 导出
__all__ = [
    "IndicatorType",
    "Indicator",
    "IndicatorRegistry",
    "CachedIndicator",
    "cached_indicator",
    "default_cached_indicator"
]
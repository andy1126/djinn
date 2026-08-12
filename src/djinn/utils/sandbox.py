"""受限执行沙箱:策略 / 用户指标源码共用的 AST 白名单 + 受限内建。

安全模型属「纵深防御」:AST 层拒绝 import / 危险调用 / 下划线属性访问,
运行时用受限 ``__builtins__`` + 受限 globals。不隔离进程/文件系统。
"""

from __future__ import annotations

import ast
from typing import Any

from djinn.utils.exceptions import StrategyError

# 危险调用名:即使受限 __builtins__ 不含,AST 层也直接拒绝(双保险)。
_DANGEROUS_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "globals",
    "locals",
    "vars",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "breakpoint",
    "exit",
    "quit",
    "super",
    "classmethod",
    "staticmethod",
    "property",
}

# 非下划线开头但具逃逸风险的属性名。
_DANGEROUS_ATTRS = {
    "mro",
    "f_globals",
    "f_locals",
    "gi_frame",
    "gi_code",
    "func_globals",
}

# 受限内建白名单。
SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


class _SandboxValidator(ast.NodeVisitor):
    """AST 白名单校验:收集违规,一次性抛出。"""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def _bad(self, node: ast.AST, msg: str) -> None:
        self.errors.append(f"第 {getattr(node, 'lineno', '?')} 行: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        self._bad(node, "禁止 import")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._bad(node, "禁止 from ... import")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bad(node, "禁止 async def")
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        self._bad(node, "禁止 await")
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self._bad(node, "禁止 yield")
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._bad(node, "禁止 yield from")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_CALLS:
            self._bad(node, f"禁止调用 {node.func.id}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr in _DANGEROUS_ATTRS:
            self._bad(node, f"禁止访问属性 .{node.attr}")
        self.generic_visit(node)


def validate_source(source_code: str) -> ast.Module:
    """解析 + 沙箱校验用户源码;违规抛 :class:`StrategyError`,返回 AST 供 exec。"""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise StrategyError(f"源码语法错误: {e}") from e

    validator = _SandboxValidator()
    validator.visit(tree)
    if validator.errors:
        raise StrategyError("沙箱校验未通过:\n" + "\n".join(validator.errors))
    return tree

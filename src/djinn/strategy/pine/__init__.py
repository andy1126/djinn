"""Pine Script → djinn Python 转译(受支持子集)。

用 :mod:`pynescript`(ANTLR)把 Pine 解析成 AST,再生成 djinn 的 ``signals(self, data)``
模板,落到与内置策略同构的形态,复用 :func:`djinn.strategy.user.compile_user_strategy`
的沙箱与编译。

支持子集:
- ``//@version=`` / ``strategy(...)`` / ``plot(...)`` / ``alert(...)`` → 忽略。
- ``x = input.int/float/bool/string(defval, title, minval=, maxval=)`` → ``param(...)``。
- ``x = ta.sma/ema/rsi/macd/atr/bb/highest/lowest/crossover/crossunder/...`` 等指标赋值。
- 表达式:算术/比较/布尔 ``and/or/not``、三元 ``?:``、``nz()``、``math.abs/min/max``、
  ``series[n]``(历史引用 → ``.shift(n)``)。
- ``if (cond)`` + ``strategy.entry(id, strategy.long)`` → 做多信号;
  ``strategy.close(id)`` → 平仓信号。

不支持(抛带定位的 :class:`StrategyError`):``strategy.exit``、short、``for``/``while``、
``var`` 持久状态、函数定义、元组解构、``request.*``、``switch``。
"""

from __future__ import annotations

from typing import Any

from pynescript.ast import (
    Add,
    And,
    Assign,
    Attribute,
    BinOp,
    BoolOp,
    Call,
    Compare,
    Conditional,
    Constant,
    Div,
    Eq,
    Expr,
    Gt,
    GtE,
    If,
    Lt,
    LtE,
    Mod,
    Mult,
    Name,
    Not,
    NotEq,
    Script,
    Sub,
    Subscript,
    UAdd,
    UnaryOp,
    USub,
    parse,
)

from djinn.utils.exceptions import StrategyError

# Pine ta.* → djinn.indicators 映射。
_TA_MAP: dict[str, str] = {
    "sma": "sma",
    "ema": "ema",
    "wma": "wma",
    "rma": "rma",
    "rsi": "rsi",
    "macd": "macd",
    "stoch": "stoch",
    "cci": "cci",
    "obv": "obv",
    "atr": "atr",
    "bb": "bb",
    "stdev": "stdev",
    "variance": "variance",
    "highest": "highest",
    "lowest": "lowest",
    "change": "change",
    "roc": "roc",
    "mom": "momentum",
    "crossover": "cross_over",
    "crossunder": "cross_under",
    "valuewhen": "valuewhen",
    "barssince": "barssince",
}

_MATH_MAP: dict[str, str] = {
    "abs": "abs",
    "min": "min",
    "max": "max",
    "sqrt": "sqrt",
    "pow": "pow",
    "log": "log",
    "exp": "exp",
    "round": "round",
    "sign": "sign",
}

# 内置序列变量 → 生成的 Python 局部变量名(open 是 Python 内建,改名 open_)。
_BUILTIN_VARS: dict[str, str] = {
    "close": "close",
    "open": "open_",
    "high": "high",
    "low": "low",
    "volume": "volume",
}

_IGNORED_CALLS = {
    "strategy",
    "plot",
    "plotshape",
    "plotchar",
    "alert",
    "bgcolor",
    "fill",
}

_BINOPS: dict[type, str] = {
    Add: "+",
    Sub: "-",
    Mult: "*",
    Div: "/",
    Mod: "%",
}

_CMPS: dict[type, str] = {
    Eq: "==",
    NotEq: "!=",
    Lt: "<",
    LtE: "<=",
    Gt: ">",
    GtE: ">=",
}


class _PineCodeGen:
    def __init__(self) -> None:
        self.params: dict[str, dict[str, Any]] = {}  # name -> {default, min, max, desc}
        self.computed: list[str] = []  # 计算变量名(在 signals 内局部定义)
        self.compute_lines: list[tuple[str, str]] = []  # (target, expr)
        self.signals: list[tuple[str, str]] = []  # (kind, cond_expr)

    # ── 顶层 ────────────────────────────────────────────
    def generate(self, tree: Script) -> str:
        for stmt in tree.body:
            self._stmt(stmt)
        return self._render()

    def _stmt(self, node: Any) -> None:
        if isinstance(node, Expr):
            self._expr_stmt(node.value)
        elif isinstance(node, Assign):
            self._assign(node)
        else:
            raise StrategyError(f"不支持的 Pine 语句: {type(node).__name__}")

    def _expr_stmt(self, value: Any) -> None:
        if isinstance(value, If):
            self._if(value)
            return
        if isinstance(value, Call):
            if isinstance(value.func, Name) and value.func.id in _IGNORED_CALLS:
                return  # strategy()/plot()/alert() 声明忽略
            if self._call_namespace(value) == "strategy":
                self._signal_from_call(value, "always")
                return
        # 其他裸表达式忽略(如注释性计算)
        return

    def _assign(self, node: Assign) -> None:
        target = node.target
        if not isinstance(target, Name):
            raise StrategyError("暂不支持元组/列表解构赋值(如 [a, b] = ta.macd(...))")
        name = target.id
        if isinstance(node.value, Call) and self._is_input_call(node.value):
            self.params[name] = self._input_params(node.value)
            return
        self.computed.append(name)
        self.compute_lines.append((name, self._expr(node.value)))

    def _if(self, node: If) -> None:
        if node.orelse:
            raise StrategyError(
                "暂不支持 if/else 分支(仅支持 if (cond) + strategy.entry/close)"
            )
        cond = self._expr(node.test)
        for stmt in node.body:
            if isinstance(stmt, Expr) and isinstance(stmt.value, Call):
                self._signal_from_call(stmt.value, cond)
            else:
                raise StrategyError("if 体内仅支持 strategy.entry / strategy.close")

    def _signal_from_call(self, call: Call, cond: str) -> None:
        ns = self._call_namespace(call)
        if ns != "strategy":
            raise StrategyError(f"不支持的语句调用: {ns}")
        action = call.func.attr if isinstance(call.func, Attribute) else ""
        if action == "entry":
            direction = self._entry_direction(call)
            if direction != "long":
                raise StrategyError("暂不支持 strategy.short(仅做多/平仓)")
            self.signals.append(("long", cond))
        elif action == "close":
            self.signals.append(("flat", cond))
        elif action == "exit":
            raise StrategyError("暂不支持 strategy.exit(止损/止盈)")
        else:
            raise StrategyError(f"不支持 strategy.{action}")

    def _entry_direction(self, call: Call) -> str:
        # strategy.entry(id, strategy.long) → 第二个参数是 strategy.long/short
        if len(call.args) < 2:
            return "long"
        d = call.args[1].value
        if isinstance(d, Attribute):
            return str(d.attr)
        return "long"

    # ── input / 调用识别 ─────────────────────────────────
    def _is_input_call(self, call: Call) -> bool:
        if isinstance(call.func, Name):
            return str(call.func.id) == "input"
        if isinstance(call.func, Attribute) and isinstance(call.func.value, Name):
            return str(call.func.value.id) == "input"
        return False

    def _input_params(self, call: Call) -> dict[str, Any]:
        args = call.args
        if not args or not isinstance(args[0].value, Constant):
            raise StrategyError("input 默认值必须是常量")
        default = args[0].value.value
        desc = ""
        minimum = None
        maximum = None
        for a in args[1:]:
            if (
                a.name is None
                and isinstance(a.value, Constant)
                and isinstance(a.value.value, str)
            ):
                desc = a.value.value
            elif a.name == "minval" and isinstance(a.value, Constant):
                minimum = a.value.value
            elif a.name == "maxval" and isinstance(a.value, Constant):
                maximum = a.value.value
        return {"default": default, "min": minimum, "max": maximum, "desc": desc}

    def _call_namespace(self, call: Call) -> str:
        if isinstance(call.func, Attribute) and isinstance(call.func.value, Name):
            return str(call.func.value.id)
        if isinstance(call.func, Name):
            return str(call.func.id)
        return ""

    # ── 表达式 ───────────────────────────────────────────
    def _expr(self, node: Any) -> str:
        if isinstance(node, Name):
            return self._name(node.id)
        if isinstance(node, Constant):
            return self._constant(node.value)
        if isinstance(node, BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise StrategyError(f"不支持的运算符: {type(node.op).__name__}")
            return f"({self._expr(node.left)} {op} {self._expr(node.right)})"
        if isinstance(node, BoolOp):
            joiner = " and " if isinstance(node.op, And) else " or "
            return "(" + joiner.join(self._expr(v) for v in node.values) + ")"
        if isinstance(node, UnaryOp):
            if isinstance(node.op, Not):
                return f"(not {self._expr(node.operand)})"
            if isinstance(node.op, USub):
                return f"(-{self._expr(node.operand)})"
            if isinstance(node.op, UAdd):
                return f"(+{self._expr(node.operand)})"
            raise StrategyError(f"不支持的一元运算符: {type(node.op).__name__}")
        if isinstance(node, Compare):
            if len(node.ops) == 1:
                return f"({self._expr(node.left)} {self._cmp_str(node.ops[0])} {self._expr(node.comparators[0])})"
            parts: list[str] = []
            left = node.left
            for op, cmp in zip(node.ops, node.comparators, strict=False):
                parts.append(
                    f"({self._expr(left)} {self._cmp_str(op)} {self._expr(cmp)})"
                )
                left = cmp
            return "(" + " and ".join(parts) + ")"
        if isinstance(node, Conditional):
            return f"({self._expr(node.body)} if {self._expr(node.test)} else {self._expr(node.orelse)})"
        if isinstance(node, Subscript):
            return self._subscript(node)
        if isinstance(node, Call):
            return self._call(node)
        raise StrategyError(f"不支持的 Pine 表达式: {type(node).__name__}")

    def _name(self, name: str) -> str:
        if name in _BUILTIN_VARS:
            return _BUILTIN_VARS[name]
        if name == "hl2":
            return "((high + low) / 2)"
        if name == "hlc3":
            return "((high + low + close) / 3)"
        if name == "ohlc4":
            return "((open_ + high + low + close) / 4)"
        if name in self.params:
            return f"self.{name}"
        if name in self.computed:
            return name
        raise StrategyError(f"未定义变量 {name!r}")

    def _constant(self, value: Any) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, str):
            return repr(value)
        return repr(value)

    def _cmp_str(self, op: Any) -> str:
        s = _CMPS.get(type(op))
        if s is None:
            raise StrategyError(f"不支持的比较运算符: {type(op).__name__}")
        return s

    def _subscript(self, node: Subscript) -> str:
        # Pine 的 series[n] 是历史引用 → .shift(n)
        if isinstance(node.slice, Constant) and isinstance(node.slice.value, int):
            return f"{self._expr(node.value)}.shift({node.slice.value})"
        raise StrategyError("仅支持整型历史引用 series[n]")

    def _call(self, node: Call) -> str:
        ns = self._call_namespace(node)
        if ns == "ta":
            attr = node.func.attr if isinstance(node.func, Attribute) else ""
            mapped = _TA_MAP.get(attr)
            if mapped is None:
                raise StrategyError(f"不支持的 ta.{attr}")
            args = ", ".join(self._expr(a.value) for a in node.args)
            return f"{mapped}({args})"
        if ns == "math":
            attr = node.func.attr if isinstance(node.func, Attribute) else ""
            mapped = _MATH_MAP.get(attr)
            if mapped is None:
                raise StrategyError(f"不支持的 math.{attr}")
            args = ", ".join(self._expr(a.value) for a in node.args)
            return f"{mapped}({args})"
        if ns == "nz":
            x = self._expr(node.args[0].value)
            y = self._expr(node.args[1].value) if len(node.args) > 1 else "0"
            return f"{x}.fillna({y})"
        if ns == "strategy":
            raise StrategyError("strategy.* 只能在 if/语句中调用")
        raise StrategyError(f"不支持的函数调用: {ns}()")

    # ── 渲染 ────────────────────────────────────────────
    def _render(self) -> str:
        lines: list[str] = []
        for name, p in self.params.items():
            parts = [f"{name} = param({self._constant(p['default'])}"]
            if p["min"] is not None:
                parts.append(f", min={self._constant(p['min'])}")
            if p["max"] is not None:
                parts.append(f", max={self._constant(p['max'])}")
            desc = p["desc"] or name
            parts.append(f", description={self._constant(desc)}")
            parts.append(")")
            lines.append("".join(parts))

        if not self.signals:
            raise StrategyError(
                "未发现 strategy.entry / strategy.close,无法生成交易策略"
            )

        lines.append("")
        lines.append("def signals(self, data):")
        lines.append('    close = data["close"]')
        lines.append('    open_ = data["open"]')
        lines.append('    high = data["high"]')
        lines.append('    low = data["low"]')
        lines.append('    volume = data["volume"]')
        for target, expr in self.compute_lines:
            lines.append(f"    {target} = {expr}")
        lines.append("    sig = pd.Series(0, index=data.index, dtype=int)")
        for kind, cond in self.signals:
            if cond == "always":
                lines.append(
                    "    sig.loc[:] = 1" if kind == "long" else "    sig.loc[:] = -1"
                )
            else:
                lines.append(f"    sig[{cond}] = {'1' if kind == 'long' else '-1'}")
        lines.append("    return state_from_signals(sig)")
        return "\n".join(lines) + "\n"


def pine_to_python(source: str) -> str:
    """把 Pine 源码转译为 djinn 的 Python 策略源码。"""
    try:
        tree = parse(source, "<pine>")
    except Exception as e:  # pynescript 可能抛任意解析异常
        raise StrategyError(f"Pine 解析失败: {e}") from e
    if not isinstance(tree, Script):
        raise StrategyError("Pine 解析结果异常")
    return _PineCodeGen().generate(tree)


__all__ = ["pine_to_python"]

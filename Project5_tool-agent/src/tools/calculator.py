"""M1-a：真相计算器（APTX 年龄差、案件编号演算……）。

---- 接口约定（eval/run.py 自检依赖）----
  TOOL_SCHEMA: dict   OpenAI function calling 格式
  run(args: dict) -> str
    args["expression"]: 数学表达式字符串，如 "2 + 3 * 4"
    返回：计算结果的字符串表示（自检期望 "2 + 3 * 4" → 含 "14"）

要求：
  - 支持四则运算 + 常用函数（sqrt / sin / cos / log / pow 等）
  - 禁止任意代码执行（不要用 eval 裸跑用户输入；可用 ast 白名单或简单解析）

主题用途：算 APTX4869 相关年龄差、代号数字、sqrt(4869) 等。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Detective bureau calculator for math expressions. "
            "Use for APTX age gaps, code numbers like 4869, sqrt, etc. "
            "Supports +, -, *, /, **, parentheses, and sqrt/sin/cos/log/abs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": ( 
                        "Math expression, e.g. '17 - 7', '4869 * 2', 'sqrt(4869)'."
                    ),
                },
            },
            "required": ["expression"],
        },
    },
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "pow": pow,
}


def _eval_node(node: ast.AST) -> Any:
    """安全求值 AST 节点。

    TODO(M1-a): 实现白名单求值
      1. Constant / Num → 直接返回数值
      2. BinOp → 查 _BIN_OPS 递归求左右
      3. UnaryOp → 查 _UNARY_OPS
      4. Call → 函数名必须在 _FUNCS，且只允许位置参数
      5. Name → 可允许 pi / e 常量；其余拒绝
      6. 其他节点类型 → raise ValueError
    """
    # ---- 你的代码开始 ----
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    #########################################################
    # 数字字面量（Constant / Num）→ 直接返回数值
    #########################################################
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) or isinstance(node.value, float):
            return node.value
        else:
            raise ValueError(f"不支持的数字类型: {type(node.value)}")
    if isinstance(node, ast.Num):
        if isinstance(node.n, int) or isinstance(node.n, float):
            return node.n
        else:
            raise ValueError(f"不支持的数字类型: {type(node.n)}")
    #########################################################
    # 二元运算（BinOp）→ 查 _BIN_OPS 递归求左右
    #########################################################
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"unsupported operator: {op_type}")
        return _BIN_OPS[op_type](_eval_node(node.left), _eval_node(node.right))
    #########################################################
    # 一元运算（UnaryOp）→ 查 _UNARY_OPS
    #########################################################
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"unsupported unary: {op_type}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))
    #########################################################
    # 函数调用（Call）→ 函数名必须在 _FUNCS，且只允许位置参数
    #########################################################
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only simple function names allowed")
        name = node.func.id
        if name not in _FUNCS:
            raise ValueError(f"function not allowed: {name}")
        if node.keywords:
            raise ValueError("keyword args not allowed")
        args = [_eval_node(a) for a in node.args]
        return _FUNCS[name](*args)
    #########################################################
    # 变量名（Name）→ 可允许 pi / e 常量；其余拒绝
    #########################################################
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise ValueError(f"name not allowed: {node.id}")
    #########################################################
    # 其他节点类型（其他节点类型）→ raise ValueError
    #########################################################
    raise ValueError(f"不支持的节点类型: {type(node)}")
    # ---- 你的代码结束 ----


def run(args: dict) -> str:
    """执行计算器。

    TODO(M1-a):
      1. 从 args 取 expression（缺省时返回错误提示字符串）
      2. ast.parse(expression, mode="eval")
      3. 用 _eval_node 求值
      4. return str(result)
      5. 捕获异常，return f"Error: ..."
    """
    # ---- 你的代码开始 ----
    try:
        expr = args.get("expression")
        if not expr or not isinstance(expr, str):
            return "Error: missing 'expression'"
        tree = ast.parse(expr.strip(), mode="eval")
        result = _eval_node(tree)   # 或 _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    # ---- 你的代码结束 ----

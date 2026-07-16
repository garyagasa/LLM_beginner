"""M1-b：博士的受限 Python 沙箱（阿笠研究所风）。

---- 接口约定（eval/run.py 自检依赖）----
  TOOL_SCHEMA: dict
  run(args: dict) -> str
    args["code"]: 要执行的 Python 代码字符串
    自检：print(sum(range(10))) → 输出中含 "45"

安全要求（tutor 必检）：
  - 限制 / 禁止危险 import（如 os / subprocess / socket）
  - 设置执行超时（建议 2–5 秒）
  - 捕获 stdout（print 输出必须能返回给 agent）
  - 不要用裸 exec 放开整个 builtins

主题用途：批量算年龄表、判断暗号是否回文、小脚本推理。
"""

from __future__ import annotations

from typing import Any

TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "Run short Python snippets in Dr. Agasa's restricted lab sandbox. "
            "Good for loops, age tables, palindrome ciphers. Print to stdout."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute.",
                },
            },
            "required": ["code"],
        },
    },
}

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "print": print,
    "round": round,
    "pow": pow,
    "isinstance": isinstance,
}


def run(args: dict) -> str:
    """在受限环境中执行代码并返回 stdout。

    TODO(M1-b):
      1. 取 args["code"]；缺失则返回错误字符串
      2. 准备 stdout 捕获：io.StringIO + contextlib.redirect_stdout
      3. 构造受限 globals，例如 {"__builtins__": _SAFE_BUILTINS}
      4. 用超时机制执行（signal.alarm / threading+timeout 等）
      5. 成功：返回捕获的 stdout；若为空可返回 "(no output)"
      6. 失败：返回 f"Error: ..."
    """
    # ---- 你的代码开始 ----
    raise NotImplementedError("TODO(M1-b): 实现 python_sandbox.run")
    # ---- 你的代码结束 ----

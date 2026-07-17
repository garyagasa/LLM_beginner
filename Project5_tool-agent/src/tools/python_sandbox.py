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

import contextlib
import io
import signal
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
    "True": True,
    "False": False,
    "None": None,
}

_FORBIDDEN_SNIPPETS = (
    "import os",
    "import subprocess",
    "import socket",
    "import sys",
    "__import__",
)

_EXEC_TIMEOUT_SEC = 2


def _timeout_handler(_signum, _frame) -> None:
    raise TimeoutError("execution timed out")


def run(args: dict) -> str:
    """在受限环境中执行代码并返回 stdout。"""
    try:
        code = args.get("code")
        if not code or not isinstance(code, str):
            return "Error: missing 'code'"

        lowered = code.lower()
        for bad in _FORBIDDEN_SNIPPETS:
            if bad in lowered:
                return f"Error: forbidden: {bad}"

        buf = io.StringIO()
        globals_dict = {"__builtins__": _SAFE_BUILTINS}
        locals_dict: dict[str, Any] = {}

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_EXEC_TIMEOUT_SEC)
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, globals_dict, locals_dict)  # noqa: S102
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        out = buf.getvalue().strip()
        return out if out else "(no output)"

    except TimeoutError:
        return "Error: execution timed out"
    except Exception as e:
        return f"Error: {e}"

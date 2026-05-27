"""任务五：Python 沙箱工具（受限 exec，限制 import、超时、stdout 捕获）。"""

TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": "在受限 Python 沙箱中执行代码，捕获 stdout。限制 import 和超时。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码",
                },
            },
            "required": ["code"],
        },
    },
}

import io
import sys
import signal
from contextlib import contextmanager


@contextmanager
def timeout_context(seconds: int):
    """超时上下文管理器。

    Args:
        seconds: 超时秒数
    """
    # TODO: 用 signal.alarm 或 threading 实现超时
    raise NotImplementedError("TODO: 实现 timeout_context")


def run(args: dict) -> str:
    """在受限沙箱中执行 Python 代码。

    Args:
        args: {"code": str}  Python 代码

    Returns:
        str: stdout 输出或错误信息
    """
    code = args.get("code", "")
    # TODO: 1. 检查代码安全性（黑名单 import、builtins 限制等）
    # TODO: 2. 重定向 stdout 到 StringIO
    # TODO: 3. 在超时限制下 exec 代码
    # TODO: 4. 返回捕获的 stdout 或异常信息
    raise NotImplementedError("TODO: 实现 python_sandbox.run")

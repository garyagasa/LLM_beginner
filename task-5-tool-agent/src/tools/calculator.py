"""任务五：计算器工具（四则运算 + 高级函数）。"""

TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行数学表达式计算，支持四则运算及高级函数（sin/cos/log 等）",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(16) + sin(pi/2)'",
                },
            },
            "required": ["expression"],
        },
    },
}


def run(args: dict) -> str:
    """执行数学表达式计算。

    Args:
        args: {"expression": str}  数学表达式

    Returns:
        str: 计算结果或错误信息
    """
    expression = args.get("expression", "")
    # TODO: 1. 安全地评估表达式（用 ast.literal_eval 或受限 eval）
    # TODO: 2. 支持 math 模块函数（sin, cos, log, sqrt, pi, e 等）
    # TODO: 3. 捕获异常并返回错误信息
    # TODO: 4. 返回字符串结果
    raise NotImplementedError("TODO: 实现 calculator.run")

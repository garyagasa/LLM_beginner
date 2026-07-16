"""工具包：毛利侦探事务所情报工具箱。

每个工具模块必须导出 TOOL_SCHEMA + run(args) -> str。

自检：`from src.tools import calculator, python_sandbox, file_search, wiki, conan_wiki`
"""

from . import calculator, conan_wiki, file_search, python_sandbox, wiki

# 工具名 → 模块，供 ReActAgent 路由使用
TOOL_MODULES = {
    "calculator": calculator,
    "python_sandbox": python_sandbox,
    "file_search": file_search,
    "wiki": wiki,
    "conan_wiki": conan_wiki,
}


def get_all_schemas() -> list[dict]:
    """返回全部工具的 OpenAI function calling schema 列表。"""
    return [mod.TOOL_SCHEMA for mod in TOOL_MODULES.values()]


def call_tool(name: str, args: dict) -> str:
    """按名称调用工具；未知工具返回错误字符串（不要抛异常打断循环）。"""
    mod = TOOL_MODULES.get(name)
    if mod is None:
        return f"Error: unknown tool '{name}'. Available: {list(TOOL_MODULES)}"
    try:
        return str(mod.run(args if isinstance(args, dict) else {}))
    except Exception as e:  # noqa: BLE001 — 故意吞掉，交给 Observation
        return f"Error calling {name}: {e}"

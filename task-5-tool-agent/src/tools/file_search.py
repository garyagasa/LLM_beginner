"""任务五：文件搜索工具（本地目录文件名/内容检索）。"""

import os

TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "在本地目录中搜索文件名或文件内容",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词（匹配文件名和内容）",
                },
                "directory": {
                    "type": "string",
                    "description": "搜索的根目录，默认为当前目录",
                },
                "search_content": {
                    "type": "boolean",
                    "description": "是否搜索文件内容（默认只搜索文件名）",
                },
            },
            "required": ["query"],
        },
    },
}


def run(args: dict) -> str:
    """在本地目录搜索文件。

    Args:
        args: {"query": str, "directory"?: str, "search_content"?: bool}

    Returns:
        str: 匹配的文件列表（每行一个路径，附匹配片段）
    """
    query = args.get("query", "")
    directory = args.get("directory", ".")
    search_content = args.get("search_content", False)

    # TODO: 1. 递归遍历目录
    # TODO: 2. 过滤 .gitignore 中的路径
    # TODO: 3. 匹配文件名（大小写不敏感）
    # TODO: 4. 如 search_content=True，用 grep 搜索文件内容
    # TODO: 5. 返回格式化的匹配结果
    raise NotImplementedError("TODO: 实现 file_search.run")

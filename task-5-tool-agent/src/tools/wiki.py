"""任务五：维基百科工具（调用 Wikipedia API）。"""

import wikipedia

TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "wikipedia",
        "description": "查询维基百科获取词条摘要或搜索结果",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要查询的关键词",
                },
                "lang": {
                    "type": "string",
                    "description": "语言代码，默认 zh（中文）",
                },
                "action": {
                    "type": "string",
                    "enum": ["summary", "search"],
                    "description": "summary = 返回词条摘要，search = 返回搜索列表",
                },
            },
            "required": ["query"],
        },
    },
}


def run(args: dict) -> str:
    """查询维基百科。

    Args:
        args: {"query": str, "lang"?: str, "action"?: str}

    Returns:
        str: 词条摘要/搜索结果或错误信息
    """
    query = args.get("query", "")
    lang = args.get("lang", "zh")
    action = args.get("action", "summary")

    # TODO: 1. 设置语言（wikipedia.set_lang(lang)）
    # TODO: 2. 若 action == "summary": 返回 wikipedia.summary(query)
    # TODO: 3. 若 action == "search": 返回 wikipedia.search(query) 结果
    # TODO: 4. 捕获 wikipedia.exceptions（PageError, DisambiguationError 等）
    # TODO: 5. 返回格式化文本
    raise NotImplementedError("TODO: 实现 wiki.run")

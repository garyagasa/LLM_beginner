"""M1-d：现实世界维基百科（对照柯南百科用）。

---- 接口约定（eval/run.py 自检依赖）----
  TOOL_SCHEMA: dict
  run(args: dict) -> str
    args["query"]: 检索词，如 "柯南·道尔" / "福尔摩斯"
    自检：网络可用时，返回文本长度 > 50；离线则跳过

柯南相关设定请优先用 conan_wiki；本工具查现实世界百科。
"""

from __future__ import annotations

TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "Query real-world Wikipedia (not the Conan fandom wiki). "
            "Use for authors like Arthur Conan Doyle, history, science. "
            "For Detective Conan characters / APTX4869 / Black Organization, "
            "use conan_wiki instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Article title, e.g. '柯南·道尔' or 'Sherlock Holmes'.",
                },
            },
            "required": ["query"],
        },
    },
}


def run(args: dict) -> str:
    """查询维基百科摘要。

    TODO(M1-d):
      1. query = args.get("query"); 缺失则返回错误字符串
      2. wikipedia-api 或 MediaWiki API；中文优先，可回退英文
      3. 找不到：返回 "No Wikipedia page found for: ..."
      4. 网络异常可 raise（自检会 skip）
    """
    # ---- 你的代码开始 ----
    raise NotImplementedError("TODO(M1-d): 实现 wiki.run")
    # ---- 你的代码结束 ----

"""M1-c：米花町档案柜检索（本地文件）。

---- 接口约定（eval/run.py 自检依赖）----
  TOOL_SCHEMA: dict
  run(args: dict) -> str
    args["pattern"]: 文件名 glob 或内容关键词
    args["dir"]:     搜索根目录（自检会传项目根或 data/agent-fixtures）
    自检 1：pattern="README.md", dir=<项目根> → 输出含 "README.md"

建议行为：
  - 同时支持「按文件名匹配」和「按文件内容包含关键词」
  - 路径越界保护：解析后的路径必须仍在 dir 之下
  - 限制扫描文件数量 / 单文件大小

主题用途：在 data/agent-fixtures 里翻 APTX / 红黑双方档案。
"""

from __future__ import annotations

TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": (
            "Search Beika Town case-file cabinet (local files) by filename "
            "pattern and/or content keyword. Returns matching paths and snippets. "
            "Look under data/agent-fixtures for APTX4869 and red/black dossiers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Filename glob (e.g. '*.md') or content keyword "
                        "(e.g. 'APTX4869', '黑衣组织', '赤井')."
                    ),
                },
                "dir": {
                    "type": "string",
                    "description": "Root directory to search. Defaults to cwd.",
                },
            },
            "required": ["pattern"],
        },
    },
}


def run(args: dict) -> str:
    """搜索本地档案。

    TODO(M1-c):
      1. pattern = args.get("pattern"); root = Path(args.get("dir") or ".")
      2. root = root.resolve()；若不存在返回错误
      3. 遍历 root.rglob("*")（跳过隐藏目录 / __pycache__）
      4. 文件名匹配 + 内容关键词匹配，合并去重
      5. 路径越界检查；返回路径列表（可附 snippet）
      6. 无命中：返回 "No files matched."
    """
    # ---- 你的代码开始 ----
    raise NotImplementedError("TODO(M1-c): 实现 file_search.run")
    # ---- 你的代码结束 ----

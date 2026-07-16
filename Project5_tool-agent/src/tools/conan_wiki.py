"""M1-e：名侦探柯南中文 Wiki（Fandom）。

---- 接口约定（eval/run.py 自检依赖）----
  TOOL_SCHEMA: dict
  run(args: dict) -> str
    args["query"]: 词条名，如 "灰原哀" / "赤井秀一" / "APTX4869"
    自检：网络可用时返回文本长度 > 50；离线则 skip

站点：https://detectiveconan.fandom.com/zh/wiki/
API：https://detectiveconan.fandom.com/zh/api.php

重点覆盖主线红黑相关人物与 APTX4869（查询时可带空格或别名）。
"""

from __future__ import annotations

TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "conan_wiki",
        "description": (
            "Query the Chinese Detective Conan Fandom wiki "
            "(detectiveconan.fandom.com/zh) and return a short summary. "
            "Use for main-plot characters (工藤新一/江户川柯南, 灰原哀/宫野志保, "
            "赤井秀一, 安室透/降谷零, 琴酒, 贝尔摩德, APTX4869, 黑衣组织, etc.). "
            "Prefer this over general Wikipedia for Conan lore."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Wiki page title, e.g. '灰原哀', '赤井秀一', 'APTX4869', '琴酒'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

# Fandom MediaWiki API（中文站）
CONAN_API = "https://detectiveconan.fandom.com/zh/api.php"
USER_AGENT = "Project5-tool-agent/1.0 (Detective Conan themed homework)"


def run(args: dict) -> str:
    """查询柯南中文 Wiki 摘要。

    TODO(M1-e):
      1. query = args.get("query"); 缺失则返回错误字符串
      2. 用 requests.get(CONAN_API, params={...}, headers={"User-Agent": USER_AGENT}, timeout=15)
         推荐两步：
         a) action=query & list=search & srsearch=query  —— 处理别名/模糊名
         b) action=query & prop=extracts & exintro=1 & explaintext=1 & titles=<命中标题>
            或 action=parse & page=<title> & prop=wikitext / text
      3. 把 HTML/wikitext 收成纯文本，截断到约 1500 字返回
      4. 找不到：返回 "No Conan wiki page found for: ..."
      5. 网络异常：raise 或返回 Error 字符串
         （自检对 conan_wiki 的网络异常会标为 skip）

    提示：灰原哀页含「18岁（外表年龄7岁）」；APTX 词条名可能是 APTX4869 / APTX-4869。
    """
    # ---- 你的代码开始 ----
    raise NotImplementedError("TODO(M1-e): 实现 conan_wiki.run")
    # ---- 你的代码结束 ----

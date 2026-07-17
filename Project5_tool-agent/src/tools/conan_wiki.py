"""M1-e：名侦探柯南中文 Wiki（Fandom）。

---- 接口约定（eval/run.py 自检依赖）----
  TOOL_SCHEMA: dict
  run(args: dict) -> str
    args["query"]: 词条名，如 "灰原哀" / "赤井秀一" / "APTX4869"
    自检：网络可用时返回文本长度 > 50；离线则 skip

站点：https://detectiveconan.fandom.com/zh/wiki/
API：https://detectiveconan.fandom.com/zh/api.php
"""

from __future__ import annotations

import html
import re

import requests

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

CONAN_API = "https://detectiveconan.fandom.com/zh/api.php"
USER_AGENT = "Project5-tool-agent/1.0 (Detective Conan themed homework)"
MAX_LEN = 1500
TIMEOUT = 15


def _api(params: dict) -> dict:
    resp = requests.get(
        CONAN_API,
        params={**params, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _search_title(query: str) -> str | None:
    data = _api({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
    })
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    # 优先标题完全匹配
    for hit in hits:
        if hit.get("title") == query:
            return hit["title"]
    return hits[0]["title"]


def _fetch_extract(title: str) -> str | None:
    data = _api({
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    })
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None
        text = (page.get("extract") or "").strip()
        if text:
            return text
    return None


def _fetch_parse(title: str) -> str | None:
    """Fandom 部分页面 extracts 为空，回退 parse 取正文前段。"""
    data = _api({
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": 1,
    })
    parse = data.get("parse")
    if not parse:
        return None
    html_content = parse.get("text", {}).get("*", "")
    text = _html_to_text(html_content)
    return text or None


def _fetch_page_text(title: str) -> str | None:
    text = _fetch_extract(title)
    if text:
        return text
    return _fetch_parse(title)


def _query_variants(query: str) -> list[str]:
    q = query.strip()
    variants = [q]
    compact = re.sub(r"[\s\-_]", "", q)
    if compact and compact not in variants:
        variants.append(compact)
    upper = compact.upper()
    if upper.startswith("APTX"):
        for alt in ("APTX-4869", "APTX4869"):
            if alt not in variants:
                variants.append(alt)
    return variants


def run(args: dict) -> str:
    """查询柯南中文 Wiki 摘要。"""
    query = args.get("query")
    if not query or not isinstance(query, str):
        return "Error: missing 'query'"

    query = query.strip()
    tried: list[str] = []

    for candidate in _query_variants(query):
        if candidate in tried:
            continue
        tried.append(candidate)
        text = _fetch_page_text(candidate)
        if text:
            if candidate != query:
                header = f"[Conan Wiki: {candidate}]\n"
                return (header + text)[:MAX_LEN]
            return text[:MAX_LEN]

    title = _search_title(query)
    if title and title not in tried:
        text = _fetch_page_text(title)
        if text:
            header = f"[Conan Wiki: {title}]\n"
            return (header + text)[:MAX_LEN]

    return f"No Conan wiki page found for: {query}"

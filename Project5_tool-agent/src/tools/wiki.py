"""M1-d：现实世界维基百科（对照柯南百科用）。

---- 接口约定（eval/run.py 自检依赖）----
  TOOL_SCHEMA: dict
  run(args: dict) -> str
    args["query"]: 检索词，如 "柯南·道尔" / "福尔摩斯"
    自检：网络可用时，返回文本长度 > 50；离线则 skip

柯南相关设定请优先用 conan_wiki；本工具查现实世界百科。
"""

from __future__ import annotations

import re

import requests

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

USER_AGENT = "Project5-tool-agent/1.0 (Detective Conan themed homework)"
MAX_LEN = 1500
TIMEOUT = 15

_WIKI_APIS = {
    "zh": "https://zh.wikipedia.org/w/api.php",
    "en": "https://en.wikipedia.org/w/api.php",
}

_EN_FALLBACK_TITLES = {
    "柯南·道尔": "Arthur Conan Doyle",
    "阿瑟·柯南·道尔": "Arthur Conan Doyle",
    "柯南道尔": "Arthur Conan Doyle",
    "福尔摩斯": "Sherlock Holmes",
    "夏洛克·福尔摩斯": "Sherlock Holmes",
}


def _api(language: str, params: dict) -> dict:
    resp = requests.get(
        _WIKI_APIS[language],
        params={**params, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _search_title(language: str, query: str) -> str | None:
    data = _api(language, {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 3,
    })
    hits = data.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def _fetch_extract(language: str, title: str) -> str | None:
    data = _api(language, {
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


def _lookup(language: str, query: str) -> str | None:
    for title in _candidate_titles(query):
        text = _fetch_extract(language, title)
        if text:
            if title != query.strip():
                prefix = f"[{language} Wikipedia: {title}]\n"
                return (prefix + text)[:MAX_LEN]
            return text[:MAX_LEN]

    title = _search_title(language, query)
    if title:
        text = _fetch_extract(language, title)
        if text:
            prefix = f"[{language} Wikipedia: {title}]\n"
            return (prefix + text)[:MAX_LEN]
    return None


def _candidate_titles(query: str) -> list[str]:
    q = query.strip()
    titles = [q]
    if q in _EN_FALLBACK_TITLES:
        titles.append(_EN_FALLBACK_TITLES[q])
    alt = re.sub(r"[·・]", "", q)
    if alt and alt not in titles:
        titles.append(alt)
    return titles


def run(args: dict) -> str:
    """查询维基百科摘要（中文优先，必要时回退英文）。"""
    query = args.get("query")
    if not query or not isinstance(query, str):
        return "Error: missing 'query'"

    query = query.strip()
    zh_err: Exception | None = None
    try:
        text = _lookup("zh", query)
        if text:
            return text
    except requests.RequestException as e:
        zh_err = e

    try:
        text = _lookup("en", query)
        if text:
            return text
    except requests.RequestException as e:
        if zh_err is not None:
            return f"Error: zh/en wikipedia request failed ({zh_err}; {e})"
        return f"Error: wikipedia request failed ({e})"

    if zh_err is not None:
        return f"Error: zh wikipedia unreachable, and no english page found for: {query}"
    return f"No Wikipedia page found for: {query}"

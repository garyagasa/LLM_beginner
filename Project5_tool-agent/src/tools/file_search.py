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
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

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

MAX_FILES = 5000
MAX_FILE_BYTES = 512 * 1024
MAX_SNIPPET_CHARS = 120

_SKIP_DIR_NAMES = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _should_skip_dir(name: str) -> bool:
    return name.startswith(".") or name in _SKIP_DIR_NAMES


def _looks_like_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[]")


def _name_matches(name: str, pattern: str) -> bool:
    if _looks_like_glob(pattern):
        return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name.lower(), pattern.lower())
    return pattern.lower() in name.lower()


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _content_matches(text: str, pattern: str) -> bool:
    return pattern in text or pattern.lower() in text.lower()


def _make_snippet(text: str, pattern: str) -> str:
    lower_text = text.lower()
    idx = lower_text.find(pattern.lower())
    if idx < 0:
        snippet = text[:MAX_SNIPPET_CHARS]
    else:
        start = max(0, idx - 40)
        end = min(len(text), idx + len(pattern) + 80)
        snippet = text[start:end]
    snippet = " ".join(snippet.split())
    if len(snippet) > MAX_SNIPPET_CHARS:
        snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
    return snippet


def _is_filename_pattern(pattern: str) -> bool:
    """Treat dotted or glob patterns as filename queries, not content keywords."""
    return _looks_like_glob(pattern) or "." in pattern


def run(args: dict) -> str:
    """搜索本地档案。"""
    pattern = args.get("pattern")
    if not pattern or not isinstance(pattern, str):
        return "Error: missing 'pattern'"

    pattern = pattern.strip()
    root = Path(args.get("dir") or ".").resolve()
    if not root.exists():
        return f"Error: directory not found: {root}"
    if not root.is_dir():
        return f"Error: not a directory: {root}"

    matches: dict[str, dict[str, str | None]] = {}
    scanned = 0

    for path in root.rglob("*"):
        if scanned >= MAX_FILES:
            break
        if not path.is_file():
            continue

        if any(_should_skip_dir(part) for part in path.relative_to(root).parts):
            continue
        if not _is_under_root(path, root):
            continue

        scanned += 1
        rel = str(path.relative_to(root))
        if rel.startswith(".."):
            continue

        reasons: list[str] = []
        snippet: str | None = None

        if _name_matches(path.name, pattern):
            reasons.append("filename")

        if not _is_filename_pattern(pattern):
            text = _read_text(path)
            if text is not None and _content_matches(text, pattern):
                reasons.append("content")
                snippet = _make_snippet(text, pattern)
        elif reasons:
            text = _read_text(path)
            if text is not None:
                snippet = _make_snippet(text, pattern)

        if reasons:
            key = str(path.resolve())
            prev = matches.get(key)
            merged_reasons = set(prev["reasons"].split(",")) if prev else set()
            merged_reasons.update(reasons)
            matches[key] = {
                "rel": rel,
                "reasons": ",".join(sorted(merged_reasons)),
                "snippet": snippet or (prev.get("snippet") if prev else None),
            }

    if not matches:
        return "No files matched."

    lines: list[str] = []
    for item in sorted(matches.values(), key=lambda x: x["rel"]):
        line = item["rel"]
        if item.get("snippet"):
            line += f" | snippet: {item['snippet']}"
        lines.append(line)

    return "\n".join(lines)

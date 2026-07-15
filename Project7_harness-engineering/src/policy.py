"""Harness 的确定性策略层：所有 I/O 在执行前先经过这里。"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable


class PolicyEngine:
    def __init__(self, root: str | Path, config: dict):
        self.root = Path(root).resolve()
        self.config = dict(config)
        self.allowed_commands = {
            self._command_name(command)
            for command in self.config.get("allowed_commands", [])
        }

    @staticmethod
    def _command_name(command: str) -> str:
        return Path(str(command)).stem.lower()

    @staticmethod
    def _matches(relative: Path, patterns: Iterable[str]) -> bool:
        value = relative.as_posix()
        for pattern in patterns:
            # fnmatch("a.py", "**/*.py") 在不同 Python 版本上行为不完全直观，
            # 因此同时检查去掉 **/ 后的根目录文件形式。
            if fnmatch.fnmatch(value, pattern):
                return True
            if pattern.startswith("**/") and fnmatch.fnmatch(value, pattern[3:]):
                return True
        return False

    def check_path(self, path: str | Path, operation: str = "read") -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"路径逃逸工作区：{path}") from exc

        if operation == "read":
            patterns = self.config.get("readable_globs", ["**/*"])
        elif operation == "write":
            patterns = self.config.get("writable_globs", [])
        else:
            raise ValueError(f"未知路径操作：{operation}")

        if not self._matches(relative, patterns):
            raise PermissionError(f"策略不允许 {operation}：{relative.as_posix()}")
        return resolved

    def check_command(self, argv: list[str] | tuple[str, ...]) -> list[str]:
        if not isinstance(argv, (list, tuple)) or not argv:
            raise PermissionError("命令必须是非空 argv 列表")
        normalized = [str(item) for item in argv]
        command = self._command_name(normalized[0])
        if command not in self.allowed_commands:
            raise PermissionError(f"策略不允许命令：{command}")
        return normalized


"""受策略约束的文件与命令运行时。"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .policy import PolicyEngine


class SandboxRuntime:
    def __init__(self, root: str | Path, policy: PolicyEngine):
        self.root = Path(root).resolve()
        self.policy = policy
        self.timeout = int(policy.config.get("command_timeout_seconds", 30))
        self.max_output_chars = int(policy.config.get("max_output_chars", 12000))

    def read_file(self, path: str | Path) -> str:
        safe_path = self.policy.check_path(path, "read")
        return safe_path.read_text(encoding="utf-8")

    def write_file(self, path: str | Path, content: str) -> None:
        safe_path = self.policy.check_path(path, "write")
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8", newline="\n")

    @staticmethod
    def _safe_environment() -> dict[str, str]:
        # 不把整个父进程环境交给 Agent 命令，避免 API Key 等秘密意外泄露。
        keep = ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME")
        env = {key: os.environ[key] for key in keep if key in os.environ}
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def run_command(self, argv: list[str], timeout: int | None = None) -> dict:
        safe_argv = self.policy.check_command(argv)
        effective_timeout = self.timeout if timeout is None else min(int(timeout), self.timeout)
        try:
            completed = subprocess.run(
                safe_argv,
                cwd=self.root,
                shell=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=effective_timeout,
                env=self._safe_environment(),
            )
            return {
                "exit_code": completed.returncode,
                "stdout": completed.stdout[-self.max_output_chars :],
                "stderr": completed.stderr[-self.max_output_chars :],
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return {
                "exit_code": 124,
                "stdout": stdout[-self.max_output_chars :],
                "stderr": stderr[-self.max_output_chars :],
                "timed_out": True,
            }


"""最小 Harness：隔离、执行、验证、反馈、重试、留痕和停止。"""
from __future__ import annotations

import shutil
from pathlib import Path

from .policy import PolicyEngine
from .runtime import SandboxRuntime
from .state import RunStore


class AgentHarness:
    def __init__(self, config: dict, runs_dir: str | Path, verifier_command: list[str]):
        self.config = dict(config)
        self.store = RunStore(runs_dir)
        self.verifier_command = [str(part) for part in verifier_command]

    @staticmethod
    def _copy_workspace(source: Path, destination: Path) -> None:
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
        )

    @staticmethod
    def _feedback(attempt: int, verification: dict) -> dict:
        if verification.get("timed_out"):
            failure_type = "verifier_timeout"
        elif verification.get("exit_code") != 0:
            failure_type = "verification_failed"
        else:
            failure_type = "none"
        return {
            "attempt": attempt,
            "failure_type": failure_type,
            "exit_code": verification.get("exit_code"),
            "stdout": verification.get("stdout", ""),
            "stderr": verification.get("stderr", ""),
        }

    def run(self, agent, task: str, workspace: str | Path) -> dict:
        source = Path(workspace).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"工作区不存在：{source}")

        run_id = self.store.create(task, self.config)
        run_dir = self.store.run_dir(run_id)
        isolated = run_dir / "workspace"
        self._copy_workspace(source, isolated)

        policy = PolicyEngine(isolated, self.config)
        runtime = SandboxRuntime(isolated, policy)
        max_attempts = max(1, int(self.config.get("max_attempts", 1)))
        feedback = None
        final_verification = {
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
        }
        attempts = 0
        success = False

        self.store.append(
            run_id,
            {"type": "run_start", "source_workspace": str(source),
             "isolated_workspace": str(isolated)},
        )

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            self.store.append(run_id, {"type": "attempt_start", "attempt": attempt})
            try:
                action = agent.act(task=task, runtime=runtime, feedback=feedback)
                if not isinstance(action, dict):
                    action = {"summary": str(action)}
                self.store.append(
                    run_id,
                    {"type": "agent_action", "attempt": attempt, "action": action},
                )
            except Exception as exc:  # 异常也必须变成可观察的失败证据。
                self.store.append(
                    run_id,
                    {"type": "agent_error", "attempt": attempt,
                     "error_type": type(exc).__name__, "error": str(exc)},
                )

            final_verification = runtime.run_command(self.verifier_command)
            self.store.append(
                run_id,
                {"type": "verification", "attempt": attempt, **final_verification},
            )
            success = final_verification.get("exit_code") == 0
            if success:
                break
            feedback = self._feedback(attempt, final_verification)

        self.store.append(
            run_id,
            {"type": "run_finish", "success": success, "attempts": attempts,
             "stop_reason": "verified" if success else "attempt_budget_exhausted"},
        )
        return {
            "run_id": run_id,
            "success": success,
            "attempts": attempts,
            "workspace": str(isolated),
            "trace_path": str(self.store.trace_path(run_id)),
            "final_verification": final_verification,
        }


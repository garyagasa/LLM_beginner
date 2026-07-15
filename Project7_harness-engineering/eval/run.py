"""任务七自检：策略、持久状态、隔离、反馈重试、trace 与预算。"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from _eval_harness import run_tests


def _config(**overrides):
    config = json.loads((ROOT / "config" / "harness.json").read_text(encoding="utf-8"))
    config.update(overrides)
    return config


def _fixture_copy(parent: Path) -> Path:
    source = ROOT / "data" / "toy-repo"
    if not (source / "calculator.py.orig").exists():
        raise FileNotFoundError("请先运行 python data/download.py")
    target = parent / "source-repo"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    shutil.copy2(target / "calculator.py.orig", target / "calculator.py")
    return target


def test_policy_guardrails():
    try:
        from src.policy import PolicyEngine
    except ImportError as e:
        return {"test": "policy_guardrails", "pass": False, "error": str(e)}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "workspace"
        root.mkdir()
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        policy = PolicyEngine(root, _config())
        allowed_path = policy.check_path("ok.py", "read")
        allowed_command = policy.check_command([sys.executable, "-m", "pytest", "-q"])

        blocked_path = blocked_command = False
        try:
            policy.check_path("../escape.py", "write")
        except PermissionError:
            blocked_path = True
        try:
            policy.check_command(["git", "push"])
        except PermissionError:
            blocked_command = True

        return {
            "test": "policy_guardrails",
            "pass": bool(allowed_path) and bool(allowed_command) and blocked_path and blocked_command,
            "blocked_path_escape": blocked_path,
            "blocked_forbidden_command": blocked_command,
        }


def test_run_store_persistence():
    try:
        from src.state import RunStore
    except ImportError as e:
        return {"test": "run_store_persistence", "pass": False, "error": str(e)}

    with tempfile.TemporaryDirectory() as td:
        runs = Path(td) / "runs"
        store = RunStore(runs)
        run_id = store.create("demo task", _config())
        store.append(run_id, {"type": "attempt_start", "attempt": 1})
        store.append(run_id, {"type": "verification", "exit_code": 1})

        reloaded = RunStore(runs).load(run_id)
        events = reloaded.get("events", []) if isinstance(reloaded, dict) else []
        kinds = {event.get("type") for event in events if isinstance(event, dict)}
        return {
            "test": "run_store_persistence",
            "pass": {"attempt_start", "verification"}.issubset(kinds),
            "run_id": run_id,
            "event_count": len(events),
        }


class FixOnFeedbackAgent:
    """第一次保持 Bug；拿到验证反馈后才通过 runtime 修复。"""

    def __init__(self):
        self.calls = []

    def act(self, task, runtime, feedback=None):
        self.calls.append(feedback)
        if feedback:
            runtime.write_file(
                "calculator.py",
                "def add(a, b):\n"
                '    """Return a + b."""\n'
                "    return a + b\n\n\n"
                "def multiply(a, b):\n"
                "    return a * b\n",
            )
            return {"summary": "fixed add after reading verifier feedback"}
        return {"summary": "incorrectly assumed the code was already correct"}


class NeverFixAgent:
    def act(self, task, runtime, feedback=None):
        return {"summary": "no change"}


def _run_harness(agent, max_attempts=3):
    from src.harness import AgentHarness

    temp = tempfile.TemporaryDirectory()
    base = Path(temp.name)
    source = _fixture_copy(base)
    original = (source / "calculator.py").read_text(encoding="utf-8")
    harness = AgentHarness(
        config=_config(max_attempts=max_attempts),
        runs_dir=base / "runs",
        verifier_command=[sys.executable, "-m", "pytest", "-q"],
    )
    result = harness.run(
        agent=agent,
        task="修复 calculator.add，使 pytest 全部通过。",
        workspace=source,
    )
    return temp, source, original, result


def test_feedback_retry_loop():
    try:
        agent = FixOnFeedbackAgent()
        temp, source, original, result = _run_harness(agent)
    except (ImportError, FileNotFoundError, TypeError, AttributeError) as e:
        return {"test": "feedback_retry_loop", "pass": False, "error": str(e)}
    try:
        verification = result.get("final_verification", {})
        second_feedback = agent.calls[1] if len(agent.calls) > 1 else None
        return {
            "test": "feedback_retry_loop",
            "pass": bool(result.get("success"))
                    and result.get("attempts") == 2
                    and isinstance(second_feedback, dict)
                    and verification.get("exit_code") == 0,
            "attempts": result.get("attempts"),
            "second_attempt_received_feedback": isinstance(second_feedback, dict),
            "exit_code": verification.get("exit_code"),
        }
    finally:
        temp.cleanup()


def test_workspace_isolation():
    try:
        temp, source, original, result = _run_harness(FixOnFeedbackAgent())
    except (ImportError, FileNotFoundError, TypeError, AttributeError) as e:
        return {"test": "workspace_isolation", "pass": False, "error": str(e)}
    try:
        isolated = Path(result.get("workspace", ""))
        source_unchanged = (source / "calculator.py").read_text(encoding="utf-8") == original
        isolated_changed = isolated.exists() and (
            isolated / "calculator.py"
        ).read_text(encoding="utf-8") != original
        return {
            "test": "workspace_isolation",
            "pass": source_unchanged and isolated_changed and isolated.resolve() != source.resolve(),
            "source_unchanged": source_unchanged,
            "isolated_changed": isolated_changed,
        }
    finally:
        temp.cleanup()


def test_trace_observability():
    try:
        temp, source, original, result = _run_harness(FixOnFeedbackAgent())
    except (ImportError, FileNotFoundError, TypeError, AttributeError) as e:
        return {"test": "trace_observability", "pass": False, "error": str(e)}
    try:
        trace_value = result.get("trace_path")
        trace_path = Path(trace_value) if trace_value else None
        if not trace_path or not trace_path.exists():
            return {"test": "trace_observability", "pass": False,
                    "error": "run() 返回值缺少有效 trace_path"}
        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        kinds = {event.get("type") for event in events}
        required = {"run_start", "attempt_start", "verification", "run_finish"}
        return {
            "test": "trace_observability",
            "pass": required.issubset(kinds),
            "event_types": sorted(str(kind) for kind in kinds),
        }
    finally:
        temp.cleanup()


def test_budget_stop():
    try:
        temp, source, original, result = _run_harness(NeverFixAgent(), max_attempts=1)
    except (ImportError, FileNotFoundError, TypeError, AttributeError) as e:
        return {"test": "budget_stop", "pass": False, "error": str(e)}
    try:
        return {
            "test": "budget_stop",
            "pass": result.get("success") is False and result.get("attempts") == 1,
            "success": result.get("success"),
            "attempts": result.get("attempts"),
        }
    finally:
        temp.cleanup()


if __name__ == "__main__":
    run_tests(
        [
            test_policy_guardrails,
            test_run_store_persistence,
            test_feedback_retry_loop,
            test_workspace_isolation,
            test_trace_observability,
            test_budget_stop,
        ],
        ROOT,
    )

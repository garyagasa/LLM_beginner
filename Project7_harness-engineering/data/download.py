"""生成任务七的离线 toy repo 与评测任务，不需要网络。"""
import json
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent
TOY_REPO = DATA_DIR / "toy-repo"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


BUGGY = """def add(a, b):
    \"\"\"Return a + b.\"\"\"
    return a - b


def multiply(a, b):
    return a * b
"""


TESTS = """from calculator import add, multiply


def test_add():
    assert add(2, 3) == 5
    assert add(-2, 3) == 1


def test_multiply_regression():
    assert multiply(4, 5) == 20
"""


def create_toy_repo() -> None:
    TOY_REPO.mkdir(parents=True, exist_ok=True)
    write_text(TOY_REPO / "calculator.py", BUGGY)
    write_text(TOY_REPO / "calculator.py.orig", BUGGY)
    write_text(TOY_REPO / "test_calculator.py", TESTS)
    write_text(
        TOY_REPO / "ISSUE.md",
        "修复 calculator.add，使测试全部通过；不要修改测试，也不要破坏 multiply。\n",
    )
    tasks = [
        {
            "id": "calculator-add",
            "workspace": "toy-repo",
            "issue": "修复 calculator.add，使 pytest 全部通过。",
            "verifier": ["python", "-m", "pytest", "-q"],
        }
    ]
    write_text(DATA_DIR / "tasks.json", json.dumps(tasks, ensure_ascii=False, indent=2))

    # git 只用于方便观察 diff；缺失 git 或身份配置不影响任务。
    if not (TOY_REPO / ".git").exists():
        quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        try:
            subprocess.run(["git", "init"], cwd=TOY_REPO, check=True, **quiet)
            subprocess.run(["git", "add", "."], cwd=TOY_REPO, check=True, **quiet)
            subprocess.run(
                ["git", "-c", "user.name=harness-tutor", "-c",
                 "user.email=tutor@nndl.ai", "commit", "-m", "initial fixture"],
                cwd=TOY_REPO,
                check=True,
                **quiet,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("[提示] git 初始化跳过，不影响自检。")

    print(f"已生成 {TOY_REPO.relative_to(DATA_DIR.parent)}")
    print("下一步：运行 python eval/run.py 观察 Harness 的失败反馈与重试闭环")


if __name__ == "__main__":
    create_toy_repo()

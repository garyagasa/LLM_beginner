"""M4（可选）：用 Qwen-Agent 实现功能相同的对照版（柯南主题五工具）。

安装（可选依赖）：
  pip install qwen-agent soundfile python-dateutil json5

目标：
  - 注册与手写版相同的 5 个工具（含 conan_wiki）
  - 跑 data/tasks.json，对比与 ReActAgent 的成功率

本文件不参与 eval/run.py 必检；写完后自行运行：
  python -m src.qwen_agent_baseline
  python -m src.qwen_agent_baseline --limit 3        # 只跑前 3 题
  python -m src.qwen_agent_baseline --skip-react     # 只跑 Qwen-Agent
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from . import config
from .tools import TOOL_MODULES

# ---------------------------------------------------------------------------
# 与 eval/run.py 一致的答案判定（避免重复跑时逻辑不一致）
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = text.replace(",", "").replace("，", "")
    return re.sub(r"\s+", "", text)


def answer_matches(answer: str, expected_keywords: list) -> bool:
    norm_answer = normalize_answer(answer)
    for expected in expected_keywords:
        if isinstance(expected, list):
            if not any(normalize_answer(keyword) in norm_answer for keyword in expected):
                return False
        elif normalize_answer(expected) not in norm_answer:
            return False
    return True


# ---------------------------------------------------------------------------
# 工具包装：把 src/tools 桥接到 Qwen-Agent BaseTool
# ---------------------------------------------------------------------------


def _require_qwen_agent():
    try:
        from qwen_agent.agents import FnCallAgent  # noqa: F401
        from qwen_agent.tools.base import BaseTool  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "缺少 Qwen-Agent 依赖。请安装：\n"
            "  pip install qwen-agent soundfile python-dateutil json5"
        ) from e


class ProjectTool:
    """工厂：为每个 src/tools 模块生成 Qwen-Agent BaseTool 实例。"""

    @staticmethod
    def create(tool_name: str):
        from qwen_agent.tools.base import BaseTool

        mod = TOOL_MODULES[tool_name]
        fn = mod.TOOL_SCHEMA["function"]

        class _WrappedTool(BaseTool):
            description = fn["description"]
            parameters = fn["parameters"]

            def __init__(self):
                self.name = tool_name
                self._mod = mod
                super().__init__()

            def call(self, params: str | dict, **kwargs) -> str:
                args = self._verify_json_format_args(params)
                return str(self._mod.run(args))

        return _WrappedTool()


def _qwen_system_message() -> str:
    """Qwen-Agent 原生 function calling，无需 ReAct 文本格式。"""
    tool_lines = []
    for name, mod in TOOL_MODULES.items():
        desc = mod.TOOL_SCHEMA["function"].get("description", "")
        tool_lines.append(f"- {name}: {desc}")
    tools = "\n".join(tool_lines)
    return (
        "你是毛利侦探事务所的情报助手，协助江户川柯南调查黑衣组织与 APTX4869 相关线索。\n"
        "工具分工：conan_wiki 查柯南设定；wiki 查现实百科；"
        "file_search 查本地 data/agent-fixtures/ 档案；"
        "calculator 做数学；python_sandbox 跑受限 Python。\n"
        f"可用工具：\n{tools}\n"
        "请按需调用工具，用中文简洁作答，答案须包含题目要求的关键数字或名称。"
    )


def build_qwen_agent():
    """构造 Qwen-Agent FnCallAgent（function calling，对接 vLLM）。"""
    _require_qwen_agent()
    from qwen_agent.agents import FnCallAgent

    function_list = [ProjectTool.create(name) for name in TOOL_MODULES]
    llm_cfg = {
        "model": config.MODEL,
        "model_server": config.BASE_URL,
        "api_key": config.API_KEY,
        "generate_cfg": {
            "temperature": config.TEMPERATURE,
        },
    }
    return FnCallAgent(
        llm=llm_cfg,
        function_list=function_list,
        system_message=_qwen_system_message(),
        name="柯南情报助手(Qwen-Agent)",
    )


def _extract_final_text(messages: list) -> str:
    """从 Qwen-Agent 返回的消息列表中提取最后一条 assistant 文本。"""
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "assistant":
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(getattr(item, "text", item)))
            return "".join(parts).strip()
    return ""


def run_qwen_task(agent, task: str) -> str:
    """用 Qwen-Agent 跑单题，返回最终 assistant 文本。"""
    messages = [{"role": "user", "content": task}]
    last_response: list = []
    for response in agent.run(messages=messages):
        if response:
            last_response = response
    return _extract_final_text(last_response)


def _evaluate_tasks(
    label: str,
    runner,
    tasks: list[dict],
    limit: int | None = None,
) -> dict[str, Any]:
    """对任务列表跑评测，返回成功率与明细。"""
    subset = tasks[:limit] if limit else tasks
    success = 0
    details: list[dict[str, Any]] = []

    for t in subset:
        tid = t["id"]
        expected = t.get("expected_answer_contains", [])
        try:
            answer = runner(t["task"])
            ok = answer_matches(answer, expected)
            success += int(ok)
            details.append({
                "id": tid,
                "success": ok,
                "final_answer_preview": str(answer)[:120],
                "expected_answer_contains": expected,
            })
        except Exception as e:
            details.append({"id": tid, "success": False, "error": str(e)})

    rate = success / len(subset) if subset else 0.0
    return {
        "label": label,
        "rate": round(rate, 3),
        "success": success,
        "n": len(subset),
        "details": details,
    }


def run_tasks(
    *,
    limit: int | None = None,
    skip_react: bool = False,
    skip_qwen: bool = False,
) -> dict[str, Any]:
    """在 tasks.json 上跑对照版并打印成功率。"""
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        print("请先运行: python data/download.py")
        return {}

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}

    if not skip_qwen:
        print("=== Qwen-Agent 对照 ===")
        agent = build_qwen_agent()
        qwen_result = _evaluate_tasks(
            "Qwen-Agent",
            lambda task: run_qwen_task(agent, task),
            tasks,
            limit=limit,
        )
        results["qwen_agent"] = qwen_result
        print(f"成功率: {qwen_result['success']}/{qwen_result['n']} = {qwen_result['rate']:.1%}")

    if not skip_react:
        print("\n=== 手写 ReActAgent 对照 ===")
        from .agent import ReActAgent

        react_agent = ReActAgent()

        def _run_react(task: str) -> str:
            trace = react_agent.run(task)
            return str(trace.get("final_answer", ""))

        react_result = _evaluate_tasks(
            "ReActAgent",
            _run_react,
            tasks,
            limit=limit,
        )
        results["react_agent"] = react_result
        print(f"成功率: {react_result['success']}/{react_result['n']} = {react_result['rate']:.1%}")

    if "qwen_agent" in results and "react_agent" in results:
        q = results["qwen_agent"]["rate"]
        r = results["react_agent"]["rate"]
        diff = q - r
        results["comparison"] = {
            "qwen_rate": q,
            "react_rate": r,
            "delta_qwen_minus_react": round(diff, 3),
        }
        print("\n=== 对比 ===")
        print(f"Qwen-Agent : {q:.1%}")
        print(f"ReActAgent : {r:.1%}")
        print(f"差值 (Qwen - ReAct): {diff:+.1%}")

    out_path = ROOT / "eval" / "qwen_baseline_result.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果写入 {out_path.relative_to(ROOT)}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="M4: Qwen-Agent vs 手写 ReAct 对照评测")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    parser.add_argument("--skip-react", action="store_true", help="只跑 Qwen-Agent")
    parser.add_argument("--skip-qwen", action="store_true", help="只跑 ReActAgent")
    args = parser.parse_args()
    run_tasks(limit=args.limit, skip_react=args.skip_react, skip_qwen=args.skip_qwen)


if __name__ == "__main__":
    main()

"""M4（可选）：用 Qwen-Agent 实现功能相同的对照版（柯南主题五工具）。

安装（可选依赖）：
  pip install qwen-agent

目标：
  - 注册与手写版相同的 5 个工具（含 conan_wiki）
  - 跑 data/tasks.json，对比与 ReActAgent 的成功率

本文件不参与 eval/run.py 必检；写完后自行运行：
  python -m src.qwen_agent_baseline
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_qwen_agent():
    """构造 Qwen-Agent 助手。

    TODO(M4):
      1. pip install qwen-agent
      2. 查阅 https://github.com/QwenLM/Qwen-Agent 文档
      3. 把 src/tools 下 4 个工具包装成 Qwen-Agent 可注册的 tool
      4. 配置 llm：model_server 指向 Ollama / vLLM（与 src/config.py 一致）
      5. return agent 实例
    """
    # ---- 你的代码开始 ----
    raise NotImplementedError("TODO(M4): 用 Qwen-Agent 实现对照版")
    # ---- 你的代码结束 ----


def run_tasks() -> None:
    """在 tasks.json 上跑对照版并打印成功率。"""
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        print("请先运行: python data/download.py")
        return

    # TODO(M4): 加载 agent，逐题跑，打印答案与成功率
    raise NotImplementedError("TODO(M4): 实现对照评测 run_tasks")


if __name__ == "__main__":
    run_tasks()

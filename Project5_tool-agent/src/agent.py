"""M2：手写 ReAct Agent —— 毛利侦探事务所情报助手（名侦探柯南主题）。

人设建议（写进 system prompt）：
  你是协助江户川柯南 / 工藤新一的 AI 助手，擅长查 APTX4869、红黑双方人物与档案。
  柯南设定用 conan_wiki；现实百科用 wiki；本地档案用 file_search。

---- 接口约定（eval/run.py 自检依赖）----
  class ReActAgent:
      def run(self, task: str) -> AgentTrace

  AgentTrace: dict，至少包含
      steps: list[dict]       # 每步建议含 thought / action(=tool名) / action_input / observation
      final_answer: str
      success: bool           # 自检不依赖此字段判分，但仍需给出

ReAct 一轮格式（实践书 / 论文常见写法）：
  Thought: ...
  Action: <tool_name>
  Action Input: <json or plain text>
  Observation: <tool result>          ← 由你的代码写入，不是模型生成
  ...（重复）
  Thought: ...
  Final Answer: ...

推荐实现顺序：
  1. build_system_prompt() — 拼工具列表 + 格式说明 + 1~2 条 few-shot（可用 APTX 年龄差示例）
  2. parse_model_output()  — 从模型文本抽出 Thought / Action / Action Input / Final Answer
  3. run() 循环            — 调 LLM → 解析 → 调工具 → 把 Observation 追加进历史
"""

from __future__ import annotations

from typing import Any

from . import config
from .llm import chat
from .tools import call_tool, get_all_schemas
# 解析 Action Input 时你可能会用到：import json / import re
# 工具列表：from .tools import TOOL_MODULES

AgentTrace = dict[str, Any]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """构造 system prompt：柯南人设 + 工具说明 + 输出格式 + few-shot。

    TODO(M2-a):
      1. 写明人设：柯南主题助手；conan_wiki vs wiki 的分工
      2. 从 get_all_schemas() 列出工具（含 conan_wiki）
      3. 明确规定输出格式 Thought / Action / Action Input / Final Answer
      4. few-shot 示例建议：
           「新一 17 岁外表 7 岁，年龄差？」→ calculator → Final Answer: 10
           或「灰原哀是谁？」→ conan_wiki
      5. 每次只输出一个 Action，或直接 Final Answer；不要自己编 Observation
    """
    # ---- 你的代码开始 ----
    raise NotImplementedError("TODO(M2-a): 实现 build_system_prompt")
    # ---- 你的代码结束 ----


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def parse_model_output(text: str) -> dict[str, Any]:
    """解析模型输出。

    返回字典建议字段：
      thought: str | None
      action: str | None          # 工具名
      action_input: dict | str | None
      final_answer: str | None

    TODO(M2-b):
      1. 用正则或按行扫描提取 Thought / Action / Action Input / Final Answer
      2. Action Input 尽量 json.loads；失败则保留原始字符串
      3. 若存在 Final Answer，优先视为终止
      4. 解析失败时返回空字段，由 run() 决定重试提示
    """
    # ---- 你的代码开始 ----
    raise NotImplementedError("TODO(M2-b): 实现 parse_model_output")
    # ---- 你的代码结束 ----


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """手写 ReAct 循环。"""

    def __init__(
        self,
        max_steps: int | None = None,
        model: str | None = None,
    ):
        self.max_steps = max_steps or config.MAX_STEPS
        self.model = model
        # 可选：错误注入钩子（M3）；默认关闭
        self._inject_error_once = False
        self._error_injected = False

    def enable_error_injection(self) -> None:
        """M3：下一次真实工具调用将被替换为错误 Observation（测恢复能力）。"""
        self._inject_error_once = True
        self._error_injected = False

    def run(self, task: str) -> AgentTrace:
        """执行 ReAct 循环，返回 trace。

        TODO(M2-c):
          1. messages = [
               {"role": "system", "content": build_system_prompt()},
               {"role": "user", "content": task},
             ]
          2. steps = []
          3. for step in range(self.max_steps):
               a. text = chat(messages, model=self.model)
               b. parsed = parse_model_output(text)
               c. 若 parsed["final_answer"]：
                    记录 step，return {steps, final_answer, success: True}
               d. 若缺少 action：
                    把「请按格式输出 Action 或 Final Answer」作为 user/system 追加，continue
               e. observation = call_tool(action, action_input)
                  # M3：若 enable_error_injection 且尚未注入，
                  #     则 observation = "Error: simulated tool failure" 并标记已注入
               f. steps.append({
                    "thought": ..., "action": ..., "action_input": ...,
                    "observation": ..., "tool": action,   # tool 字段供自检 extract_used_tools
                  })
               g. 把模型输出 + f"Observation: {observation}" 追加进 messages
          4. 步数耗尽：return {steps, final_answer: 最后想法或 "", success: False}
        """
        # ---- 你的代码开始 ----
        raise NotImplementedError("TODO(M2-c): 实现 ReActAgent.run")
        # ---- 你的代码结束 ----

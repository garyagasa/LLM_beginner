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

import json
import re
from typing import Any

from . import config
from .llm import chat
from .tools import call_tool, get_all_schemas

AgentTrace = dict[str, Any]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _format_tool_catalog() -> str:
    lines: list[str] = []
    for schema in get_all_schemas():
        fn = schema.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        param_bits = []
        for key, meta in params.items():
            pdesc = meta.get("description", "")
            param_bits.append(f"{key}: {pdesc}" if pdesc else key)
        param_text = ", ".join(param_bits) if param_bits else "无参数"
        lines.append(f"- {name}({param_text}) — {desc}")
    return "\n".join(lines)


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
    tools = _format_tool_catalog()
    system_prompt = f"""
      你是毛利侦探事务所的情报助手，协助江户川柯南 / 工藤新一调查黑衣组织与 APTX4869 相关线索。
      工具分工：
      - conan_wiki：名侦探柯南世界观（角色、组织、药物设定）
      - wiki：现实世界百科（如柯南·道尔、福尔摩斯）
      - file_search：检索本地档案柜 data/agent-fixtures/（APTX、红黑双方名单等）
      - calculator：数学演算（年龄差、4869 相关计算）
      - python_sandbox：受限 Python 代码执行
      可用工具：
      {tools}
      你必须严格按下面 ReAct 格式输出；每次只执行一个 Action，或直接给出 Final Answer。
      不要编造 Observation；Observation 由系统在你调用工具后填入。
      格式：
      Thought: <你的推理>
      Action: <工具名，必须是上面列表之一>
      Action Input: <JSON 对象，例如 {{"expression": "17 - 7"}}>
      或者结束时：
      Thought: <总结>
      Final Answer: <给用户的最终答案>
      规则：
      1. 需要查本地档案时，file_search 的 dir 常用 "data/agent-fixtures" 或项目根目录 "."
      2. 需要算数时用 calculator；需要写代码时用 python_sandbox
      3. 先查资料再计算；答案要包含题目要求的关键数字或名称
      4. Action Input 必须是合法 JSON
      示例 1：
      用户：新一真实 17 岁、外表 7 岁，年龄差是多少？
      Thought: 这是简单减法，用计算器。
      Action: calculator
      Action Input: {{"expression": "17 - 7"}}
      示例 2：
      用户：灰原哀是谁？
      Thought: 这是柯南世界观人物，查 conan_wiki。
      Action: conan_wiki
      Action Input: {{"query": "灰原哀"}}
    """
    return system_prompt
    # ---- 你的代码结束 ----


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def _parse_action_input(raw: str | None) -> dict | str | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return text


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
    result: dict[str, Any] = {
        "thought": None,
        "action": None,
        "action_input": None,
        "final_answer": None,
    }
    if not text or not text.strip():
        return result
    # Final Answer 优先
    m_final = re.search(
        r"Final Answer:\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_final:
        result["final_answer"] = m_final.group(1).strip()
        return result
    m_thought = re.search(
        r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_thought:
        result["thought"] = m_thought.group(1).strip()
    m_action = re.search(r"Action:\s*(\w+)", text, flags=re.IGNORECASE)
    if m_action:
        result["action"] = m_action.group(1).strip()
    m_input = re.search(
        r"Action Input:\s*(.+?)(?=\n(?:Thought:|Action:|Observation:|Final Answer:)|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_input:
        result["action_input"] = _parse_action_input(m_input.group(1))
    return result
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

        # 步骤 1：初始化对话历史 —— system prompt 定义人设与格式，user 传入本次任务
        messages: list[dict[str, str]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": task},
        ]

        # 步骤 2：初始化 trace 记录；last_text 用于步数耗尽时兜底返回
        steps: list[dict[str, Any]] = []
        last_text = ""

        # 步骤 3：ReAct 主循环，最多执行 max_steps 轮
        for _ in range(self.max_steps):

            # 步骤 3a：调用 LLM，获取当前轮的 Thought / Action / Final Answer 文本
            text = chat(messages, model=self.model)
            last_text = text

            # 步骤 3b：从模型输出中解析结构化字段
            parsed = parse_model_output(text)

            # 步骤 3c：若模型给出 Final Answer，说明任务结束，记录并返回成功 trace
            if parsed.get("final_answer"):
                steps.append({
                    "thought": parsed.get("thought"),
                    "final_answer": parsed["final_answer"],
                })
                return {
                    "steps": steps,
                    "final_answer": parsed["final_answer"],
                    "success": True,
                }

            # 步骤 3d：若缺少 Action，说明格式不对；追加纠正提示后进入下一轮
            action = parsed.get("action")
            if not action:
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        "格式有误。请严格输出 Thought + Action + Action Input，"
                        "或直接输出 Final Answer。不要编造 Observation。"
                    ),
                })
                continue

            # 步骤 3e：规范化 Action Input（必须是 dict 才能传给 call_tool）
            action_input = parsed.get("action_input")
            if not isinstance(action_input, dict):
                action_input = {} if action_input is None else {"input": action_input}

            # 步骤 3e（M3）：错误注入 —— 测试 Agent 能否从模拟失败中恢复
            if self._inject_error_once and not self._error_injected:
                observation = "Error: simulated tool failure"
                self._error_injected = True
            else:
                observation = call_tool(action, action_input)

            # 步骤 3f：把本轮推理、工具调用与观察结果写入 steps（tool 字段供 eval 统计）
            step: dict[str, Any] = {
                "thought": parsed.get("thought"),
                "action": action,
                "action_input": action_input,
                "observation": observation,
                "tool": action,
            }
            steps.append(step)

            # 步骤 3g：将模型输出与 Observation 追加进 messages，供下一轮推理使用
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })

        # 步骤 4：步数耗尽仍未得到 Final Answer，返回失败 trace
        return {
            "steps": steps,
            "final_answer": last_text,
            "success": False,
        }
        # ---- 你的代码结束 ----

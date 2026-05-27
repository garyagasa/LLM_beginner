"""任务五：ReAct Agent 循环。

接口约定（eval/run.py 会自动检测）：
  - class ReActAgent 含 run(task: str) -> AgentTrace
  - AgentTrace 是 dict 含 steps, final_answer, success: bool
"""

from typing import List, Dict, Any
from openai import OpenAI


class AgentTrace:
    """Agent 执行轨迹。

    Attributes:
        steps: List[Dict]  每步的 thought, action, action_input, observation
        final_answer: str  最终答案
        success: bool  是否成功
    """

    def __init__(self):
        self.steps: List[Dict[str, str]] = []
        self.final_answer: str | None = None
        self.success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转为 dict（eval/run.py 要求的格式）。"""
        return {
            "steps": self.steps,
            "final_answer": self.final_answer,
            "success": self.success,
        }


class ReActAgent:
    """ReAct Agent：Thought → Action → Action Input → Observation 循环。"""

    def __init__(
        self,
        api_base: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b-instruct",
        max_steps: int = 10,
    ):
        """初始化。

        Args:
            api_base: OpenAI 兼容 API 地址
            model: 模型名称
            max_steps: 最大执行步数
        """
        # TODO: 1. 初始化 OpenAI 客户端
        # TODO: 2. 加载工具 schema 列表（从 src/tools/ 导入）
        # TODO: 3. 构建工具名 → run 函数的映射
        # TODO: 4. 构建 ReAct prompt 模板
        raise NotImplementedError("TODO: 实现 ReActAgent.__init__")

    def run(self, task: str) -> AgentTrace:
        """执行 ReAct 循环完成给定任务。

        Args:
            task: 任务描述

        Returns:
            AgentTrace: 执行轨迹
        """
        trace = AgentTrace()

        # TODO: 1. 将 task 填入 prompt 模板
        # TODO: 2. 主循环（直至 Final Answer 或步数上限）：
        #     a. 调 LLM（带工具 schema 作为 function calling）
        #     b. 解析模型输出（Thought / Action / Action Input）
        #     c. 若 Action == "Final Answer"：记录 final_answer，退出
        #     d. 否则：路由到对应工具，执行 run()
        #     e. 将 Observation 追加到 context
        #     f. 错误处理：捕获工具执行异常，将错误信息作为 Observation
        # TODO: 3. 判断 success（有 final_answer 且非空）
        # TODO: 4. 返回 trace

        raise NotImplementedError("TODO: 实现 ReActAgent.run")

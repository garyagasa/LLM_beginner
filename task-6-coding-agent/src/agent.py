"""任务六：Coding Agent 主循环。

接口约定（eval/run.py 会自动检测）：
  - class CodingAgent 含 run(repo_path: str, issue: str) -> Trace
  - Trace 含 steps, patch, tests_passed: bool

架构：
  agent loop → model → tool call / skill load / subagent dispatch → observation → loop
"""

from typing import List, Dict, Any


class Trace:
    """Coding Agent 执行轨迹。

    Attributes:
        steps: List[Dict]  每步的 thought, action, observation
        patch: str  git diff 格式的代码修改
        tests_passed: bool  最终测试是否通过
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.patch: str | None = None
        self.tests_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转为 dict（eval/run.py 要求的格式）。"""
        return {
            "steps": self.steps,
            "patch": self.patch,
            "tests_passed": self.tests_passed,
        }


class CodingAgent:
    """极简版 Claude Code：理解任务 → 修改代码 → 运行测试 → 迭代。"""

    def __init__(
        self,
        api_base: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b-instruct",
        max_steps: int = 20,
    ):
        """初始化。

        Args:
            api_base: OpenAI 兼容 API 地址
            model: 模型名称
            max_steps: 最大执行步数
        """
        # TODO: 1. 初始化 OpenAI 客户端
        # TODO: 2. 连接 MCP server（获取工具列表）
        # TODO: 3. 初始化 SkillLoader
        # TODO: 4. 加载 system prompt（含工具描述 + skill 摘要）
        raise NotImplementedError("TODO: 实现 CodingAgent.__init__")

    def run(self, repo_path: str, issue: str) -> Trace:
        """在给定仓库上执行编码任务。

        Args:
            repo_path: 目标仓库路径
            issue: 任务描述 / issue 文本

        Returns:
            Trace: 执行轨迹（含 steps, patch, tests_passed）
        """
        trace = Trace()

        # TODO: 1. 将 issue 作为任务目标填入 system prompt
        # TODO: 2. 主循环（最多 max_steps 步）：
        #     a. 调 LLM（带工具 schema，支持 function calling）
        #     b. 解析模型输出：
        #        - 如果是 tool call: 路由到 MCP tool 执行，记录 observation
        #        - 如果是 skill 引用: 用 SkillLoader.load() 加载并注入 context
        #        - 如果是 subagent 调用: 创建子 agent 处理（并行/串行）
        #     c. 检测终止条件：
        #        - 模型输出 "DONE" 或类似信号
        #        - 所有测试通过
        #        - 步数用尽
        # TODO: 3. 收尾：
        #     a. 运行 git diff 获取 patch
        #     b. 运行测试获取 tests_passed
        #     c. 填充 trace

        raise NotImplementedError("TODO: 实现 CodingAgent.run")

"""任务六：Subagents 模块。

每个 subagent 是一个独立的 agent 实例，处理特定子任务：
  - search_agent: 代码搜索（grep / find / ast analysis）
  - test_agent: 测试执行与诊断

Subagent 有独立的 context，主 agent 只看摘要结果。
"""

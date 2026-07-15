# 任务七：Mini Harness Engineering

> 主大纲见仓库根 [README](../README.md)。本任务建立在任务六 Mini Coding Agent 之上，但第一阶段完全不需要 LLM、GPU 或 API Key。

## 为什么需要这个任务

任务五、六已经包含工具调用、Agent 循环、MCP、Skill、Subagent、测试与简单错误恢复。这些都是 Harness 的组成部分，但学习重点仍然是“Agent 能做什么”。

本任务把视角翻转为：**基座 Agent 固定不变，怎样通过改造它周围的运行系统，让同一个 Agent 更可靠？**

```text
Task / Issue
    |
    v
Harness: policy -> isolated workspace -> agent attempt -> verifier
              ^                                  |
              |--------- structured feedback ----|
                           |
                     durable trace/state
```

完成后，你应该能清楚区分：

- Agent loop：模型决定下一步做什么；
- Harness loop：系统决定允许做什么、如何验证、失败后反馈什么、何时重试或停止；
- 模型能力问题：Agent 不知道怎样修；
- Harness 问题：Agent 无法看见错误、越权执行、状态丢失或把失败误报为成功。

## 学习目标

围绕任务六的 Coding Agent，实现一个最小但完整的可靠执行壳：

1. **策略与权限**：路径 allowlist、命令 allowlist、超时、尝试次数预算；
2. **隔离工作区**：每次运行复制到独立目录，失败尝试不能污染源仓库；
3. **确定性验证器**：以测试退出码而不是 Agent 自报的 `success` 为准；
4. **结构化反馈循环**：把 stdout、stderr、退出码和失败类型反馈给下一次尝试；
5. **持久状态**：将配置、事件、验证结果与最终状态写入文件，可在中断后检查；
6. **可观测性**：使用 JSONL trace 记录每次尝试和验证，而不是只保留聊天文本；
7. **回归与预算**：验证新修复没有破坏旧行为，并在达到预算时可靠停止。

## 准备

```bash
pip install -r requirements.txt
python data/download.py
python eval/run.py
```

`data/download.py` 不访问网络，只生成一个带确定性 Bug 和 pytest 测试的 toy repo。本项目附带一份最小参考实现，第一次运行即可观察完整闭环；建议随后逐个修改组件并完成后面的消融实验。

## 建议目录结构

```text
Project7_harness-engineering/
├── config/harness.json
├── data/
│   ├── download.py
│   └── toy-repo/              # download.py 生成
├── runs/                      # 运行时生成，不提交
├── src/                       # 可运行的最小参考实现
│   ├── policy.py
│   ├── runtime.py
│   ├── state.py
│   └── harness.py
└── eval/
    ├── run.py
    └── tutor_prompt.md
```

## 分阶段实现

### 阶段一：确定性 Harness，不接 LLM

先使用 `eval/run.py` 内置的 Fake Agent。它第一次故意不修 Bug，第二次读取验证反馈后才修复。这样可以把“模型是否聪明”这个变量拿掉，只测试 Harness 本身：

```text
attempt 1 -> pytest failed -> capture evidence -> feedback
attempt 2 -> fix -> pytest passed -> success
```

如果 Fake Agent 都无法稳定完成任务，换成真实 LLM 只会让问题更难定位。

### 阶段二：接入任务六 Coding Agent

为任务六的 `CodingAgent` 写一个 adapter，使它满足：

```python
class AgentBackend:
    def act(self, task: str, runtime, feedback: dict | None) -> dict:
        ...
```

模型只能通过 `runtime` 暴露的安全工具读写文件和运行命令，不能绕过 Harness 直接操作源仓库。

### 阶段三：让 Harness 从失败中改进（进阶）

收集一批 trace，将失败按类型聚类：路径越权、命令被拒、测试失败、超时、预算耗尽。只允许 Agent 提议**有边界的配置或说明修改**，然后在 held-out 任务和 replay 任务上验证；没有稳定提升就拒绝更新。

```text
mine failures -> propose bounded edit -> validate -> accept/reject
```

不要在入门阶段允许 Agent 任意重写验证器、权限系统或评测数据，否则它可能通过“修改考试规则”获得虚假提升。

## 实现约定

| 文件 | 必须导出 |
| --- | --- |
| `src/policy.py` | `PolicyEngine(root, config)`；`check_path(path, operation)`；`check_command(argv)`，越权时抛 `PermissionError` |
| `src/runtime.py` | `SandboxRuntime(root, policy)`；`read_file`、`write_file`、`run_command`；命令必须有 cwd、timeout 和输出捕获 |
| `src/state.py` | `RunStore(runs_dir)`；`create(task, config)`、`append(run_id, event)`、`load(run_id)` |
| `src/harness.py` | `AgentHarness(config, runs_dir, verifier_command)`；`run(agent, task, workspace) -> dict` |

`AgentHarness.run` 返回值至少包含：

```python
{
    "run_id": "...",
    "success": True,
    "attempts": 2,
    "workspace": ".../isolated-run-dir",
    "final_verification": {"exit_code": 0, "stdout": "...", "stderr": "..."},
}
```

## 不可妥协的边界

- 不得使用 `shell=True`；命令以参数列表执行；
- 所有读写路径经过 `Path.resolve()` 后必须仍位于隔离工作区；
- 验证器和 Agent 使用相同的隔离工作区，但 Agent 不能修改验证命令；
- Agent 返回 `success=True` 不代表成功，只有验证器通过才算成功；
- 每次尝试都必须记录事件，即使发生异常；
- 达到 `max_attempts` 或超时后必须停止；
- 源 toy repo 在运行前后内容必须一致。

## 自检

```bash
python data/download.py
python eval/run.py
```

| 测试 | 通过标准 |
| --- | --- |
| `policy_guardrails` | 允许工作区内读写和 pytest；拒绝 `../` 路径逃逸与未授权命令 |
| `run_store_persistence` | 事件追加到 JSONL，重新创建 `RunStore` 后仍可读取 |
| `feedback_retry_loop` | Fake Agent 第一次失败、拿到结构化反馈、第二次成功 |
| `workspace_isolation` | 实际修改发生在独立副本，源 toy repo 保持原始 Bug |
| `trace_observability` | trace 至少包含 run、attempt、verification、finish 事件 |
| `budget_stop` | 永不修复的 Agent 在 `max_attempts` 后停止且不能误报成功 |

## 实验

1. **无 Harness vs 有 Harness**：固定同一 Agent、同一模型和同一任务，对比任务成功率、误报率和源仓库污染率；
2. **反馈消融**：只返回“测试失败”与返回完整结构化错误，比较重试成功率；
3. **验证消融**：相信 Agent 自报 vs 以 pytest 为准，比较 false positive；
4. **状态消融**：只保留聊天上下文 vs 文件持久化，模拟进程中断后能否恢复诊断；
5. **回归测试**：改进 Harness 后，在旧任务 replay 集上检查是否退化；
6. **成本指标**：记录尝试次数、工具调用数、wall time 和 token（接入模型后）。

实验时至少固定随机种子并重复 5 次。Harness 的价值不是让某一次演示更漂亮，而是让一组任务的成功率更稳定、失败更可解释。

## 参考资料

- Mitchell Hashimoto, *My AI Adoption Journey*，其中提出“每当 Agent 犯错，就工程化地避免它再次犯同类错误”；
- OpenAI, *Harness engineering: leveraging Codex in an agent-first world*；
- Lilian Weng, *Harness Engineering for Self-Improvement*；
- Zheng et al., *SEAGym: An Evaluation Environment for Self-Evolving LLM Agents*；
- Anthropic, *Effective harnesses for long-running agents*。

## 时间

约 2-3 周。第一周完成确定性 Harness；第二周接入任务六；第三周选做失败聚类、replay 与受限自改进。

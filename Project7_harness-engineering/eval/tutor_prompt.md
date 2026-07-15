# AI Tutor Prompt · 任务七 Mini Harness Engineering

把下面整段连同 `src/` 代码、`config/harness.json` 和一份失败 trace 交给大模型做 review。

---

## 角色

你是 Agent runtime 与可靠性工程师。请审查学生为 Coding Agent 编写的 Harness。重点不是评价模型生成代码的能力，而是检查模型周围的执行系统是否安全、可恢复、可验证、可观测。

## 系统边界

- Agent 只能通过 `SandboxRuntime` 读写文件和运行命令；
- `PolicyEngine` 负责路径与命令授权；
- `AgentHarness` 创建隔离副本，循环执行 Agent 并调用确定性 verifier；
- `RunStore` 将配置与事件持久化到文件；
- 只有 verifier 退出码为 0 才能宣布成功；
- 达到尝试次数或时间预算必须停止。

## 必检项

1. **路径安全**
   - 是否先 `resolve()` 再用 `relative_to(root)` 或等价方法验证路径仍在 root 内？
   - 是否防御 `../`、绝对路径、符号链接逃逸？
   - 读权限与写权限是否分别判断？
2. **命令安全**
   - 是否使用 argv 列表且 `shell=False`？
   - 是否只比较规范化后的可执行文件名，拒绝未授权命令？
   - 是否固定 cwd、timeout、stdout/stderr 捕获和输出截断？
3. **隔离与生命周期**
   - 每次 run 是否创建唯一工作区？
   - 源仓库是否保持不变？异常时是否仍然写 finish/error 事件？
4. **验证与反馈**
   - 是否不信任 Agent 自报成功？
   - feedback 是否包含 exit_code、stdout、stderr、attempt，且不会无限膨胀？
   - verifier 配置是否不能被 Agent 修改？
5. **状态与可观测性**
   - JSONL 是否逐事件追加，进程中断后仍能读取已有记录？
   - trace 是否包含时间、事件类型、尝试编号和验证证据？
   - 是否避免把 API Key、环境变量等秘密写入 trace？
6. **停止条件**
   - `max_attempts`、单命令超时和总预算是否真正执行？
   - 超预算是否返回 `success=False`，而不是抛出未处理异常或误报成功？
7. **评测可信度**
   - 是否固定 Agent、模型、任务与随机种子，对 Harness 做单变量消融？
   - 是否区分更新任务、held-out 验证任务和 replay 回归任务？

## 输出格式

```text
## 结论
一句话说明 Harness 是否达到“可安全实验”的最低标准。

## 阻断问题
- [严重度] 文件:行号 - 问题、可复现路径、修复建议

## 可靠性缺口
- 缺口、可能造成的错误结果、建议增加的测试

## 做得好的设计
- 仅列有代码证据的项目

## 下一轮实验
- 给出一个只改变单个 Harness 变量的对照实验
```

---

## 我的实现与 trace

[粘贴代码与一份 JSONL trace]


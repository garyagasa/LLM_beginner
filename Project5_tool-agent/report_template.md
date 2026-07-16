# 任务五实验报告：Conan Agent（工具调用 / ReAct）

> 填写说明：把所有 `[待填]` 换成你的内容；不需要的可选节可删。  
> 建议提交到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 时附上本报告 + `eval/result.json` + 1～2 段完整 trace。

| 项目 | 内容 |
|------|------|
| 姓名 / 学号 | `[待填]` |
| 完成日期 | `[待填]` |
| 代码仓库 / 分支 | `[待填]` |
| 自检命令 | `python eval/run.py` |

---

## 1. 实验概述

用 2～4 句话概括你做了什么、主题是什么、最终自检是否通过。

**示例写法（可改写）：**  
本实验实现了一个名侦探柯南主题的 ReAct 工具调用 Agent：五个工具（计算器、Python 沙箱、档案检索、维基百科、柯南 Wiki）+ 手写 Thought/Action/Observation 循环，本地用 vLLM 部署 Qwen2.5 模型，在 28 题任务集上评测关键词命中率。

- 主题目标：`[待填，例如：协助查 APTX4869 / 红黑人物 / 年龄差]`
- 完成模块：M1 工具 `[ ]` / M2 ReAct `[ ]` / M3 错误恢复 `[ ]` / M4 Qwen-Agent `[ ]`
- 一句话结论：`[待填]`

---

## 2. 环境与模型部署

| 项目 | 取值 |
|------|------|
| GPU | `[待填，如 RTX 3090 × 1，CUDA_VISIBLE_DEVICES=1]` |
| 推理引擎 | vLLM / Ollama / 其他：`[待填]` |
| 模型 | `[待填，如 Qwen/Qwen2.5-7B-Instruct]` |
| API | `OPENAI_BASE_URL=` `[待填]` |
| 依赖 | `pip install -r requirements.txt` + `[待填：vllm 版本等]` |
| 最大步数 `REACT_MAX_STEPS` | `[待填]` |

启动命令（可贴实际用的）：

```bash
# [待填] 例如：
# export HF_ENDPOINT=https://hf-mirror.com
# bash scripts/serve_vllm.sh
```

冒烟结果（可选）：

```text
[待填：curl /v1/models 或一句 chat 输出]
```

---

## 3. 系统设计

### 3.1 整体流程

用文字或简图说明 ReAct 循环：

```text
用户任务
  → system prompt（人设 + 工具列表 + few-shot）
  → LLM 生成 Thought / Action / Action Input
  → call_tool → Observation 写回历史
  → … 直至 Final Answer 或步数上限
  → 返回 AgentTrace {steps, final_answer, success}
```

你实际实现里的特殊点：`[待填，如解析失败重试、错误注入钩子]`

### 3.2 工具一览

| 工具名 | 文件 | 作用 | 实现要点（各 1 句） |
|--------|------|------|-------------------|
| `calculator` | `src/tools/calculator.py` | `[待填]` | `[待填：如 AST 白名单]` |
| `python_sandbox` | `src/tools/python_sandbox.py` | `[待填]` | `[待填：受限 builtins / 超时 / stdout]` |
| `file_search` | `src/tools/file_search.py` | `[待填]` | `[待填：越界保护]` |
| `wiki` | `src/tools/wiki.py` | `[待填]` | `[待填]` |
| `conan_wiki` | `src/tools/conan_wiki.py` | `[待填]` | `[待填：Fandom API / User-Agent]` |

`wiki` vs `conan_wiki` 的路由原则（写进 prompt 的那句）：`[待填]`

### 3.3 Prompt 设计

- 人设：`[待填]`
- few-shot 条数与内容概要：`[待填，如：APTX 年龄差用 calculator；查灰原哀用 conan_wiki]`
- 输出格式约束：Thought / Action / Action Input / Final Answer
- 你认为最有效的一条 prompt 技巧：`[待填]`

---

## 4. 实现说明（按模块）

### 4.1 M1 工具

各工具遇到的坑与解决办法（没有可写「顺利」）：

| 工具 | 问题 | 处理 |
|------|------|------|
| calculator | `[待填]` | `[待填]` |
| python_sandbox | `[待填]` | `[待填]` |
| file_search | `[待填]` | `[待填]` |
| wiki / conan_wiki | `[待填]` | `[待填]` |

### 4.2 M2 ReActAgent

- 解析策略（正则 / 按行）：`[待填]`
- Action Input 如何解析 JSON：`[待填]`
- 终止条件：`[待填]`
- trace 里 `tool` / `action` 字段如何填写：`[待填]`

### 4.3 M3 错误恢复（可选）

- 是否实现 `enable_error_injection`：`[是/否]`
- 注入策略与恢复率（若测了）：`[待填]`

### 4.4 M4 Qwen-Agent 对照（可选）

- 是否完成：`[是/否]`
- 手写 ReAct 成功率 vs Qwen-Agent：`[待填]`
- 差异观察：`[待填]`

---

## 5. 自检结果

运行：

```bash
python data/download.py   # 若尚未生成 tasks / 夹具
python eval/run.py
```

| 测试 | 通过？ | 指标 / 备注 |
|------|--------|-------------|
| `tools_individual` | `[待填]` | `[待填：各工具 True / skip]` |
| `multi_tool_success_rate` | `[待填]` | rate=`[待填]`（通过线 > 0.6），n=28 |
| `error_recovery` | `[待填 / skip]` | `[待填]` |

可粘贴 `eval/result.json` 摘要，或附文件。

**按题型粗分（建议自己统计 `details`）：**

| 题型 | 题号范围（约） | 做对 / 总数 | 主要失败原因 |
|------|----------------|-------------|----------------|
| 纯计算 | 如 1,8,11,21–28 | `[待填]` | `[待填]` |
| 档案检索 | 如 3,7,10,12–14,17–18 | `[待填]` | `[待填]` |
| 百科 / 柯南 Wiki | 如 4,5,9,16,19–20 | `[待填]` | `[待填]` |
| 沙箱脚本 | 如 2,6,15,27 | `[待填]` | `[待填]` |
| 多工具组合 | 如 5,9,16,19,20,25 | `[待填]` | `[待填]` |

---

## 6. 案例分析（至少 2 个）

### 6.1 成功案例

- 任务原文：`[待填，从 tasks.json 复制]`
- 调用工具序列：`[待填，如 file_search → calculator]`
- 最终答案：`[待填]`
- 完整 trace（可缩略 Observation）：

```text
Thought: ...
Action: ...
Action Input: ...
Observation: ...
...
Final Answer: ...
```

### 6.2 失败 / 有趣案例

- 任务原文：`[待填]`
- 错在哪（选错工具 / 解析失败 / 幻觉 / 关键词没命中）：`[待填]`
- 若改进 prompt 或解析，预期如何变化：`[待填]`

---

## 7. 对比实验（选做，建议至少 1 项）

从 README「实验建议」里选，或自拟：

| 实验 | 设置 A | 设置 B | 命中率 / 观察 |
|------|--------|--------|----------------|
| 模型尺寸 | 7B | 3B / 其他 | `[待填]` |
| 有无 `conan_wiki` | 仅 wiki | wiki+conan_wiki | `[待填]` |
| few-shot | 0 条 | 1～2 条 APTX 示例 | `[待填]` |
| 手写 vs Qwen-Agent | ReActAgent | Qwen-Agent | `[待填]` |

简要结论：`[待填]`

---

## 8. 收获与不足

**收获（知识点）：**

1. `[待填：Function calling / ReAct / …]`
2. `[待填]`
3. `[待填]`

**不足与下一步：**

1. `[待填]`
2. `[待填]`

---

## 9. 提交清单

- [ ] 本报告（可导出 PDF / 直接贴 Discussion）
- [ ] `eval/result.json`
- [ ] 关键代码路径说明（`src/agent.py`、`src/tools/*`）
- [ ] 至少 1 段完整成功 trace
- [ ] （可选）M4 对比表、AI Tutor review 摘要

---

## 附录 A：目录结构（实现后）

```text
Project5_tool-agent/
├── report_template.md          # 本模版；可另存为 report.md 填写
├── scripts/serve_vllm.sh
├── src/
│   ├── agent.py
│   ├── config.py / llm.py
│   └── tools/
├── data/agent-fixtures/        # 柯南档案柜
├── data/tasks.json             # 28 题
└── eval/result.json
```

## 附录 B：参考

- ReAct: https://arxiv.org/abs/2210.03629
- 柯南中文 Wiki: https://detectiveconan.fandom.com/zh/wiki
- 本仓库 README / `eval/tutor_prompt.md`

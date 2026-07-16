# 任务五：Conan Tool Agent

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

![logo](/assets/logo.png)

实现 ReAct 循环，让 LLM 自主调用工具完成多步任务。外壳是 **毛利侦探事务所情报助手**：查 APTX4869、红黑双方人物、算变小年龄差；内核仍是实践书里的 ReAct + 工具调用。

**预计耗时**：约 2 周

---

## 主题设定

你在实现一个协助 **江户川柯南 / 工藤新一** 的 Agent：


| 情报源           | 工具名              | 用途                                                                                |
| ------------- | ---------------- | --------------------------------------------------------------------------------- |
| 真相计算器         | `calculator`     | APTX 年龄差、4869 相关演算                                                                |
| 阿笠研究所沙箱       | `python_sandbox` | 小脚本、暗号回文、批量计算                                                                     |
| 米花町档案柜        | `file_search`    | 读 `data/agent-fixtures/` 里的红黑 / APTX 速查卡                                          |
| 现实维基百科        | `wiki`           | 现实世界条目（如「柯南·道尔」）                                                                  |
| **柯南中文 Wiki** | `conan_wiki`     | [detectiveconan.fandom.com/zh](https://detectiveconan.fandom.com/zh/wiki) 主线人物与设定 |


人物重心：**红方**（赤井秀一、安室透/降谷零、柯南/灰原等）与 **黑衣组织**（琴酒、贝尔摩德、雪莉/宫野志保等），以及 **APTX4869 变小年龄** 玩法。

离线评测数字以夹具 `aptx4869_dossier.md` 为准（新一 17→7，灰原 18→7），避免动画时间线争议。

---



## 你要完成什么

按模块顺序做；每个源文件里有 `TODO(Mx-…)` 注释，搜 `TODO` 即可定位。


| 阶段     | 模块       | 文件                           | 你要交付什么                                                | 自检                        |
| ------ | -------- | ---------------------------- | ----------------------------------------------------- | ------------------------- |
| **M1** | 五个工具     | `src/tools/*.py`             | 每个工具导出 `TOOL_SCHEMA` + `run(args) -> str`             | `tools_individual`        |
| **M2** | 手写 ReAct | `src/agent.py`               | `ReActAgent.run(task) -> AgentTrace`（柯南人设 + few-shot） | `multi_tool_success_rate` |
| **M3** | 错误恢复     | `src/agent.py` 钩子            | 工具失败时把错误塞进 Observation，agent 能改试                      | `error_recovery`（可选）      |
| **M4** | 框架对照     | `src/qwen_agent_baseline.py` | 用 Qwen-Agent 复现同等能力并对比成功率                             | 手动跑，不进必检                  |


已搭好、**一般不用改**的文件：

- `src/config.py`：模型地址 / 步数等环境变量
- `src/llm.py`：OpenAI 兼容 `chat()` 封装
- `src/tools/__init__.py`：工具注册表 + `call_tool()`

---



## 推荐完成顺序

```text
准备环境 → M1 五个工具 → 跑 tools_individual
        → M2 ReAct 循环 → 跑 multi_tool_success_rate
        → M3 错误恢复（可选）
        → M4 Qwen-Agent 对照（可选）→ 写实验笔记提交
```



### 0. 准备（两个 conda 环境）

本机驱动是 **CUDA 12.6**，不能把最新 `pip install vllm`（常带 **cu130**）装进业务环境。推荐拆开：

| 环境 | 用途 | 安装 |
|------|------|------|
| `llm_pj5` | 写 Agent、跑 `eval/run.py` | `pip install -r requirements.txt` |
| `vllm_pj5` | 只跑推理服务 | `bash scripts/setup_vllm_env.sh` |

```bash
cd Project5_tool-agent

# --- A. Agent 环境 ---
conda activate llm_pj5          # 或自行 conda create -n llm_pj5 python=3.11
pip install -r requirements.txt

# --- B. vLLM 环境（首次，较久；装 cu126 的 torch + vllm 0.8.5）---
bash scripts/setup_vllm_env.sh

# --- C. 起服务（在 tmux 里开着）---
conda activate vllm_pj5
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/serve_vllm.sh
# 另开终端用 llm_pj5 写代码 / 评测
```

客户端默认已指向 vLLM（`src/config.py`）：

```bash
# llm_pj5 里无需再改，除非端口不同
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
export OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct
```

生成评测数据：

```bash
conda activate llm_pj5
python data/download.py
```

冒烟测试（服务起来后，在 `llm_pj5`）：

```bash
curl http://localhost:8000/v1/models
python -c "from src.llm import chat; print(chat([{'role':'user','content':'用一句话介绍江户川柯南'}]))"
```

> 默认 `CUDA_VISIBLE_DEVICES=1`。小模型：`MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/serve_vllm.sh`。



### M1：实现五个工具

每个文件必须导出：


| 符号                       | 含义                                                      |
| ------------------------ | ------------------------------------------------------- |
| `TOOL_SCHEMA`            | OpenAI function calling 格式的 dict（骨架已写好，可润色 description） |
| `run(args: dict) -> str` | 执行工具，**始终返回字符串**（异常也转成 `Error: ...`）                    |



| 文件                  | TODO         | 行为要点                 | 自检期望                             |
| ------------------- | ------------ | -------------------- | -------------------------------- |
| `calculator.py`     | `TODO(M1-a)` | 白名单求值；支持 `sqrt` 等    | `"2 + 3 * 4"` → 含 `14`           |
| `python_sandbox.py` | `TODO(M1-b)` | 受限 exec、捕获 stdout    | `print(sum(range(10)))` → 含 `45` |
| `file_search.py`    | `TODO(M1-c)` | 文件名/内容检索；防越界         | 项目根搜 `README.md` 能命中             |
| `wiki.py`           | `TODO(M1-d)` | 现实维基；中文优先            | 有网时长度 > 50；离线 skip               |
| `conan_wiki.py`     | `TODO(M1-e)` | Fandom 柯南中文 Wiki API | 查「灰原哀」有网时长度 > 50；离线 skip         |


`conan_wiki` API 入口：`https://detectiveconan.fandom.com/zh/api.php`（记得带 User-Agent）。

单工具冒烟：

```bash
python -c "from src.tools.calculator import run; print(run({'expression': '17-7'}))"
python -c "from src.tools.conan_wiki import run; print(run({'query': '灰原哀'})[:200])"
```



### M2：手写 ReAct 循环

编辑 `src/agent.py`：


| TODO         | 函数                         | 做什么                                                        |
| ------------ | -------------------------- | ---------------------------------------------------------- |
| `TODO(M2-a)` | `build_system_prompt()`    | 柯南人设 + 工具列表 + 格式 + few-shot（建议带 APTX 年龄差示例）                |
| `TODO(M2-b)` | `parse_model_output(text)` | 抽出 action / action_input / final_answer                    |
| `TODO(M2-c)` | `ReActAgent.run(task)`     | LLM → 解析 → `call_tool` → Observation → Final Answer / 步数上限 |


`AgentTrace` 约定：

```python
{
  "steps": [
    {
      "thought": "...",
      "action": "conan_wiki",
      "action_input": {"query": "灰原哀"},
      "observation": "...",
      "tool": "conan_wiki",
    },
  ],
  "final_answer": "...",
  "success": True,
}
```

评测在 `data/tasks.json` 的 **28** 题上看 **最终答案关键词命中率 > 60%**（不信任自报 `success`）。大约答对 17 题即过线。

### M3 / M4

同原任务：错误注入钩子可选；Qwen-Agent 对照可选（`python -m src.qwen_agent_baseline`）。

---



## 目录结构

```text
Project5_tool-agent/
├── README.md
├── data/
│   ├── download.py              # 生成柯南 tasks + 档案夹具
│   ├── tasks.json               # download 后生成
│   └── agent-fixtures/          # 10 份主线档案（红黑/APTX/宫野/赤井…）
├── src/
│   ├── agent.py                 # M2 / M3 ← 你写
│   ├── qwen_agent_baseline.py   # M4 ← 可选
│   └── tools/
│       ├── calculator.py        # M1-a
│       ├── python_sandbox.py    # M1-b
│       ├── file_search.py       # M1-c
│       ├── wiki.py              # M1-d 现实维基
│       └── conan_wiki.py        # M1-e 柯南 Fandom ← 新增
└── eval/
    ├── run.py
    └── tutor_prompt.md
```

---



## 实现约定（自检依赖，勿改签名）


| 文件                    | 必须导出                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| `src/tools/{name}.py` | `TOOL_SCHEMA`、`run(args) -> str`；模块名：`calculator` / `python_sandbox` / `file_search` / `wiki` / `conan_wiki` |
| `src/agent.py`        | `class ReActAgent`，含 `run(task: str) -> AgentTrace`                                                          |


---



## 自检

```bash
python eval/run.py
```


| 测试                        | 通过标准                                        |
| ------------------------- | ------------------------------------------- |
| `tools_individual`        | 5 个工具单元测试通过（`wiki` / `conan_wiki` 无网可 skip） |
| `multi_tool_success_rate` | `tasks.json` 关键词命中率 **> 60%**               |
| `error_recovery`          | 默认 skip；实现 M3 后可自行启用                        |


---



## 前置阅读

- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [柯南中文 Wiki](https://detectiveconan.fandom.com/zh/wiki)（查人物时当资料站）
- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)（M4）
- 实践书 v2「ReAct 智能体」一节



## 实验建议

- 手写 ReAct vs Qwen-Agent
- 只开 `wiki` vs 加上 `conan_wiki` 对主线人物题的命中率
- APTX 年龄题：few-shot 里是否示范「先查档案再算」
- 不同模型尺寸的工具选择准确率



## AI Tutor 反馈

`eval/tutor_prompt.md` + 你的 `src/` 代码。

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions)。


建议附上：

- 实验报告：复制 [`report_template.md`](report_template.md) 为 `report.md` 填写
- `eval/result.json`
- 1～2 段完整 ReAct trace（成功 + 失败各一更佳）

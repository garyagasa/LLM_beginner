# 任务三：指令微调与偏好对齐

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

在 **Qwen2.5-0.5B** 上完成 **SFT（监督微调）→ DPO（偏好对齐）** 两阶段训练，亲手实现 LoRA、chat template、loss masking 与 DPO 损失。本任务扩展实践书 v2「监督微调与 LoRA」「偏好对齐：DPO」两节：不用 PEFT/TRL 黑盒，先写通原理，再可选对照官方库。

**预计耗时**：2–3 周（含阅读论文与实验对比）

---

## 你要完成什么

| 阶段 | 模块 | 文件 | 核心交付 |
|------|------|------|----------|
| **M1** | 手写 LoRA | `src/lora.py` | 低秩注入、`forward`、只训 LoRA 参数；可 `merge_lora` |
| **M2** | 对话格式 | `src/chat.py` | Qwen chat template；只对 **assistant** 算 loss |
| **M3** | SFT 训练 | `src/dataset.py` + `train_sft.py` | MOSS 中文多轮对话微调；产出 `ckpt/sft/` |
| **M4** | DPO 对齐 | `src/dpo.py` + `train_dpo.py` | reference model + DPO loss；产出 `ckpt/dpo/` |
| **可选** | 效果对比 | `src/compare.py` | 同一指令上 base / SFT / DPO 生成对比（写进实验报告） |
| **贯通任务五** | 工具格式 SFT | 换 MOSS plugin 数据 | 让模型学会工具调用**输出格式**（任务五再真调 API） |

每个源文件里都有 **`TODO:`** 注释，按顺序实现即可；`eval/run.py` 会按「实现约定」里的函数签名自动导入评测。

---

## 背景与目标

### 为什么做这个任务

预训练模型（base）只会「续写文本」，不会按人类偏好对话。工业界典型路径：

```
Base Model  →  SFT（学会对话格式与任务）  →  DPO/RLHF（对齐人类偏好）
```

本任务在 0.5B 小模型上走通全流程，重点理解：

1. **LoRA**：只训低秩旁路，显存友好（可训参数 < 5%）
2. **Chat template**：Qwen 的 `<|im_start|>` / `` 多轮格式
3. **Loss masking**：user/system 不参与 loss，避免模型学「模仿用户提问」
4. **DPO**：无需单独 reward model，直接用偏好对 `(chosen, rejected)` 优化策略

### 通过标准（自检）

```bash
python eval/run.py
```

| 测试 | 含义 | 通过标准 |
|------|------|----------|
| `lora_param_count` | LoRA 是否足够「轻量」 | 可训练参数占比 **< 5%** |
| `loss_masking` | mask 是否合理 | labels 中 `-100` 占比在 **20%–90%**（user/system 全为 -100） |
| `sft_vs_base` | SFT 是否产出 checkpoint | `ckpt/sft/` **非空**；质量需 `src/compare.py` **手动对比** |

结果写入 `eval/result.json`。`[跳过]` 表示前置未就绪（如模型未下载），不是失败。

---

## 快速开始

### 0. 环境

```bash
cd Project3_sft-dpo
pip install -r requirements.txt

# 验证 GPU（推荐；CPU 仅适合调试小样本）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**推荐配置（8GB 显存）**

| 项目 | 建议 |
|------|------|
| 基座 | Qwen2.5-0.5B（约 1GB） |
| SFT 数据 | MOSS 子集 **1–5 万条** |
| LoRA | `r=8`, `alpha=16`, `q_proj,v_proj` |
| batch | `batch_size=1`, `grad_accum=8` |
| 精度 | `bfloat16` 或 `float16` |

国内下载慢时：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 1. 下载基座模型

```bash
python data/download.py
```

会下载 **Qwen/Qwen2.5-0.5B** 到 `models/Qwen2.5-0.5B/`，并打印 MOSS / DPO 数据的获取命令。

### 2. 下载 SFT 数据（MOSS）

```bash
# 在 Project3_sft-dpo 目录下
huggingface-cli download OpenMOSS-Team/moss-003-sft-data \
  moss-003-sft-no-tools.jsonl.zip \
  --repo-type dataset \
  --local-dir ./data/moss-sft

cd data/moss-sft && unzip moss-003-sft-no-tools.jsonl.zip && cd ../..
```

完整集 110 万条，**不必全下**；训练时用 `--max-samples 10000` 取子集即可。

### 3. （可选）下载 DPO 偏好数据

任选其一，导出为 `data/dpo/train.jsonl`（字段含 `prompt` / `chosen` / `rejected` 或你在 `DPODataset` 里映射的等价名）：

- [hiyouga/DPO-En-Zh-20k](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k) — 中英混合
- [argilla/distilabel-intel-orca-dpo-pairs](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) — 英文
- 或自造：用强模型给 MOSS 回复打偏好标签

### 4. 按里程碑实现代码

建议顺序：

```
src/lora.py  →  src/chat.py  →  src/dataset.py  →  train_sft.py
       →  src/dpo.py  →  train_dpo.py  →  src/compare.py
```

每完成 M1、M2 可先跑自检：

```bash
python eval/run.py   # 前两项通过后再训练
python train_sft.py --max-samples 5000 --epochs 1
python train_dpo.py --data data/dpo/train.jsonl
python src/compare.py
```

---

## 实施细节（各模块要做什么）

### M1：`src/lora.py` — 手写 LoRA

**论文**：[LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)

对线性层 `W ∈ R^{out×in}`，旁路低秩分解：

\[
y = Wx + \frac{\alpha}{r} \cdot B(Ax), \quad A \in R^{in \times r},\; B \in R^{r \times out}
\]

**必须实现**

| 函数 / 类 | 要求 |
|-----------|------|
| `LoRALinear` | 包装 `nn.Linear`；A Kaiming 初始化，B 零初始化；**原 W 冻结** |
| `inject_lora(model, target_modules, r, alpha)` | 替换 `q_proj`、`v_proj` 等；仅 LoRA 参数 `requires_grad=True` |
| `merge_lora(model)` | 合并 LoRA 到原权重，便于导出推理 |
| `lora_state_dict` / `load_lora_state_dict` | 保存 / 加载 `ckpt/sft`、`ckpt/dpo` |

**自检关注点**：`inject_lora` 后 `trainable / total < 0.05`。

---

### M2：`src/chat.py` — Chat template + Loss masking

**Qwen2.5 多轮格式**（示意）：

```text
<|im_start|>system
You are a helpful assistant.
<|im_start|>user
你好
<|im_start|>assistant
你好！很高兴见到你。
```

**必须实现**

| 函数 | 要求 |
|------|------|
| `format_messages(messages)` | `List[{"role","content"}]` → 上述格式字符串 |
| `build_labels(input_ids, messages)` | 与 `input_ids` 同 shape；**user / system / 模板符 → -100**；**assistant 正文 → 真实 token id**（多轮 assistant 都要参与） |

**常见坑**

- 用字符串 `find` 对中文 token 边界不准 → 建议**按 turn 分段 tokenize** 再对齐 span
- 忘记 mask `<|im_start|>assistant` 行本身，只 mask 了 user → mask 比例异常

---

### M3：SFT — `src/dataset.py` + `train_sft.py`

**数据**：[MOSS-003-sft-data](https://huggingface.co/datasets/OpenMOSS-Team/moss-003-sft-data)

MOSS jsonl 中 `chat.turn_*`：`Human` → `user`，`MOSS` → `assistant`。

**训练流程**

1. `format_messages` → tokenize → `build_labels`
2. `inject_lora` → `AdamW` 只优化 LoRA
3. `CrossEntropyLoss`（HF `labels` 传 -100 处自动忽略）
4. 保存 `ckpt/sft/lora.pt`（或你约定的文件名，保证目录非空）

```bash
CUDA_VISIBLE_DEVICES=0 python train_sft.py \
  --max-samples 10000 \
  --epochs 1 \
  --lora-r 8 \
  --grad-accum 8
```

---

### M4：DPO — `src/dpo.py` + `train_dpo.py`

**论文**：[DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)

**损失**（单条样本）：

\[
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)
\]

其中 \(y_w\)=chosen，\(y_l\)=rejected。

**必须做对**

1. **Reference model** = SFT 结束时的策略快照，**全程 freeze**，只做 forward
2. 每条样本 **4 次 forward**：policy(chosen/rejected) + ref(chosen/rejected)
3. 序列 log prob：对非 -100 位置平均（或求和，与论文一致即可，前后统一）
4. 仅更新 **policy** 的 LoRA；保存到 `ckpt/dpo/`

```bash
python train_dpo.py --sft-ckpt ckpt/sft --data data/dpo/train.jsonl --beta 0.1
```

---

### 可选：`src/compare.py`

对固定中文 prompt 打印 **base / SFT / DPO** 生成，附在实验报告或 discussion 提交里，证明 SFT 相对 base 有**可观察差异**。

---

## 实现约定（eval 导入签名）

| 文件 | 必须导出 |
|------|----------|
| `src/lora.py` | `inject_lora(model, target_modules, r, alpha) -> model`、`merge_lora(model) -> model` |
| `src/chat.py` | `format_messages(messages: List[dict]) -> str`；`build_labels(input_ids, messages) -> labels` |
| `ckpt/sft/` | SFT 后的 LoRA 权重目录 |
| `ckpt/dpo/` | DPO 后的 LoRA 权重目录 |

训练脚本 `train_sft.py` / `train_dpo.py` 无强制签名，但需产出上述 checkpoint。

---

## 项目结构

```text
Project3_sft-dpo/
├── README.md                 # 本文件
├── requirements.txt
├── data/
│   ├── download.py           # 下载 Qwen + 打印 MOSS/DPO 提示
│   ├── moss-sft/             # MOSS jsonl（自行下载）
│   └── dpo/                  # DPO 偏好数据（自行准备）
├── models/
│   └── Qwen2.5-0.5B/         # download.py 下载
├── src/
│   ├── lora.py               # M1 TODO
│   ├── chat.py               # M2 TODO
│   ├── dataset.py            # M3 TODO
│   ├── dpo.py                # M4 TODO
│   └── compare.py            # 可选对比脚本
├── train_sft.py              # M3 TODO
├── train_dpo.py              # M4 TODO
├── ckpt/
│   ├── sft/                  # 训练产出
│   └── dpo/
└── eval/
    ├── run.py                # 自检（勿改）
    └── tutor_prompt.md       # 贴给大模型做 code review
```

---

## 前置阅读

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [HF TRL 文档](https://huggingface.co/docs/trl) / [PEFT 文档](https://huggingface.co/docs/peft)（**对照用，本任务先手写**）
- 实践书 v2《大语言模型与智能体》「监督微调与 LoRA」「偏好对齐：DPO」

---

## 实验建议（写进报告）

1. **全量 vs LoRA**：显存占用与生成质量
2. **LoRA rank 消融**：r = 4 / 8 / 16 / 32
3. **灾难性遗忘**：C-Eval 子集上 base vs SFT 准确率
4. **SFT-only vs SFT+DPO**：同一批「刁钻问题」上人工偏好对比
5. **（贯通任务五）** 用 `moss-003-sft-with-tools` 训一版，观察工具 JSON 格式是否更稳定

---

## AI Tutor 反馈

实现过程中可把 `src/lora.py`、`src/chat.py`、`train_sft.py`、`train_dpo.py` 贴给大模型，并使用 `eval/tutor_prompt.md` 中的评审清单。

---

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 提交，建议包含：

- `eval/result.json` 自检结果
- `src/compare.py` 的 base vs SFT（vs DPO）输出样例
- 简短实验结论（LoRA rank、是否 DPO 更好等）

---

## 常见问题

**Q：`eval/run.py` 报 `NotImplementedError`？**  
A：正常，说明还在框架阶段；按 M1→M4 填完 TODO 再跑。

**Q：显存 OOM？**  
减小 `--max-length`、`--max-samples`，保持 `batch_size=1`，增大 `grad_accum`；只对 `q_proj,v_proj` 注入 LoRA；开启 gradient checkpointing（加分项）。

**Q：能否直接用 PEFT / TRL？**  
建议先手写通过自检，再可选分支用 PEFT 对照参数量与 loss 是否一致。

**Q：DPO 数据没有中文？**  
可先用英文集走通流程，或自建 500–2000 条中文偏好对。

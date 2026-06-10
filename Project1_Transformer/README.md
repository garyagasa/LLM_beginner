# 任务一：熟悉 Transformer

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 一句话目标

用约 300 行 PyTorch 从零写一个 Transformer encoder，在 ChnSentiCorp 中文情感二分类上做到 dev 准确率 ≥ 0.80（参考基线约 0.85），并能用注意力热图说清楚模型「在看什么」。

## 任务情境

假装你刚入职某 NLP 团队，组长让你「先把 Transformer 自己撸一遍」。规则：

- 不许调 `nn.MultiheadAttention` 或任何高层封装
- 不许加载预训练模型
- 两周后周会要汇报：dev 准确率 + 几张注意力热图 + 你对 mask、residual、LayerNorm 的理解

## 输入 / 输出

| | 内容 |
|---|---|
| **给你** | ChnSentiCorp 数据（HF 自动拉，约 9.6K 训练样本，二分类）/ PyTorch 2.0+ / 单卡 8GB GPU（CPU 也能跑但慢 ~10×） |
| **交付** | 1. `ckpt/best.pt` 2. `figures/` 下 ≥ 3 张注意力热图 3. `eval/result.json` 4. 一段 200–500 字实验观察文字 |

---

## Definition of Done（必做 5 项）

| 里程碑 | 验收标准 | 对应自检 |
|--------|----------|----------|
| **M1** | 手写 `scaled_dot_product_attention`，与官方实现误差 < 1e-5 | `attention_correctness` |
| **M2** | 手写 `MultiHeadAttention` + `TransformerBlock`，前向形状正确 | `mha_forward` / `mha_padding_mask` / `transformer_block_forward` |
| **M3** | ChnSentiCorp 训练分类器，dev 准确率 ≥ 0.80 | `classifier_accuracy` |
| **M4** | causal mask 不泄漏未来 token；toy LM 能前向 | `causal_mask` / `toy_lm_forward` |
| **M5** | `figures/` 下 ≥ 3 张注意力热图 | `attention_heatmaps` |

---

## Stretch Goals（加分 4 项，任选）

> M1–M5 全部完成后再做。无自动自检，靠 **wandb 曲线 / 准确率表 / 实验记录** 在提交帖里展示。

| 加分项 | 验收标准 | 主要改动文件 | 建议耗时 |
|--------|----------|-------------|----------|
| **S1** | head 数 / 层数消融：≥ 3 组配置 + dev 准确率对比表 | `train.py`（调参运行） | 0.5–1 天 |
| **S2** | 拆掉 residual 或 LayerNorm，记录训练是否还能收敛 | `src/block.py` | 0.5 天 |
| **S3** | dev 准确率 > 0.88（强结果） | `train.py`（调参运行） | 0.5–1 天 |
| **S4** | 绝对 PE 换成 RoPE，与 baseline 对比 dev acc | `src/model.py` | 1–2 天 |

---

## 文件地图（你要改哪些、不用改哪些）

| 文件 | 谁写 | 做什么 |
|------|------|--------|
| `data/download.py` | **只运行** | 下载 ChnSentiCorp → `data/*.parquet` |
| `src/attention.py` | **你实现** | M1 缩放点积注意力 + M2 多头注意力 |
| `src/block.py` | **你实现** | M2 FFN + TransformerBlock（Pre-norm + residual） |
| `src/model.py` | **你实现** | M3 字符 tokenizer、位置编码、分类模型、checkpoint 读写 |
| `train.py` | **只运行**（S1/S3 调参） | M3 训练循环、保存 `ckpt/`、M5 自动生成热图；S1/S3 消融与冲高 |
| `eval/run.py` | **只运行** | 全里程碑自检，写 `eval/result.json` |
| `src/block.py` | **S2 改动** | 临时去掉 residual / LayerNorm 做对比实验 |
| `src/model.py` | **S4 改动** | 实现 RoPE 并与绝对 PE 对比 |

---

## 两周日程（第几天 → 哪个文件 → 做什么）

> 按天推进；每阶段结束跑对应自检命令，通过再进入下一天。

### 第 1 天：环境

| 操作 | 命令 / 文件 |
|------|-------------|
| 安装依赖 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`（GPU）或 `cpu` 版 |
| | `pip install -r requirements.txt` |
| 验证 GPU（可选） | `nvidia-smi`；训练时 `export CUDA_VISIBLE_DEVICES=4` 选空闲卡 |
| 验证 PyTorch | `python -c "import torch; print(torch.cuda.is_available())"` |

**当日产出**：环境可用，能 `import torch`。

---

### 第 2–3 天：数据（M3 前置）

| 操作 | 文件 / 命令 |
|------|-------------|
| 下载数据集 | `python data/download.py` |
| 国内镜像（可选） | `export HF_ENDPOINT=https://hf-mirror.com` 后再下载 |

**当日产出**：`data/train.parquet`、`data/validation.parquet`、`data/test.parquet` 三个文件存在。

---

### 第 4–5 天：M1 — 缩放点积注意力

| 文件 | 要实现什么 |
|------|-----------|
| `src/attention.py` | `scaled_dot_product_attention(Q, K, V, mask=None)` |

**实现要点**（按函数内注释 4 步）：

1. `scores = Q @ K^T / sqrt(d_k)`
2. `mask` 为 True 的位置填 `-inf`（不要用乘 0）
3. `softmax(dim=-1)`
4. `attn_weights @ V`

**当日命令**：

```bash
python eval/run.py M1
# 或
python eval/run.py attention_correctness
```

**通过标准**：`[通过] attention_correctness`，`max_abs_diff < 1e-5`。

**常见坑**：缩放用 `sqrt(d_k)` 不是 `sqrt(d_model)`；`mask` 语义是 True = 屏蔽。

---

### 第 6 天：M2（上）— 多头注意力

| 文件 | 要实现什么 |
|------|-----------|
| `src/attention.py` | `class MultiHeadAttention`：`W_q/W_k/W_v/W_o`、分头、合并、dropout |

**实现要点**：

1. `x` → Q/K/V 线性投影 → reshape 为 `(B, H, T, d_k)`
2. 调用 `scaled_dot_product_attention`
3. 合并头 → `W_o` 输出投影

**当日命令**：

```bash
python eval/run.py mha_forward mha_padding_mask
```

**通过标准**：输出形状 `(B, T, d_model)`；padding mask 不泄漏。

**常见坑**：`view` 前加 `.contiguous()`；`d_model % n_heads == 0`。

---

### 第 7 天：M2（下）— Transformer Block

| 文件 | 要实现什么 |
|------|-----------|
| `src/block.py` | `FeedForward`：Linear → GELU → Dropout → Linear |
| `src/block.py` | `TransformerBlock`：Pre-norm 双 residual |

**结构**（Pre-norm）：

```
x = x + Attention(LayerNorm(x), mask)
x = x + FFN(LayerNorm(x))
```

**当日命令**：

```bash
python eval/run.py M2
```

**通过标准**：`mha_forward`、`mha_padding_mask`、`transformer_block_forward` 全部 `[通过]`。

---

### 第 8–10 天：M3 — 分类模型 + 训练

#### 第 8 天：`src/model.py`（tokenizer + 位置编码）

| 类 / 函数 | 做什么 |
|-----------|--------|
| `CharTokenizer` | `build_from_texts`、`encode`、`decode`、`to_dict` / `from_dict` |
| `PositionalEncoding` | 正弦/余弦 PE，`register_buffer` |

#### 第 9 天：`src/model.py`（分类器）

| 类 / 函数 | 做什么 |
|-----------|--------|
| `TransformerClassifier` | Embedding + PE + N×Block + LayerNorm + [CLS] pooling + Linear |
| `_create_padding_mask` | PAD 位置 mask，形状 `(B, 1, 1, T)` |
| `save_checkpoint` / `load_for_eval` | 保存/加载 model + config + tokenizer |

#### 第 10 天：训练

| 操作 | 命令 |
|------|------|
| 选 GPU | `export CUDA_VISIBLE_DEVICES=4`（选空闲卡，`nvidia-smi` 查看） |
| 开始训练 | `python train.py` |
| 关闭 wandb（可选） | `python train.py --no-wandb` |
| 自定义 run 名 | `python train.py --wandb-run-name d128-l4-seed42` |

**`train.py` 已写好**（AdamW、cosine LR、warmup、early stopping、热图），一般**不用改**，只需调参：

| 超参 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 128 | 卡住可降到 64 |
| `n_heads` | 4 | 必须整除 d_model |
| `n_layers` | 4 | 卡住可降到 2 |
| `lr` | 3e-4 | 震荡太大可降到 1e-4 |
| `batch_size` | 32 | |
| `epochs` | 5 | early stop patience=3 |

**当日命令**：

```bash
python eval/run.py M3
```

**通过标准**：`classifier_accuracy` 中 `accuracy >= 0.80`；生成 `ckpt/best.pt`。

**常见坑**：忘 padding mask；mean pooling 未排除 PAD；lr 过大导致 dev 震荡。

**训练耗时参考**（单张 3090）：约 **5–10 分钟**（含热图）。

---

### 第 11–12 天：M4 — Causal Mask + Toy LM

| 文件 | 要理解 / 验证什么 |
|------|------------------|
| `src/attention.py` | 同一套 attention 加 **上三角 causal mask**（`True` = 未来位置） |
| （概念） | 未来 token 的 V 改变，不应影响过去位置的输出 |

M4 **不要求改新文件**——自检会直接用你的 `scaled_dot_product_attention` 和 `MultiHeadAttention` 测 causal 行为；`eval/run.py` 里还有一个最小 `ToyLM` 做前向验证。

**建议动手实验**（可选，任务二预热）：

- 用 `MultiHeadAttention` + causal mask 搭一个 next-token 小模型
- 观察：改最后一个 token 的输入，前面 logits 不变

**当日命令**：

```bash
python eval/run.py M4
```

**通过标准**：`causal_mask`（泄漏 diff < 1e-6）、`toy_lm_forward`。

---

### 第 13–14 天：M5 — 热图 + 报告 + 总自检

| 操作 | 说明 |
|------|------|
| 热图生成 | `train.py` 训练结束会自动画 3 张图到 `figures/`（正/负/长句） |
| 重新生成热图 | 再跑一次 `python train.py`（会重训）；或自行调用 `train.plot_attention_heatmaps` |
| 看图 | 轴标签为 `[CLS]`、`#1`、`#2`…（`#i` = 第 i 个字符位置） |
| 写报告 | 200–500 字：模型关注了哪些词、消融发现了什么 |

**当日命令**：

```bash
python eval/run.py M5      # 检查 figures/ ≥ 3 张 png
python eval/run.py         # 全量自检，生成 eval/result.json
python eval/run.py --list   # 查看所有测试项
```

**通过标准**：`attention_heatmaps` 的 `count >= 3`；`eval/result.json` 中 M1–M5 相关项均为 `[通过]`。

---

## 加分日程（第 15 天起，可选）

M1–M5 完成后，按兴趣选做。建议用 wandb 给每组实验起名，方便写报告。

### S1 — 层数 / 头数消融（第 15–16 天）

| 操作 | 说明 |
|------|------|
| **改什么** | 一般**不用改代码**，只改 `train.py` 命令行超参 |
| **跑几组** | 至少 3 组，其余超参保持一致（`seed`、`lr`、`epochs` 相同） |
| **记录什么** | 每组最终 `dev_acc`、参数量、训练时长 |

**推荐三组（示例）**：

```bash
export CUDA_VISIBLE_DEVICES=4

# A：小模型（baseline 对照）
python train.py --d_model 64 --n_layers 2 --n_heads 4 \
  --wandb-run-name s1-small-l2-d64

# B：默认配置
python train.py --d_model 128 --n_layers 4 --n_heads 4 \
  --wandb-run-name s1-baseline-l4-d128

# C：更深 / 更宽
python train.py --d_model 192 --n_layers 6 --n_heads 4 --d_ff 768 \
  --wandb-run-name s1-large-l6-d192
```

**交付**：一张准确率表（markdown 或 wandb 截图），例如：

| 配置 | d_model | n_layers | n_heads | dev_acc |
|------|---------|----------|---------|---------|
| small | 64 | 2 | 4 | ? |
| baseline | 128 | 4 | 4 | ? |
| large | 192 | 6 | 4 | ? |

**验收**：`python eval/run.py M3` 至少一组 ≥ 0.80；表里 ≥ 3 行且有简要结论（如「4 层比 2 层好，6 层提升有限」）。

---

### S2 — 拆 residual 或 LayerNorm（第 16–17 天）

| 操作 | 说明 |
|------|------|
| **改什么** | `src/block.py` 的 `TransformerBlock.forward` |
| **实验 A** | 去掉 residual：`x = attention(norm1(x))`（不加 `x +`） |
| **实验 B** | 去掉 LayerNorm：直接用 `x` 进 attention / FFN |
| **对比** | 与完整 Pre-norm block 各训一轮，看 loss 是否发散、dev acc 差多少 |

**实现提示**（在 `TransformerBlock` 加开关，避免改一份代码无法恢复）：

```python
# block.py 示例：__init__ 增加 use_residual=True, use_layernorm=True
# forward 里按 flag 分支
```

**命令**：

```bash
# 完整 block（对照）
python train.py --wandb-run-name s2-full-block

# 改完 block 后
python train.py --wandb-run-name s2-no-residual
# 或
python train.py --wandb-run-name s2-no-layernorm
```

**交付**：两组 wandb 曲线 + 一句话结论（如「去掉 residual 后 epoch 2 loss 爆炸 / dev 下降 15%」）。

**注意**：做完 S2 记得**恢复** `block.py` 到正常 Pre-norm 版本，否则 M2 自检和后续训练会受影响。

---

### S3 — 冲高 dev 准确率 > 0.88（第 17–18 天）

| 操作 | 说明 |
|------|------|
| **改什么** | 主要调 `train.py` 超参；模型结构可微调 |
| **起点** | 你 baseline 已到 ~0.82–0.85 时再冲，否则先保证 M3 |

**可尝试的组合**（每次只动 1–2 个变量）：

```bash
# 更多 epoch + 略低 lr
python train.py --lr 2e-4 --epochs 10 --dropout 0.15 \
  --wandb-run-name s3-lr2e4-e10

# 略大模型
python train.py --d_model 192 --n_layers 4 --n_heads 4 --d_ff 768 \
  --lr 2e-4 --epochs 8 --wandb-run-name s3-d192

# 更大 batch（梯度更稳）
python train.py --batch_size 64 --lr 2e-4 --epochs 8 \
  --wandb-run-name s3-bs64
```

**验收**：

```bash
python eval/run.py M3   # accuracy > 0.88
```

**现实预期**：字符级小 Transformer 在 9.6K 样本上 **0.88 有难度**；冲到 0.85+ 已经不错，可在报告里写清楚尝试了哪些配置。

---

### S4 — RoPE 替换绝对位置编码（第 18–20 天）

| 操作 | 说明 |
|------|------|
| **改什么** | `src/model.py`：新增 `RotaryPositionalEncoding`（或 `apply_rope`），在 `TransformerClassifier` 里替换 `PositionalEncoding` |
| **不改什么** | `attention.py` / `block.py` 主体逻辑可保持不变；RoPE 通常作用在 Q/K 上 |

**实现步骤建议**：

1. 在 `src/model.py` 实现 RoPE（旋转 Q/K，按 head_dim 配对 sin/cos）
2. `TransformerClassifier.__init__` 用 flag 或新类切换 PE 类型
3. 分别训练 **绝对 PE（baseline）** 与 **RoPE** 各一轮，超参相同

**命令**：

```bash
# baseline（已有 PositionalEncoding）
python train.py --wandb-run-name s4-abs-pe

# 换成 RoPE 后
python train.py --wandb-run-name s4-rope
# （若你加了 --pe_type rope 之类的参数，这里写上）
```

**交付**：对比表（绝对 PE vs RoPE 的 dev_acc、收敛速度）+ 简短分析（RoPE 是否更适合长序列——本任务 `max_len=200`，差异可能不大，说清观察即可）。

---

## 日程总览（一表速查）

| 天数 | 里程碑 | 主要文件 | 核心操作 | 自检命令 |
|------|--------|----------|----------|----------|
| 1 | 环境 | `requirements.txt` | 安装 PyTorch + 依赖 | — |
| 2–3 | 数据 | `data/download.py` | 下载 parquet | — |
| 4–5 | M1 | `src/attention.py` | `scaled_dot_product_attention` | `python eval/run.py M1` |
| 6 | M2 | `src/attention.py` | `MultiHeadAttention` | `python eval/run.py mha_forward` |
| 7 | M2 | `src/block.py` | `FeedForward` + `TransformerBlock` | `python eval/run.py M2` |
| 8 | M3 | `src/model.py` | `CharTokenizer` + `PositionalEncoding` | — |
| 9 | M3 | `src/model.py` | `TransformerClassifier` + checkpoint | — |
| 10 | M3 | `train.py`（运行） | `python train.py` → `ckpt/best.pt` | `python eval/run.py M3` |
| 11–12 | M4 | `src/attention.py`（理解） | causal mask 验证 | `python eval/run.py M4` |
| 13–14 | M5 | `figures/`（产出） | 热图 + 实验报告 | `python eval/run.py` |
| 15–16 | **S1** | `train.py`（调参） | 层数/头数消融 ≥3 组 | `eval/run.py M3` + 准确率表 |
| 16–17 | **S2** | `src/block.py` | 拆 residual 或 LayerNorm | wandb 曲线对比 |
| 17–18 | **S3** | `train.py`（调参） | 冲高 dev > 0.88 | `eval/run.py M3` |
| 18–20 | **S4** | `src/model.py` | RoPE 替换绝对 PE | 对比实验 + 分析 |

---

## 实现约定

`eval/run.py` 会自动检测以下接口；接口对上就能跑自检：

| 文件 | 必须导出 |
|------|----------|
| `src/attention.py` | `scaled_dot_product_attention(Q, K, V, mask=None)` — Q/K/V 形状 `(B, H, T, D)`；`mask` 广播到此，`True` = 被屏蔽 |
| `src/model.py` | `class TransformerClassifier` + `load_for_eval(ckpt_path) -> (model, tokenize_fn)` |
| `ckpt/best.pt` | 训练产物（`train.py` 自动生成） |

接口可以改，但改了请同步调整 `eval/run.py`。

---

## 自检

```bash
python eval/run.py              # 全部测试
python eval/run.py --list       # 列出所有测试
python eval/run.py M1           # 按里程碑
python eval/run.py M2 M4        # 组合
python eval/run.py mha_forward  # 按测试名
```

| 测试 | 通过标准 | DoD |
|------|----------|-----|
| `attention_correctness` | 与 `F.scaled_dot_product_attention` 误差 < 1e-5 | M1 |
| `mha_forward` | MHA 输出形状正确、数值有限 | M2 |
| `mha_padding_mask` | PAD 位改动不影响非 PAD 输出 | M2 |
| `transformer_block_forward` | Block 输出形状与输入一致 | M2 |
| `classifier_accuracy` | dev 准确率 ≥ 0.80 | M3 |
| `causal_mask` | 未来 V 改动后过去位置输出不变 | M4 |
| `toy_lm_forward` | 最小 causal LM 前向通过 | M4 |
| `attention_heatmaps` | `figures/` 下 ≥ 3 张 `.png` | M5 |

结果写入 `eval/result.json`，提交时附上。

---

## AI Tutor 反馈

把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 整段贴给 Claude / Qwen / DeepSeek，连同你的代码。模型会按统一格式（必检 / 加分 / 优先级）给你针对性 review。

---

## 前置阅读（非必需）

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- NNDL2 第 8 章「注意力机制」「自注意力」「Transformer 模型」三节

---

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 「llm-beginner 实践成果」分类发帖，附：

1. 你的 fork 仓库链接
2. `eval/result.json` 内容（贴文本即可）
3. DoD checklist 勾选状态（M1–M5；做了加分项则注明 S1–S4）
4. ≥ 3 张注意力热图
5. 200–500 字实验观察（含消融/加分结论更佳）
6. （若做了 S1/S3）准确率对比表或 wandb 项目链接；（若做了 S2/S4）改动说明与对比结果

---

## 卡住怎么办

| 情况 | 建议 |
|------|------|
| M3 训练 3 天仍 < 0.80 | 缩模型：`python train.py --d_model 64 --n_layers 2` |
| 显存不足 | 减小 `--batch_size` 或 `--max_len` |
| GPU 被占满 | `nvidia-smi` 找空闲卡，`export CUDA_VISIBLE_DEVICES=<id>` |
| 数据集下载失败 | `export HF_ENDPOINT=https://hf-mirror.com` 后重跑 `data/download.py` |
| 热图中文乱码 | 当前热图轴标签已改为英文 `#i` 编号，无需中文字体 |
| S3 怎么都过不了 0.88 | 0.85+ 已属良好；报告里写清尝试过的超参即可 |
| S2 改完 block 自检失败 | 恢复 Pre-norm + 双 residual 后再跑 `eval/run.py M2` |
| S4 RoPE 实现复杂 | 先读懂绝对 PE，再只给 Q/K 加旋转；可参考 Llama 公式 |

**必做**约 **2 周**（M1–M5）；**加分**另留 **3–5 天**（S1–S4 任选）。

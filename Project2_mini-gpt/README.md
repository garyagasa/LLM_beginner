# 任务二：从零实现 mini-GPT

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 目标

用 PyTorch 从零搭一个 decoder-only mini-GPT，在中文网页语料上预训练并自回归生成。**扩展**实践书 v2「nanoGPT 模型」的带读：加入 BPE、RoPE、KV cache。

## 推荐配置（RTX 3090 × 1）


| 项目      | 选择                                          |
| ------- | ------------------------------------------- |
| 数据集     | **SkyPile 50 万条**（约 1.3 GB，train + dev）     |
| PyTorch | `2.12.1+cu126`（驱动 560 / CUDA 12.6，勿装 cu130） |
| 默认模型    | 4 层 / 128 dim / 512 vocab → ~0.9M 参数        |
| 预计耗时    | 下载 10～30 min · BPE 5～15 min · 训练 2～5 h      |


> **不要**用 500 万条：BPE 阶段会 OOM（当前 `tokenizer.py` 实现会把整份语料载入内存）。

---

## 快速开始

### 0. 环境

```bash
conda activate llm_pj2
pip install -r requirements.txt

# 验证 GPU（应输出 True）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.12.1+cu126 True
```

若 `cuda.is_available()` 为 `False`，说明 PyTorch CUDA 版本与驱动不匹配。当前机器驱动 560 对应 CUDA 12.6，应安装：

```bash
pip uninstall -y torch cuda-toolkit nvidia-cuda-runtime nvidia-cudnn-cu13  # 及其他 nvidia-* cu13 包
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
```

### 1. 下载数据（SkyPile 50 万条）

```bash
cd Project2_mini-gpt

# 国内镜像（可选，下载 HuggingFace 慢时使用）
export HF_ENDPOINT=https://hf-mirror.com

# 下载并生成 train.txt / dev.txt / dataset_info.json
python data/download.py --dataset skypile --num-samples 500000
```

脚本会 **streaming** 抽取前 50 万条，按 9:1 切分为 train / dev，直接写入 `data/`。若之前下过 500 万条，此命令会 **覆盖** 旧文件。

下载完成后可确认：

```bash
ls -lh data/train.txt data/dev.txt
cat data/dataset_info.json
# 期望 num_samples ≈ 500000，train ≈ 1.2 GB，dev ≈ 130 MB
```

### 2. 训练

```bash
# 指定空闲 GPU（多卡机器上避免占满已被占用的卡）
CUDA_VISIBLE_DEVICES=0 python train.py --device cuda

# 不上传 wandb
CUDA_VISIBLE_DEVICES=0 python train.py --device cuda --no-wandb
```

### 3. 自检

```bash
python eval/run.py
```

---

## 任务清单

按顺序完成，打勾即过关。

### 阶段 A · 跑通（~30 min）

- [x] 环境：`torch 2.12.1+cu126`，`cuda.is_available() == True`
- [ ] 先用唐诗验证代码链路（可选但推荐）：
  ```bash
  python data/download.py --dataset poetry
  python train.py --device cuda --epochs 5 --no-wandb
  python eval/run.py
  ```
- [x] 下载 **SkyPile 50 万条**（见上文命令）

### 阶段 B · 核心实现（`src/`）


| #   | 模块                          | 文件                 | 要点                                                              |
| --- | --------------------------- | ------------------ | --------------------------------------------------------------- |
| 1   | BPE tokenizer               | `src/tokenizer.py` | 手写 merge；字节级 UTF-8 fallback；`<|pad|>` `<|bos|>` `<|eos|>`       |
| 2   | RoPE                        | `src/rope.py`      | Q/K 旋转；freq 表可复用                                                |
| 3   | Causal attention + KV cache | `src/attention.py` | 上三角 mask；K/V 在 seq 维拼接；cache 有长度上限                              |
| 4   | Decoder + 完整模型              | `src/model.py`     | `forward` / `generate`；暴露 `block_size`                          |
| 5   | 采样策略                        | `src/sampling.py`  | greedy / top-k / top-p / temperature                            |
| 6   | 训练循环                        | `train.py`         | next-token CE loss；AdamW；warmup + cosine；grad clip；train/dev 分离 |


### 阶段 C · 训练与交付

- [ ] 在 SkyPile 50 万上完整训练（默认 20 epoch）
- [ ] 产出 `ckpt/tokenizer.json`、`ckpt/best.pt`
- [ ] `python eval/run.py` 三项自检全部通过
- [ ] （可选）把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 贴给 AI 做 code review
- [ ] 到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 提交成果

### 自检通过标准

| 测试 | 通过标准 |
|------|----------Tag
| `tokenizer_roundtrip` | encode → decode 还原中文（除已知 UTF-8 边界 case） |
| `kv_cache_equivalence` | 开/关 KV cache 的 logits 一致（误差 < 1e-4） |
| `perplexity_on_dev` | dev 困惑度 < `dataset_info.json` 中的阈值（SkyPile 默认 < 100） |

---

## 数据档位参考


| 数据集                  | 命令                                       | 规模      | 适用场景            |
| -------------------- | ---------------------------------------- | ------- | --------------- |
| 唐诗 quick-start       | `--dataset poetry`                       | ~48 KB  | 5 分钟跑通代码        |
| **SkyPile 50 万（推荐）** | `--dataset skypile --num-samples 500000` | ~1.3 GB | 3090 正式训练       |
| SkyPile 10 万         | `--dataset skypile`                      | ~280 MB | 快速实验            |
| TinyStories          | `--dataset tinystories`                  | 百 MB 级  | 英文叙事实验          |
| SkyPile 500 万        | `--num-samples 5000000`                  | ~13 GB  | ❌ BPE 阶段 OOM，勿用 |


脚本统一生成 `data/train.txt`、`data/dev.txt`、`data/dataset_info.json`。

---

## 实现约定


| 文件                    | 必须导出                                                                              |
| --------------------- | --------------------------------------------------------------------------------- |
| `src/tokenizer.py`    | `class BPETokenizer` 含 `encode`、`decode`、`vocab_size`、`from_pretrained`           |
| `src/model.py`        | `class MiniGPT` 含 `forward`、`generate`、`block_size`/`max_seq_len`、`load_for_eval` |
| `ckpt/tokenizer.json` | 训练好的 BPE 词表                                                                       |
| `ckpt/best.pt`        | dev 困惑度最优 checkpoint                                                              |


---

## 前置阅读

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [TinyStories 原论文](https://arxiv.org/abs/2305.07759)
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864)
- NNDL2 第 8 章「现代 Transformer 的常见优化」
- 实践书 v2《大语言模型与智能体》「nanoGPT 模型」「预训练循环」「解码 / 采样策略」

## 进阶实验（可选）

- 参数量扫描（10M / 50M / 100M）vs 困惑度
- KV cache 开/关的推理速度对比
- 绝对位置编码 vs RoPE 长序列外推

## 时间

约 3 周（含实现 + SkyPile 50 万训练 + 自检）。
"""M3：SFT 训练 —— MOSS 数据 + LoRA + loss masking。

用法：
    python data/download.py                          # 下载基座模型 + 查看数据获取命令
    # 下载 MOSS SFT 数据（见 README）
    python train_sft.py                              # 默认配置
    python train_sft.py --max-samples 10000 --epochs 1
    python train_sft.py --no-wandb

输出：
    ckpt/sft/          LoRA 权重（lora_state_dict）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.lora import inject_lora, lora_state_dict
from src.dataset import SFTDataset

DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B"
DEFAULT_CKPT = ROOT / "ckpt" / "sft"


def parse_args():
    p = argparse.ArgumentParser(description="SFT on MOSS with hand-written LoRA")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--data", type=str, default=str(ROOT / "data" / "moss-sft" / "moss-003-sft-no-tools.jsonl"))
    p.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--max-samples", type=int, default=10000, help="MOSS 子集大小，8GB 显存建议 1-5 万")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16)
    p.add_argument("--target-modules", type=str, default="q_proj,v_proj",
                   help="逗号分隔，可扩展为 q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--batch-size", type=int, default=1, help="显存不足时保持 1，配合 gradient_accumulation")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="llm-beginner-p3-sft")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device, grad_accum: int) -> float:
    """单个 epoch 的 SFT 训练循环。"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    # TODO: 遍历 DataLoader
    #   1. batch 移到 device
    #   2. outputs = model(input_ids=..., attention_mask=..., labels=labels)
    #   3. loss = outputs.loss / grad_accum；backward
    #   4. 每 grad_accum 步 optimizer.step() + zero_grad
    raise NotImplementedError("TODO: 实现 train_one_epoch")

    return total_loss / max(len(loader), 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if not Path(args.model).exists():
        sys.exit(f"[错误] 基座模型不存在: {args.model}\n请先运行: python data/download.py")

    # TODO: 加载 tokenizer 与 model
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device)

    # TODO: inject_lora
    # target = [m.strip() for m in args.target_modules.split(",")]
    # inject_lora(model, target_modules=target, r=args.lora_r, alpha=args.lora_alpha)

    # TODO: SFTDataset + DataLoader
    # dataset = SFTDataset(args.data, tokenizer=tokenizer, max_length=args.max_length, max_samples=args.max_samples)
    # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # TODO: optimizer 只传 requires_grad=True 的参数
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # TODO: 训练循环 + 保存 ckpt
    # Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    # torch.save(lora_state_dict(model), Path(args.ckpt_dir) / "lora.pt")

    print("TODO: 完成 train_sft.py 后可运行 python eval/run.py 检查 lora_param_count / loss_masking / sft_vs_base")


if __name__ == "__main__":
    main()

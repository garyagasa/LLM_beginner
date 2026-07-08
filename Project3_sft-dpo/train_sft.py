"""M3：SFT 训练 —— MOSS 数据 + LoRA + loss masking。"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.lora import inject_lora, lora_state_dict
from src.dataset import SFTDataset, pad_collate
from src.wandb_utils import init_wandb, wandb_log, wandb_finish

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
    p.add_argument("--log-every", type=int, default=10, help="每 N step 记录一次 wandb")
    p.add_argument("--no-wandb", action="store_true", help="禁用 Weights & Biases 日志")
    p.add_argument("--wandb-project", type=str, default="llm-beginner-p3-sft")
    p.add_argument("--wandb-run-name", type=str, default=None, help="run 名称，默认自动生成")
    p.add_argument("--wandb-entity", type=str, default=None, help="wandb 团队/用户名，默认用已登录账号")
    return p.parse_args()


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_accum: int,
    *,
    epoch: int = 1,
    log_every: int = 10,
    global_step: int = 0,
    lr: float = 2e-4,
) -> tuple[float, int]:
    """单个 epoch 的 SFT 训练循环。返回 (avg_loss, 更新后的 global_step)。"""
    model.train()
    total_loss = 0.0
    num_steps = 0
    optimizer.zero_grad()

    progress = tqdm(loader, desc=f"SFT epoch {epoch}")
    for step, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            step_loss = loss.item() * grad_accum
            total_loss += step_loss
            num_steps += 1
            progress.set_postfix(loss=f"{step_loss:.4f}")

            if global_step % log_every == 0:
                wandb_log(
                    {
                        "train/loss": step_loss,
                        "train/epoch": epoch,
                        "train/lr": lr,
                    },
                    step=global_step,
                )

    return total_loss / max(num_steps, 1), global_step


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if not Path(args.model).exists():
        sys.exit(f"[错误] 基座模型不存在: {args.model}\n请先运行: python data/download.py")

    dtype = torch.bfloat16 if (
        args.device.startswith("cuda")
        and torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    ) else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(args.device)

    target = [m.strip() for m in args.target_modules.split(",")]
    inject_lora(model, target_modules=target, r=args.lora_r, alpha=args.lora_alpha)

    dataset = SFTDataset(
        args.data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    collate = partial(pad_collate, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
    )

    init_wandb(args)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_loss = float("inf")
    avg_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss, global_step = train_one_epoch(
            model,
            loader,
            optimizer,
            args.device,
            args.grad_accum,
            epoch=epoch,
            log_every=args.log_every,
            global_step=global_step,
            lr=args.lr,
        )
        wandb_log({"train/epoch_loss": avg_loss, "train/epoch": epoch}, step=global_step)
        best_loss = min(best_loss, avg_loss)
        print(f"[SFT] epoch {epoch} avg_loss={avg_loss:.4f}")

    ckpt_path = Path(args.ckpt_dir) / "lora.pt"
    torch.save(lora_state_dict(model), ckpt_path)
    print(f"[SFT] 已保存 LoRA 权重到 {ckpt_path}")

    wandb_finish(
        summary={
            "final_train_loss": avg_loss,
            "best_train_loss": best_loss,
            "total_steps": global_step,
        }
    )


if __name__ == "__main__":
    main()
